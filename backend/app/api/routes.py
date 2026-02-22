"""API routes - POST /query endpoint per API_CONTRACT.md spec."""
import hashlib
import json
import logging
import time
from functools import lru_cache
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.router.classifier import classify
from backend.app.rag.retriever import retriever
from backend.app.llm.groq_client import generate, generate_stream
from backend.app.evaluator.output_checker import evaluate
from backend.app.config import TOP_K_SIMPLE, TOP_K_COMPLEX
from backend.app.memory.conversation import (
    get_or_create_conversation_id,
    get_history,
    add_turn,
)
from backend.app.memory.cache import query_cache

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Helper functions ---

def _generate_rationale(route_result: dict) -> str:
    """Generate a human-readable rationale for the routing decision."""
    classification = route_result["classification"]
    signals = route_result.get("signals", {})
    score = route_result.get("score", 0)
    
    if "greeting_override" in signals:
        return "Greeting detected - using fast model for quick response"
    
    if classification == "simple":
        reasons = []
        if score == 0:
            reasons.append("Short, straightforward query")
        if score == 1:
            reasons.append("Basic factual question")
        return f"Simple query (score={score}) - {', '.join(reasons) if reasons else 'no complex signals detected'}"
    
    # Complex classification - explain why
    reasons = []
    if "complex_keywords" in signals:
        kws = signals["complex_keywords"][:2]  # Show first 2
        reasons.append(f"Contains complex keywords: {', '.join(kws)}")
    if "domain_complex" in signals:
        reasons.append("Multiple domain topics requiring cross-doc reasoning")
    elif "domain_keywords" in signals:
        reasons.append(f"Domain keywords: {', '.join(signals['domain_keywords'][:2])}")
    if "long_query" in signals:
        reasons.append(f"Long query ({signals['long_query']} words)")
    if "multiple_questions" in signals:
        reasons.append(f"{signals['multiple_questions']} questions")
    if "complaint_markers" in signals:
        reasons.append("Complaint/frustration detected - needs careful handling")
    if "subordinate_clauses" in signals:
        reasons.append("Complex sentence structure")
        
    return f"Complex query (score={score}) - " + "; ".join(reasons[:3])  # Limit to 3 reasons


# --- Request / Response models ---

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class TokenUsage(BaseModel):
    input: int
    output: int


class Metadata(BaseModel):
    model_used: str
    classification: str
    rationale: str
    tokens: TokenUsage
    latency_ms: int
    chunks_retrieved: int
    evaluator_flags: list[str] = Field(default_factory=list)


class Source(BaseModel):
    document: str
    page: Optional[int] = None
    relevance_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    metadata: Metadata
    sources: list[Source]
    conversation_id: str
    warning: Optional[str] = None


# --- Main query endpoint ---

@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Process a user query through the full RAG + Router + Evaluator pipeline."""
    start_time = time.time()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # 1. Get/create conversation
    conv_id = get_or_create_conversation_id(request.conversation_id)

    # 2. Check Semantic Cache
    cached_data = query_cache.get(question)
    if cached_data:
        logger.info(f"Serving cached response for: '{question}'")
        add_turn(conv_id, question, cached_data["answer"])
        # Ensure we return a distinct conversation ID for proper trace
        return QueryResponse(
            answer=cached_data["answer"],
            metadata=Metadata(**cached_data["metadata"]),
            sources=[Source(**s) for s in cached_data["sources"]],
            conversation_id=conv_id,
            warning=cached_data.get("warning")
        )

    # 3. Classify query → route to model
    route_result = classify(question)
    classification = route_result["classification"]
    model = route_result["model_used"]
    
    # Generate human-readable rationale
    rationale = _generate_rationale(route_result)

    # 4. Retrieve relevant chunks (use fewer for simple queries to save tokens)
    top_k = TOP_K_SIMPLE if classification == "simple" else TOP_K_COMPLEX
    search_results = retriever.search(question, top_k=top_k)
    
    # 3b. Context Compression (Selective Context)
    # Compress the grabbed chunks to only keep query-relevant sentences
    from backend.app.rag.compressor import compressor
    context_chunks = [compressor.compress(chunk, question).text for chunk, score in search_results]

    # Build sources list
    sources = []
    for chunk, score in search_results:
        sources.append(Source(
            document=chunk.filename,
            page=chunk.page_number,
            relevance_score=round(score, 4),
        ))

    # 4. Get conversation history
    history = get_history(conv_id)

    # 5. Generate LLM response
    answer, token_usage, latency_ms = generate(
        model=model,
        query=question,
        context_chunks=context_chunks,
        conversation_history=history,
    )

    # 6. Evaluate output
    flags, warning = evaluate(
        answer=answer,
        chunks_retrieved=len(search_results),
        sources=[s.model_dump() for s in sources],
        query=question,
        classification=classification,
    )

    # 7. Record conversation turn
    add_turn(conv_id, question, answer)

    # 8. Calculate total latency (includes retrieval + routing overhead)
    total_latency = int((time.time() - start_time) * 1000)

    # 9. Log routing decision (required format)
    log_entry = {
        "query": question,
        "classification": classification,
        "model_used": model,
        "tokens_input": token_usage["input"],
        "tokens_output": token_usage["output"],
        "latency_ms": total_latency,
        "router_signals": route_result["signals"],
        "evaluator_flags": flags,
    }
    logger.info(f"ROUTING_LOG: {json.dumps(log_entry)}")

    # 10. Cache the response dict
    cache_data = {
        "answer": answer,
        "metadata": {
            "model_used": model,
            "classification": classification,
            "rationale": rationale,
            "tokens": token_usage,
            "latency_ms": total_latency,
            "chunks_retrieved": len(search_results),
            "evaluator_flags": flags,
        },
        "sources": [s.model_dump() for s in sources],
        "warning": warning,
    }
    query_cache.set(question, cache_data)

    # 11. Build final Pydantic response
    response = QueryResponse(
        answer=answer,
        metadata=Metadata(**cache_data["metadata"]),
        sources=sources,
        conversation_id=conv_id,
        warning=warning,
    )

    return response


# --- Streaming endpoint (Bonus) ---

class StreamQueryRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


@router.post("/query/stream")
async def handle_query_stream(request: StreamQueryRequest):
    """Stream LLM response token-by-token via Server-Sent Events."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    conv_id = get_or_create_conversation_id(request.conversation_id)

    # Check Semantic Cache first
    cached_data = query_cache.get(question)
    if cached_data:
        logger.info(f"Serving cached stream response for: '{question}'")
        add_turn(conv_id, question, cached_data["answer"])
        async def cached_stream():
            # Send the cached answer as a single token event for speed
            yield f"data: {json.dumps({'type': 'token', 'content': cached_data['answer']})}\n\n"
            # Send the done event
            done_data = {
                "type": "done",
                "metadata": cached_data["metadata"],
                "sources": cached_data["sources"],
                "conversation_id": conv_id,
                "warning": cached_data.get("warning"),
            }
            yield f"data: {json.dumps(done_data)}\n\n"
        
        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    route_result = classify(question)
    model = route_result["model_used"]
    rationale = _generate_rationale(route_result)

    # Use classification-based TOP_K for retrieval
    top_k = TOP_K_SIMPLE if route_result["classification"] == "simple" else TOP_K_COMPLEX
    search_results = retriever.search(question, top_k=top_k)
    
    # Compress the grabbed chunks to only keep query-relevant sentences
    from backend.app.rag.compressor import compressor
    context_chunks = [compressor.compress(chunk, question).text for chunk, score in search_results]

    sources = [
        {
            "document": chunk.filename,
            "page": chunk.page_number,
            "relevance_score": round(score, 4),
        }
        for chunk, score in search_results
    ]

    history = get_history(conv_id)

    def event_stream():
        full_answer = ""
        for event in generate_stream(
            model=model,
            query=question,
            context_chunks=context_chunks,
            conversation_history=history,
        ):
            if event["type"] == "token":
                full_answer += event["content"]
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "done":
                # Evaluate the complete answer
                flags, warning = evaluate(
                    answer=full_answer,
                    chunks_retrieved=len(search_results),
                    sources=sources,
                    query=question,
                    classification=route_result["classification"],
                )

                # Record conversation
                add_turn(conv_id, question, full_answer)

                # Send final metadata
                metadata_dict = {
                    "model_used": model,
                    "classification": route_result["classification"],
                    "rationale": rationale,
                    "tokens": event.get("token_usage", {"input": 0, "output": 0}),
                    "latency_ms": event.get("latency_ms", 0),
                    "chunks_retrieved": len(search_results),
                    "evaluator_flags": flags,
                }
                
                # Cache the streaming response globally
                query_cache.set(question, {
                    "answer": full_answer,
                    "metadata": metadata_dict,
                    "sources": sources,
                    "warning": warning
                })

                done_data = {
                    "type": "done",
                    "metadata": metadata_dict,
                    "sources": sources,
                    "conversation_id": conv_id,
                    "warning": warning,
                }
                yield f"data: {json.dumps(done_data)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Health check ---

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "index_ready": retriever.is_ready,
        "index_size": retriever.index.ntotal if retriever.index else 0,
    }

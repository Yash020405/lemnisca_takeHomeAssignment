"""Groq LLM client - handles chat completion and streaming.

Wraps the official Groq Python SDK for both regular and streaming responses.
Extracts token usage from the API response for logging/metadata.
"""
import logging
import time
from typing import AsyncGenerator, Dict, List, Tuple

from groq import Groq

from backend.app.config import GROQ_API_KEY, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Module-level client singleton
_client: Groq | None = None


def get_client() -> Groq:
    """Get or create the Groq client singleton."""
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. Create a .env file in backend/ "
                "with GROQ_API_KEY=your_key_here"
            )
        _client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized")
    return _client


def generate(
    model: str,
    query: str,
    context_chunks: List[str],
    conversation_history: List[Dict] | None = None,
) -> Tuple[str, Dict, int]:
    """Generate a response using the Groq API.

    Args:
        model: Groq model string (e.g., 'llama-3.1-8b-instant')
        query: User's question
        context_chunks: Retrieved document chunks for context
        conversation_history: Optional list of prior conversation messages

    Returns:
        Tuple of:
          - answer: Generated text
          - token_usage: {'input': int, 'output': int}
          - latency_ms: Response time in milliseconds
    """
    client = get_client()

    # Build messages
    messages = _build_messages(query, context_chunks, conversation_history)

    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,  # Low temperature for factual Q&A
            max_tokens=1024,
            top_p=0.9,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        answer = response.choices[0].message.content or ""
        token_usage = {
            "input": response.usage.prompt_tokens if response.usage else 0,
            "output": response.usage.completion_tokens if response.usage else 0,
        }

        logger.info(
            f"LLM response: model={model}, "
            f"tokens_in={token_usage['input']}, tokens_out={token_usage['output']}, "
            f"latency={latency_ms}ms"
        )

        return answer, token_usage, latency_ms

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Groq API error: {e}")
        return (
            f"I'm sorry, I encountered an error while processing your request. "
            f"Please try again in a moment.",
            {"input": 0, "output": 0},
            latency_ms,
        )


def generate_stream(
    model: str,
    query: str,
    context_chunks: List[str],
    conversation_history: List[Dict] | None = None,
):
    """Stream a response from the Groq API token-by-token.

    Yields:
        dicts with either:
          - {"type": "token", "content": str}  - a token chunk
          - {"type": "done", "token_usage": dict, "latency_ms": int}  - final metadata
    """
    client = get_client()
    messages = _build_messages(query, context_chunks, conversation_history)

    start_time = time.time()
    total_output_tokens = 0

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9,
            stream=True,
        )

        final_usage = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                total_output_tokens += 1  # Approximate token count
                yield {"type": "token", "content": content}

            # Groq SDK places usage on the final chunk via chunk.usage
            # or via x_groq.usage depending on SDK version
            if hasattr(chunk, "usage") and chunk.usage:
                final_usage = chunk.usage
            if hasattr(chunk, "x_groq") and chunk.x_groq and hasattr(chunk.x_groq, "usage") and chunk.x_groq.usage:
                final_usage = chunk.x_groq.usage

        # Emit done event with best available usage data
        latency_ms = int((time.time() - start_time) * 1000)
        if final_usage:
            token_usage = {
                "input": final_usage.prompt_tokens,
                "output": final_usage.completion_tokens,
            }
        else:
            # Fallback: estimate input tokens from message length
            total_chars = sum(len(m.get("content", "")) for m in messages)
            estimated_input = max(1, total_chars // 4)  # ~4 chars per token
            token_usage = {
                "input": estimated_input,
                "output": total_output_tokens,
            }
        yield {
            "type": "done",
            "token_usage": token_usage,
            "latency_ms": latency_ms,
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Groq streaming error: {e}")
        yield {
            "type": "token",
            "content": "I'm sorry, I encountered an error. Please try again.",
        }
        yield {
            "type": "done",
            "token_usage": {"input": 0, "output": 0},
            "latency_ms": latency_ms,
        }


def _build_messages(
    query: str,
    context_chunks: List[str],
    conversation_history: List[Dict] | None = None,
) -> List[Dict]:
    """Build the message list for the Groq API.

    Structure:
      1. System prompt
      2. Conversation history (if any, last N turns)
      3. Context from retrieved documents
      4. User query
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history
    if conversation_history:
        messages.extend(conversation_history)

    # Build context block
    if context_chunks:
        context_text = "\n\n---\n\n".join(
            f"[Document Context {i+1}]:\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        )
        user_content = (
            f"Based on the following documentation context, answer the user's question.\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"USER QUESTION: {query}"
        )
    elif conversation_history:
        user_content = (
            f"No new documentation context was retrieved for this query, "
            f"but you have prior conversation history above. "
            f"Use the conversation history to answer the user's follow-up question.\n\n"
            f"USER QUESTION: {query}"
        )
    else:
        user_content = (
            f"The user asked a question but no relevant documentation was found. "
            f"If this is a greeting or general question, respond naturally. "
            f"Otherwise, let the user know you couldn't find relevant documentation.\n\n"
            f"USER QUESTION: {query}"
        )

    messages.append({"role": "user", "content": user_content})

    return messages

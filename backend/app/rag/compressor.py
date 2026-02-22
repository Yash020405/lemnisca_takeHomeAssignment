"""Context Compression (Selective Context) for Token Optimization.

Implements "Context Compression" as researched in recent RAG optimization
papers. Instead of sending full 500-token chunks to the LLM, this module
breaks the retrieved chunks into sentences and scores each sentence against
the user's query.

Only the top-scoring sentences (plus surrounding context windows) are kept.
This drastically reduces the input token count without losing the factual
information necessary to answer the query.

References:
- "Selective Context" RAG Prompt Optimization
- Context distillation via Key Sentence Extraction
"""
import re
from typing import List
from rank_bm25 import BM25Okapi

from backend.app.rag.chunker import Chunk

def _split_into_sentences(text: str) -> List[str]:
    """Naive regex sentence splitter. Good enough for token optimization."""
    # Split on period, question mark, or exclamation point followed by a space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return [t for t in tokens if len(t) >= 2]

class ContextCompressor:
    """Dynamically compresses retrieved chunks based on the query."""
    
    def __init__(self, compression_ratio: float = 0.5):
        """
        Args:
            compression_ratio: Target ratio of sentences to KEEP (e.g. 0.5 = keep 50%)
        """
        self.compression_ratio = compression_ratio
        
    def compress(self, chunk: Chunk, query: str) -> Chunk:
        """Compress a single chunk by removing irrelevant sentences."""
        sentences = _split_into_sentences(chunk.text)
        
        # If the chunk is already very short, don't compress it
        if len(sentences) <= 3:
            return chunk
            
        # Target number of sentences to keep
        target_len = max(3, int(len(sentences) * self.compression_ratio))
        
        # Build BM25 index over the chunk's sentences
        corpus = [_tokenize(s) for s in sentences]
        
        # Edge case: if sentences are un-tokenizable (e.g. all symbols)
        if not any(corpus):
            return chunk
            
        bm25 = BM25Okapi(corpus)
        query_tokens = _tokenize(query)
        
        if not query_tokens:
            return chunk
            
        # Score each sentence against the query
        scores = bm25.get_scores(query_tokens)
        
        # We want to keep the highest scoring sentences, BUT we must maintain 
        # their original order in the document so it reads naturally to the LLM.
        # So we find the top-K indices, sort the indices, and extract those sentences.
        
        # Get indices of top-N scores
        top_indices = scores.argsort()[-target_len:][::-1]
        
        # Add 1 sentence of padding around the ABSOLUTE best matches 
        # to ensure context isn't entirely orphaned
        best_match_idx = top_indices[0] if len(top_indices) > 0 else 0
        context_indices = set(top_indices)
        if best_match_idx > 0:
            context_indices.add(best_match_idx - 1)
        if best_match_idx < len(sentences) - 1:
            context_indices.add(best_match_idx + 1)
            
        # Sort indices to maintain chronological order
        final_indices = sorted(list(context_indices))
        
        # Reconstruct the compressed text
        compressed_text = " ".join([sentences[i] for i in final_indices])
        
        # Return a new Chunk object so we don't mutate the cached originals
        # Prepending/appending '...' to indicate compression
        return Chunk(
            text=f"... {compressed_text} ...",
            filename=chunk.filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            metadata=chunk.metadata
        )

# Singleton instance
compressor = ContextCompressor(compression_ratio=0.5)

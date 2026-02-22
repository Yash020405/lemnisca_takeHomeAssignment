"""Hybrid retrieval: combines FAISS semantic search with BM25 keyword search.

Uses Reciprocal Rank Fusion (RRF) to merge results from both retrieval paths.
This solves the vocabulary mismatch problem where semantic search finds
conceptually related but keyword-different chunks, and BM25 catches exact
keyword matches that semantic search might rank lower.

References:
- Anthropic, "Contextual Retrieval" (2024): BM25 + embedding reduced failed
  retrievals by 49% over embedding-only.
- Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual
  Rank Learning Methods" (SIGIR 2009): RRF consistently outperforms individual
  ranking methods.
"""
import logging
import re
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict

from rank_bm25 import BM25Okapi

from backend.app.rag.chunker import Chunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    text = text.lower()
    # Split on non-alphanumeric characters, keep tokens of length >= 2
    tokens = re.findall(r'[a-z0-9]+', text)
    return [t for t in tokens if len(t) >= 2]


class BM25Index:
    """BM25 keyword index for document chunks."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Chunk] = []
        self._corpus: List[List[str]] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self.chunks = chunks
        self._corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self._corpus)
        logger.info(f"BM25 index built: {len(chunks)} documents")

    def save(self, index_path: str) -> None:
        """Serialize the BM25 index and chunks to disk."""
        if self.bm25 is None:
            return
        logger.info(f"Saving BM25 index to {index_path}")
        with open(index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks, "corpus": self._corpus}, f)
            
    def load(self, index_path: str) -> bool:
        """Load the BM25 index and chunks from disk."""
        path = Path(index_path)
        if not path.exists():
            return False
            
        logger.info(f"Loading BM25 index from {index_path}")
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
                self._corpus = data["corpus"]
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search BM25 index.

        Returns:
            List of (chunk_index, bm25_score) tuples, sorted by score descending.
        """
        if self.bm25 is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        # Get top-k indices sorted by score
        top_indices = scores.argsort()[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def reciprocal_rank_fusion(
    semantic_results: List[Tuple[int, float]],
    bm25_results: List[Tuple[int, float]],
    k: int = 60,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Tuple[int, float]]:
    """Merge two ranked lists using weighted Reciprocal Rank Fusion.

    RRF score for a document d = sum(weight / (k + rank(d))) across all lists.
    k=60 is the standard constant from the original paper.

    Args:
        semantic_results: List of (chunk_index, score) from FAISS
        bm25_results: List of (chunk_index, score) from BM25
        k: RRF constant (default 60)
        semantic_weight: Weight for semantic results
        bm25_weight: Weight for BM25 results

    Returns:
        Merged list of (chunk_index, rrf_score), sorted by score descending.
    """
    rrf_scores: Dict[int, float] = defaultdict(float)

    for rank, (idx, _) in enumerate(semantic_results):
        rrf_scores[idx] += semantic_weight / (k + rank + 1)

    for rank, (idx, _) in enumerate(bm25_results):
        rrf_scores[idx] += bm25_weight / (k + rank + 1)

    # Sort by RRF score descending
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


# Module-level singleton
bm25_index = BM25Index()

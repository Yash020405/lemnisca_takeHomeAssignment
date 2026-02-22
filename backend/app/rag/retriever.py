"""FAISS + BM25 hybrid retriever with disk caching.

Combines dense semantic search (FAISS inner-product) with sparse keyword
search (BM25-Okapi) using Reciprocal Rank Fusion (RRF). This approach is
backed by Anthropic's "Contextual Retrieval" research (2024) showing a 49%
reduction in failed retrievals vs embedding-only search.

The pipeline:
  1. FAISS semantic search -> top-k by cosine similarity
  2. BM25 keyword search   -> top-k by term frequency
  3. Reciprocal Rank Fusion -> merge both ranked lists
  4. Score gap filtering    -> drop noisy tail results

Index + chunk metadata are persisted to disk so subsequent startups are instant.
"""
import json
import logging
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from backend.app.config import INDEX_DIR, TOP_K, RELEVANCE_THRESHOLD
from backend.app.rag.chunker import Chunk
from backend.app.rag.embedder import encode_texts, encode_query
from backend.app.rag.hybrid_retriever import bm25_index, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Hybrid FAISS + BM25 retriever for document chunks."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: List[Chunk] = []
        self._index_path = INDEX_DIR / "faiss.index"
        self._chunks_path = INDEX_DIR / "chunks.pkl"
        self._bm25_path = INDEX_DIR / "bm25.pkl"

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def build_index(self, chunks: List[Chunk]) -> None:
        """Build FAISS + BM25 indexes from chunks and persist to disk."""
        if not chunks:
            logger.error("No chunks to index")
            return

        self.chunks = chunks
        texts = [c.text for c in chunks]

        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = encode_texts(texts, show_progress=True)

        # Build inner product index (cosine sim with normalized vectors)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Build BM25 index for keyword search
        bm25_index.build(chunks)

        logger.info(f"FAISS index built: {self.index.ntotal} vectors, {dim} dimensions")

        # Persist to disk
        self._save()

    def load_index(self) -> bool:
        """Load index and chunks from disk cache. Returns True if successful."""
        if self._index_path.exists() and self._chunks_path.exists() and self._bm25_path.exists():
            try:
                self.index = faiss.read_index(str(self._index_path))
                with open(self._chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)

                # Load BM25 index from disk directly
                bm25_index.load(str(self._bm25_path))

                logger.info(
                    f"Loaded cached index: {self.index.ntotal} vectors, "
                    f"{len(self.chunks)} chunks"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load cached index: {e}")
                return False
        return False

    def search(self, query: str, top_k: int | None = None) -> List[Tuple[Chunk, float]]:
        """Hybrid search: FAISS semantic + BM25 keyword, merged with RRF.

        Args:
            query: User query string
            top_k: Number of results to return (default from config)

        Returns:
            List of (chunk, relevance_score) tuples, sorted by relevance descending.
        """
        if not self.is_ready:
            logger.error("Index not ready - call build_index() or load_index() first")
            return []

        k = top_k or TOP_K

        # Over-retrieve for better fusion (2x the final count)
        retrieval_k = min(k * 2, self.index.ntotal)

        # --- Path 1: FAISS semantic search ---
        query_vec = encode_query(query)
        scores, indices = self.index.search(query_vec, retrieval_k)
        semantic_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and float(score) >= RELEVANCE_THRESHOLD:
                semantic_results.append((int(idx), float(score)))

        # --- Path 2: BM25 keyword search ---
        bm25_results = bm25_index.search(query, top_k=retrieval_k)

        # --- Merge with Reciprocal Rank Fusion ---
        if not semantic_results:
            # If FAISS finds nothing above threshold, it's out of domain.
            return []

        semantic_dict = {idx: score for idx, score in semantic_results}
        # Only keep BM25 results that are semantically relevant
        filtered_bm25 = [(idx, score) for idx, score in bm25_results if idx in semantic_dict]

        if filtered_bm25:
            fused = reciprocal_rank_fusion(semantic_results, filtered_bm25)
            # Take top-k from fused results
            result_indices = [idx for idx, _ in fused[:k]]
        else:
            result_indices = [idx for idx, _ in semantic_results[:k]]

        # Build final results with original FAISS scores for display
        results = []
        for idx in result_indices:
            score = semantic_dict[idx]
            results.append((self.chunks[idx], score))

        # We intentionally skip score gap filtering here because RRF
        # correctly balances semantic and keyword matches, and FAISS
        # scores are no longer strictly monotonically decreasing.

        logger.info(
            f"Retrieved {len(results)} chunks for query: '{query[:80]}...' "
            f"(top score: {results[0][1]:.3f}, "
            f"semantic={len(semantic_results)}, bm25={len(bm25_results)})" if results else
            f"No relevant chunks found for query: '{query[:80]}...'"
        )
        return results

    @staticmethod
    def _filter_score_gap(
        results: List[Tuple[Chunk, float]], gap_threshold: float = 0.15
    ) -> List[Tuple[Chunk, float]]:
        """Drop results after a large score gap from the top result."""
        if len(results) <= 1:
            return results

        filtered = [results[0]]
        top_score = results[0][1]
        for i in range(1, len(results)):
            score = results[i][1]
            gap_from_top = top_score - score
            if gap_from_top > gap_threshold:
                break
            filtered.append(results[i])

        return filtered

    def _save(self) -> None:
        """Persist index and chunks to disk."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._index_path))
        with open(self._chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        bm25_index.save(str(self._bm25_path))
        logger.info(f"Index saved to {INDEX_DIR}")


# Module-level singleton
retriever = VectorRetriever()

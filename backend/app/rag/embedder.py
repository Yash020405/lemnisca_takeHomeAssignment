"""Embedding generation using sentence-transformers.

Uses all-MiniLM-L6-v2: 384-dim vectors, fast inference, strong semantic quality.
This is loaded once at startup and shared across all requests.
"""
import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Module-level singleton - loaded once, reused across requests
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Embedding model loaded. Dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def encode_texts(texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
    """Encode a list of texts into normalized embeddings.

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized for cosine similarity.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2 normalize so dot product = cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """Encode a single query into a normalized embedding vector.

    Returns:
        numpy array of shape (1, embedding_dim), L2-normalized.
    """
    model = get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.astype(np.float32)

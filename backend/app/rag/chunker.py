"""Hybrid chunking strategy for RAG pipeline.

Strategy:
1. Split by paragraph (double newline) to preserve natural document boundaries
2. Merge short paragraphs (<MIN_CHUNK_SIZE chars) with the next to avoid trivial chunks
3. Split long chunks at sentence boundaries with overlap for context continuity
4. Preserve metadata (filename, page, chunk_index) through the pipeline

This balances retrieval precision (small enough chunks) with context preservation
(overlap + paragraph boundaries prevent mid-sentence splits).
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List

from backend.app.config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from backend.app.rag.pdf_loader import Document

logger = logging.getLogger(__name__)

# Sentence boundary pattern - splits on period, question mark,
# exclamation mark followed by space or end of string
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


@dataclass
class Chunk:
    """A text chunk with source metadata for retrieval tracking."""
    text: str
    filename: str
    page_number: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


def chunk_documents(documents: List[Document]) -> List[Chunk]:
    """Chunk all documents using hybrid paragraph + fixed-size strategy."""
    all_chunks: List[Chunk] = []
    chunk_index = 0

    for doc in documents:
        # Step 1: Split into paragraphs
        paragraphs = _split_paragraphs(doc.text)

        # Step 2: Merge short paragraphs
        merged = _merge_short_paragraphs(paragraphs)

        # Step 3: Split long paragraphs at sentence boundaries with overlap
        final_texts = []
        for para in merged:
            if len(para) <= CHUNK_SIZE:
                final_texts.append(para)
            else:
                final_texts.extend(_split_with_overlap(para))

        # Step 4: Create Chunk objects with metadata
        for text in final_texts:
            if text.strip():
                all_chunks.append(Chunk(
                    text=text.strip(),
                    filename=doc.filename,
                    page_number=doc.page_number,
                    chunk_index=chunk_index,
                    metadata={
                        **doc.metadata,
                        "char_count": len(text.strip()),
                    }
                ))
                chunk_index += 1

    logger.info(
        f"Chunked {len(documents)} document pages into {len(all_chunks)} chunks "
        f"(avg {sum(c.char_count for c in all_chunks) // max(len(all_chunks), 1)} chars/chunk)"
    )
    return all_chunks


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs using double newlines."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _merge_short_paragraphs(paragraphs: List[str]) -> List[str]:
    """Merge paragraphs shorter than MIN_CHUNK_SIZE with the next paragraph."""
    if not paragraphs:
        return []

    merged = []
    buffer = ""

    for para in paragraphs:
        if buffer:
            buffer = buffer + "\n\n" + para
        else:
            buffer = para

        if len(buffer) >= MIN_CHUNK_SIZE:
            merged.append(buffer)
            buffer = ""

    # Don't lose the trailing buffer
    if buffer:
        if merged:
            merged[-1] = merged[-1] + "\n\n" + buffer
        else:
            merged.append(buffer)

    return merged


def _split_with_overlap(text: str) -> List[str]:
    """Split long text at sentence boundaries with character overlap."""
    sentences = _SENTENCE_RE.split(text)
    if not sentences:
        return [text]

    chunks = []
    current = ""
    overlap_buffer = ""

    for sentence in sentences:
        # Would adding this sentence exceed the chunk size?
        candidate = (current + " " + sentence).strip() if current else sentence

        if len(candidate) > CHUNK_SIZE and current:
            # Save current chunk
            chunks.append(current)

            # Calculate overlap: take the last CHUNK_OVERLAP chars from current
            if len(current) > CHUNK_OVERLAP:
                overlap_buffer = current[-CHUNK_OVERLAP:]
                # Try to start overlap at a word boundary
                space_idx = overlap_buffer.find(" ")
                if space_idx > 0:
                    overlap_buffer = overlap_buffer[space_idx + 1:]
            else:
                overlap_buffer = current

            current = (overlap_buffer + " " + sentence).strip()
        else:
            current = candidate

    # Don't forget the last chunk
    if current:
        chunks.append(current)

    return chunks

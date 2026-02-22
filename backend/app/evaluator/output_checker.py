"""Output evaluator - flags potentially unreliable LLM responses.

Four checks:
1. no_context        - LLM answered but no relevant chunks were retrieved (hallucination risk)
2. refusal           - LLM explicitly said it can't help
3. conflicting_sources - answer references multiple docs with hedging/uncertainty language
                         (domain-specific: ClearPath docs contain known pricing discrepancies)
4. short_answer      - complex query received a suspiciously brief response

Design rationale for conflicting_sources:
  The ClearPath docs intentionally contain conflicting information (hinted in the
  API contract example response and the docs README). When the retriever pulls chunks
  from multiple documents and the LLM hedges, the user should know the answer may be
  unreliable and should verify with support. This is especially important for pricing,
  SLA, and feature availability questions where wrong info has real consequences.
"""
import logging
import re
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# --- Refusal patterns ---
REFUSAL_PHRASES = [
    "i don't have",
    "i do not have",
    "not mentioned",
    "cannot find",
    "can't find",
    "no information",
    "i'm unable",
    "i am unable",
    "not available in",
    "beyond my knowledge",
    "i don't know",
    "i do not know",
    "not covered in",
    "not specified in",
    "no relevant",
    "unable to find",
    "doesn't appear",
    "does not appear",
    "not in the documentation",
    "not in the provided",
    "outside the scope",
    "i cannot answer",
]

# --- Hedging/uncertainty language (for conflicting_sources) ---
HEDGING_PHRASES = [
    "however",
    "on the other hand",
    "differs",
    "conflicting",
    "inconsistent",
    "may vary",
    "discrepancy",
    "contradiction",
    "alternatively",
    "it depends",
    "not entirely clear",
    "appears to differ",
    "two different",
    "multiple sources",
    "varies between",
    "some documents",
    "according to one",
    "while another",
]


def evaluate(
    answer: str,
    chunks_retrieved: int,
    sources: List[Dict],
    query: str = "",
    classification: str = "simple",
) -> Tuple[List[str], str | None]:
    """Evaluate an LLM response for reliability.

    Args:
        answer: The LLM-generated answer text
        chunks_retrieved: Number of chunks that were retrieved
        sources: List of source dicts with 'document' keys
        query: Original user query (for context)
        classification: Query classification ('simple' or 'complex')

    Returns:
        Tuple of:
          - List of flag strings (empty if no issues)
          - Warning message for the user (None if no flags)
    """
    flags: List[str] = []
    answer_lower = answer.lower()

    # --- Check 1: no_context ---
    is_refusal = _check_refusal(answer_lower)

    if chunks_retrieved == 0 and not is_refusal:
        flags.append("no_context")
        logger.warning(f"Flag: no_context - LLM answered without retrieved context")

    # --- Check 2: refusal ---
    if is_refusal:
        flags.append("refusal")
        logger.warning(f"Flag: refusal - LLM indicated it cannot answer")

    # --- Check 3: conflicting_sources ---
    if _check_conflicting_sources(answer_lower, sources):
        flags.append("conflicting_sources")
        logger.warning(f"Flag: conflicting_sources - answer may contain discrepancies")

    # --- Check 4: short_answer ---
    if _check_short_answer(answer, classification):
        flags.append("short_answer")
        logger.warning(f"Flag: short_answer - complex query got a brief response ({len(answer)} chars)")

    # Build user-facing warning
    warning = None
    if flags:
        warning = "Low confidence - please verify with support."
        if "conflicting_sources" in flags:
            warning = "Multiple sources with potentially conflicting information - please verify with support."

    return flags, warning


def _check_refusal(answer_lower: str) -> bool:
    """Check if the answer is a refusal or non-answer."""
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def _check_conflicting_sources(answer_lower: str, sources: List[Dict]) -> bool:
    """Check if answer references multiple docs with uncertainty language.

    Triggers when:
      1. Sources come from >= 2 different documents, AND
      2. The answer contains hedging/uncertainty language
    """
    # Need at least 2 different source documents
    unique_docs = set()
    for src in sources:
        doc = src.get("document", "")
        if doc:
            unique_docs.add(doc)

    if len(unique_docs) < 2:
        return False

    # Check for hedging language
    hedging_count = sum(1 for phrase in HEDGING_PHRASES if phrase in answer_lower)
    return hedging_count >= 3  # Require 3+ hedging phrases to reduce false positives


def _check_short_answer(answer: str, classification: str) -> bool:
    """Flag complex queries with suspiciously short responses.

    A complex query routed to the 70B model should produce a substantive
    answer. If the response is under 50 characters, the model likely failed
    to engage with the question (e.g., gave a vague one-liner instead of
    walking through the multi-step answer the query requires).
    """
    if classification != "complex":
        return False
    # Strip whitespace for accurate length check
    stripped = answer.strip()
    return len(stripped) < 50


"""Deterministic, rule-based query classifier for model routing.

Routes queries to:
  - Simple → llama-3.1-8b-instant  (fast, cheap, good for factual lookups)
  - Complex → llama-3.3-70b-versatile  (powerful, slower, for reasoning)

Classifier Design Philosophy:
---
The router uses a WEIGHTED SIGNAL ACCUMULATION strategy rather than a single
threshold. Multiple independent signals are evaluated, each adding to a
complexity score. This is more robust than any single rule because:
  1. Short queries with complex keywords still route correctly
  2. Long but simple queries (e.g., pasted error messages) don't always trigger complex
  3. Greeting detection prevents routing small talk to the expensive model

Signal Categories:
  - Lexical: word count, question mark count, keyword presence
  - Syntactic: subordinate clause markers, conjunction density
  - Semantic intent: greeting patterns, complaint markers, comparison language
  - Domain-specific: pricing/billing keywords (often need cross-doc reasoning)
"""
import re
import logging
from typing import Dict

from backend.app.config import MODEL_SIMPLE, MODEL_COMPLEX

logger = logging.getLogger(__name__)

# --- Signal definitions ---

# Keywords that indicate the query needs deeper reasoning
COMPLEX_KEYWORDS = {
    # Explanation/reasoning
    "how", "why", "explain", "describe", "elaborate", "walk me through",
    "step by step", "steps",
    # Comparison/analysis
    "compare", "comparison", "difference", "differences", "versus", "vs",
    "pros and cons", "advantages", "disadvantages",
    # Troubleshooting
    "troubleshoot", "debug", "fix", "resolve", "error", "issue", "problem",
    "not working", "broken", "failed",
    # Configuration/integration
    "configure", "setup", "integrate", "integration", "connect", "install",
    "migrate", "migration",
    # Multi-topic
    "and also", "in addition", "furthermore", "as well as", "along with",
}

# Keywords that signal domain-specific complexity (cross-doc reasoning)
DOMAIN_COMPLEX_KEYWORDS = {
    "pricing", "enterprise", "upgrade", "downgrade", "billing", "invoice",
    "cancel", "refund", "sla", "compliance", "security", "architecture",
    "deployment", "infrastructure", "roadmap",
}

# Subordinate clause markers - signal compound/complex sentence structure
SUBORDINATE_MARKERS = {
    "if", "when", "although", "because", "since", "while", "whereas",
    "however", "nevertheless", "unless", "even though", "despite",
    "in case", "provided that",
}

# Greeting patterns - override to simple
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening)|"
    r"thanks|thank\s*you|bye|goodbye|cheers|yo|sup|what\'s\s*up)\b",
    re.IGNORECASE,
)

# Complaint/frustration markers - route to the big model for empathetic handling
COMPLAINT_MARKERS = {
    "frustrated", "annoying", "terrible", "horrible", "worst", "hate",
    "unacceptable", "ridiculous", "complain", "complaint", "angry", "upset",
    "disappointed", "dissatisfied", "waste of time", "useless",
}


def classify(query: str) -> Dict:
    """Classify a query as 'simple' or 'complex' using deterministic rules.

    Returns a dict with:
        - classification: 'simple' | 'complex'
        - model_used: Groq model string
        - signals: dict of which signals fired (for debugging/logging)
        - score: total complexity score
    """
    query_lower = query.lower().strip()
    words = query_lower.split()
    word_count = len(words)

    signals = {}
    score = 0

    # --- Signal 1: Greeting override ---
    if GREETING_PATTERNS.match(query_lower) and word_count < 6:
        return {
            "classification": "simple",
            "model_used": MODEL_SIMPLE,
            "signals": {"greeting_override": True},
            "score": 0,
        }

    # --- Signal 2: Word count ---
    if word_count >= 20:
        signals["long_query"] = word_count
        score += 2
    elif word_count >= 12:
        signals["medium_query"] = word_count
        score += 1

    # --- Signal 3: Complex keywords ---
    found_complex = [kw for kw in COMPLEX_KEYWORDS if kw in query_lower]
    if found_complex:
        signals["complex_keywords"] = found_complex
        score += min(len(found_complex), 3)  # Cap at 3 to avoid over-weighting

    # --- Signal 4: Domain complexity ---
    found_domain = [kw for kw in DOMAIN_COMPLEX_KEYWORDS if kw in query_lower]
    if len(found_domain) >= 2:  # Multiple domain topics → cross-doc reasoning
        signals["domain_complex"] = found_domain
        score += 2
    elif found_domain:
        signals["domain_keywords"] = found_domain
        score += 1

    # --- Signal 5: Multiple questions ---
    question_marks = query.count("?")
    if question_marks >= 2:
        signals["multiple_questions"] = question_marks
        score += 2

    # --- Signal 6: Subordinate clauses ---
    found_subordinate = [m for m in SUBORDINATE_MARKERS if m in query_lower]
    if found_subordinate and word_count >= 8:
        signals["subordinate_clauses"] = found_subordinate
        score += 1

    # --- Signal 7: Complaint/frustration ---
    found_complaints = [c for c in COMPLAINT_MARKERS if c in query_lower]
    if found_complaints:
        signals["complaint_markers"] = found_complaints
        score += 2  # Always route complaints to the big model

    # --- Signal 8: List/enumeration detection ---
    # Queries asking for lists or multiple items need more reasoning
    list_patterns = re.findall(
        r'\b(list|all|every|each|various|different types|options|features)\b',
        query_lower
    )
    if list_patterns and word_count >= 6:
        signals["list_request"] = list_patterns
        score += 1

    # --- Final classification ---
    # Threshold: score >= 2 → complex
    classification = "complex" if score >= 2 else "simple"
    model = MODEL_COMPLEX if classification == "complex" else MODEL_SIMPLE

    logger.info(
        f"Router: '{query[:60]}...' → {classification} "
        f"(score={score}, signals={list(signals.keys())})"
    )

    return {
        "classification": classification,
        "model_used": model,
        "signals": signals,
        "score": score,
    }

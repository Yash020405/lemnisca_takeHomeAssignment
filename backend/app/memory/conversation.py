"""Conversation memory - maintains multi-turn context.

Design decisions:
  1. In-memory dict (no external DB) - suitable for demo/assignment scope
  2. Fixed window of MAX_MEMORY_TURNS turns - controls token cost linearly
  3. Stores both user queries and assistant responses for coherent multi-turn
  4. Token cost tradeoff: 5 turns × ~200 tokens/turn = ~1000 extra tokens/request
     At 5000 queries/day, that's ~5M extra tokens - worth it for conversation quality

For production: would use Redis or a proper session store with TTL expiry.
"""
import logging
import uuid
from typing import Dict, List, Optional

from backend.app.config import MAX_MEMORY_TURNS

logger = logging.getLogger(__name__)

# In-memory conversation store: conversation_id → list of messages
_conversations: Dict[str, List[Dict]] = {}


def get_or_create_conversation_id(conversation_id: Optional[str] = None) -> str:
    """Return existing conversation ID or generate a new one."""
    if conversation_id and conversation_id in _conversations:
        return conversation_id
    new_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    _conversations[new_id] = []
    logger.info(f"Created new conversation: {new_id}")
    return new_id


def get_history(conversation_id: str) -> List[Dict]:
    """Get conversation history for LLM context.

    Returns the last MAX_MEMORY_TURNS pairs of (user, assistant) messages
    formatted for the Groq API message format.
    """
    if conversation_id not in _conversations:
        return []

    messages = _conversations[conversation_id]
    # Keep last N turns (each turn = 1 user + 1 assistant message = 2 items)
    max_messages = MAX_MEMORY_TURNS * 2
    return messages[-max_messages:]


def add_turn(conversation_id: str, user_query: str, assistant_response: str) -> None:
    """Record a conversation turn."""
    if conversation_id not in _conversations:
        _conversations[conversation_id] = []

    _conversations[conversation_id].append({
        "role": "user",
        "content": user_query,
    })
    _conversations[conversation_id].append({
        "role": "assistant",
        "content": assistant_response,
    })

    # Trim to max window
    max_messages = MAX_MEMORY_TURNS * 2
    if len(_conversations[conversation_id]) > max_messages:
        _conversations[conversation_id] = _conversations[conversation_id][-max_messages:]

    logger.debug(
        f"Conversation {conversation_id}: "
        f"{len(_conversations[conversation_id]) // 2} turns stored"
    )


def get_conversation_count() -> int:
    """Get the number of active conversations."""
    return len(_conversations)


def clear_conversation(conversation_id: str) -> None:
    """Clear a specific conversation's history."""
    if conversation_id in _conversations:
        del _conversations[conversation_id]
        logger.info(f"Cleared conversation: {conversation_id}")

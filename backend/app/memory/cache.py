import json
import logging
import re
from pathlib import Path
from backend.app.config import INDEX_DIR
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_FILE = INDEX_DIR / ".query_cache.json"

class SemanticCache:
    """Normalizes queries and caches responses to skip LLM calls for duplicate-intent questions."""
    
    def __init__(self):
        self.cache = {}
        self._load()
        
    def _normalize(self, query: str) -> str:
        """Normalize query to maximize cache hit variance.
        e.g. 'What is ClearPath?' -> 'what is clearpath'
        """
        # Lowercase
        query = query.lower()
        # Remove trailing punctuation (especially question marks)
        query = re.sub(r'[?!.]+$', '', query)
        # Collapse multiple spaces
        query = re.sub(r'\s+', ' ', query)
        return query.strip()

    def get(self, query: str) -> Optional[dict]:
        """Fetch cached response for normalized query."""
        key = self._normalize(query)
        res = self.cache.get(key)
        if res:
            logger.info(f"Semantic Cache HIT for query: '{query}' (normalized: '{key}')")
        return res

    def set(self, query: str, response_data: dict) -> None:
        """Cache response data for normalized query."""
        key = self._normalize(query)
        self.cache[key] = response_data
        self._save()
        logger.debug(f"Cached response for normalized query: '{key}'")

    def _save(self) -> None:
        """Persist cache to disk."""
        try:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save semantic cache: {e}")

    def _load(self) -> None:
        """Load cache from disk."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} entries from semantic cache.")
            except Exception as e:
                logger.error(f"Failed to load semantic cache: {e}")

# Global singleton
query_cache = SemanticCache()

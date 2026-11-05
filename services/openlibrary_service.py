"""Secondary metadata enrichment service via Open Library API."""

import time
from typing import Any, Dict, Optional
import requests

from config.settings import logger, settings

class OpenLibraryService:
    """Provides cached, rate-limited lookups against the Open Library API."""

    def __init__(self):
        self.user_agent = settings.openlibrary_user_agent
        self.delay = settings.openlibrary_delay
        self.last_call_time = 0.0
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _throttle(self) -> None:
        """Enforce rate limiting between successive requests."""
        now = time.time()
        elapsed = now - self.last_call_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_call_time = time.time()

    def search_book(self, title: str, author: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Search Open Library for book metadata with caching."""
        cache_key = f"{title.lower()}::{author.lower() if author else ''}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._throttle()
        params = {"title": title}
        if author:
            params["author"] = author

        headers = {"User-Agent": self.user_agent}
        try:
            resp = requests.get(
                "https://openlibrary.org/search.json",
                params=params,
                headers=headers,
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                docs = data.get("docs", [])
                if docs:
                    first = docs[0]
                    result = {
                        "title": first.get("title"),
                        "author_name": first.get("author_name", [author])[0] if first.get("author_name") else author,
                        "first_publish_year": first.get("first_publish_year"),
                        "isbn": first.get("isbn", [])[:3] if first.get("isbn") else [],
                        "openlibrary_key": first.get("key"),
                    }
                    self._cache[cache_key] = result
                    return result
        except Exception as e:
            logger.warning("Open Library lookup failed gracefully: %s", e)

        return None

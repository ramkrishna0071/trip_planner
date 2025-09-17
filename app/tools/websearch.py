from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict, Iterable
from urllib.parse import urlparse
import os
import re

import httpx

from .html_to_text import html_to_text

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
_level = os.getenv("TRIP_PLANNER_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _level, logging.INFO))
logger.propagate = False

@dataclass
class SourcePolicy:
    allow_domains: Optional[Sequence[str]] = None
    deny_domains: Optional[Sequence[str]] = None
    max_results: int = 6
    max_per_domain: int = 2
    recency_days: Optional[int] = None

    def allowed(self, url: str) -> bool:
        def match_any(patterns: Optional[Sequence[str]]) -> bool:
            return bool(patterns) and any(re.search(p, url) for p in patterns)
        if self.deny_domains and match_any(self.deny_domains):
            return False
        if self.allow_domains:
            return match_any(self.allow_domains)
        return True

@dataclass
class WebDoc:
    url: str
    title: str
    text: str

class WebSearcher:
    """
    Pluggable search+fetch. Replace `search()` with your provider (Bing/SerpAPI/Tavily).
    """
    SEARCH_ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, policy: SourcePolicy, *, api_key: Optional[str] = None):
        self.policy = policy
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    async def search(self, query: str) -> List[Dict]:
        """Run a Tavily search and return filtered results.

        The returned payload includes the ``url``, ``title`` and any provider
        summary text so the orchestrator can fall back to those snippets when a
        direct fetch fails. Results are pre-filtered according to the configured
        ``SourcePolicy`` so callers do not have to re-apply allow/deny logic.
        """
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable not configured")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": self.policy.max_results * 2,
            "include_answer": False,
            "include_images": False,
        }
        if self.policy.allow_domains:
            payload["include_domains"] = list(self.policy.allow_domains)
        if self.policy.deny_domains:
            payload["exclude_domains"] = list(self.policy.deny_domains)

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self.SEARCH_ENDPOINT, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_results: Iterable[Dict] = data.get("results", [])
        return self._apply_policy(raw_results)

    async def fetch(self, url: str, timeout: float = 10.0) -> Optional[WebDoc]:
        if not self.policy.allowed(url):
            return None
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent":"trip-planner/1.0"})
                r.raise_for_status()
                text = html_to_text(r.text)[:12000]
                title = self._title_from_html(r.text) or url
                return WebDoc(url=url, title=title, text=text)
        except Exception:
            logger.warning("Failed to fetch url %s", url, exc_info=True)
            return None

    @staticmethod
    def _title_from_html(html: str) -> Optional[str]:
        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return None

    def _apply_policy(self, results: Iterable[Dict]) -> List[Dict]:
        filtered: List[Dict] = []
        seen_urls: set[str] = set()
        per_domain: Dict[str, int] = {}

        for result in results:
            url = result.get("url") or result.get("href")
            if not url:
                continue
            if url in seen_urls:
                continue
            if not self.policy.allowed(url):
                continue

            domain = self._domain_for(url)
            if not domain:
                continue
            if per_domain.get(domain, 0) >= self.policy.max_per_domain:
                continue

            title = result.get("title") or ""
            snippet = result.get("content") or result.get("snippet") or ""
            filtered.append({"url": url, "title": title, "content": snippet})
            seen_urls.add(url)
            per_domain[domain] = per_domain.get(domain, 0) + 1

            if len(filtered) >= self.policy.max_results:
                break

        return filtered

    @staticmethod
    def _domain_for(url: str) -> str:
        try:
            parsed = urlparse(url)
        except ValueError:
            return ""
        return (parsed.netloc or "").lower()

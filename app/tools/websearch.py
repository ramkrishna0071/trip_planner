from dataclasses import dataclass
from typing import List, Optional, Sequence, Dict
import re
import httpx
from .html_to_text import html_to_text

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
    def __init__(self, policy: SourcePolicy):
        self.policy = policy

    async def search(self, query: str) -> List[Dict]:
        # TODO: wire up your preferred search API and return [{'url':..., 'title':...}, ...]
        return []

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
            return None

    @staticmethod
    def _title_from_html(html: str) -> Optional[str]:
        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return None

from typing import List

import httpx
import pytest

from app.tools.websearch import WebSearcher, SourcePolicy, WebDoc


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummyAsyncClient:
    def __init__(self, response_payload, *args, **kwargs):
        self.response_payload = response_payload
        self.requests: List[tuple] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def post(self, url, json):
        self.requests.append((url, json))
        return DummyResponse(self.response_payload)


@pytest.mark.asyncio
async def test_websearch_search_applies_source_policy(monkeypatch):
    payload = {
        "results": [
            {"url": "https://allowed.com/a", "title": "A"},
            {"url": "https://allowed.com/a", "title": "A duplicate"},
            {"url": "https://denied.com/x", "title": "Denied"},
            {"url": "https://other.com/1", "title": "Other 1"},
            {"url": "https://other.com/2", "title": "Other 2"},
            {"url": "https://other.com/3", "title": "Other 3"},
        ]
    }

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: DummyAsyncClient(payload, *a, **kw))

    policy = SourcePolicy(deny_domains=[r"denied\.com"], max_results=4, max_per_domain=1)
    searcher = WebSearcher(policy)

    results = await searcher.search("family travel ideas")

    assert len(results) == 2  # allowed.com and first other.com entry
    assert {r["url"] for r in results} == {"https://allowed.com/a", "https://other.com/1"}
    assert all(r["title"] for r in results)


@pytest.mark.asyncio
async def test_orchestrator_dedupes_and_fetches_unique_results(monkeypatch):
    from app import orchestrator

    collected_snippets = {}

    def fake_call_llm(payload, snippets):
        collected_snippets["value"] = list(snippets)
        return {"llm": "ok"}

    monkeypatch.setattr(orchestrator, "call_llm", fake_call_llm)

    instances = []

    class FakeSearcher:
        def __init__(self, policy):
            self.policy = policy
            self.search_calls: List[str] = []
            self.fetch_calls: List[str] = []
            instances.append(self)

        async def search(self, query: str):
            self.search_calls.append(query)
            return [
                {"url": "https://allowed.com/a", "title": "A"},
                {"url": "https://allowed.com/a", "title": "A duplicate"},
                {"url": "https://denied.com/z", "title": "Denied"},
                {"url": "https://other.com/1", "title": "Other 1"},
                {"url": "https://other.com/2", "title": "Other 2"},
            ]

        async def fetch(self, url: str):
            self.fetch_calls.append(url)
            return WebDoc(url=url, title=f"Title for {url}", text="Snippet text")

    monkeypatch.setattr(orchestrator, "WebSearcher", FakeSearcher)

    payload = {
        "origin": "Paris",
        "destinations": ["Rome", "Milan"],
        "dates": {"start": "2025-01-01", "end": "2025-01-05"},
        "budget_total": 2000.0,
        "currency": "EUR",
        "party": {"adults": 2},
        "prefs": {"objective": "balanced"},
    }

    data = await orchestrator.orchestrate_llm_trip(
        payload,
        allow_domains=[r"allowed\.com", r"other\.com"],
        deny_domains=[r"denied\.com"],
    )

    # Only unique allowed URLs should be fetched.
    assert instances, "WebSearcher should have been instantiated"
    fetch_calls = instances[0].fetch_calls
    assert fetch_calls == [
        "https://allowed.com/a",
        "https://other.com/1",
        "https://other.com/2",
    ]

    snippets = data.get("snippets")
    assert snippets is not None
    assert len(snippets) == 3
    assert {s["url"] for s in snippets} == set(fetch_calls)
    assert collected_snippets["value"] == snippets

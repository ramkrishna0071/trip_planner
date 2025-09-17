import asyncio
import sys
from types import SimpleNamespace
import importlib
from typing import List, Dict, Any
import pytest


class _OpenAIStub:
    class chat:
        class completions:
            @staticmethod
            def create(*args, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
                )

    def __init__(self, *args, **kwargs):
        pass


# Stub the openai import so tests don't require the real package
sys.modules["openai"] = SimpleNamespace(OpenAI=_OpenAIStub)

# Prefer real httpx when available, otherwise fall back to a stub that surfaces a clear error
try:  # pragma: no cover - prefer real httpx when available
    httpx = importlib.import_module("httpx")  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _AsyncClientStub:
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("httpx is required for real network calls")

    httpx = SimpleNamespace(AsyncClient=_AsyncClientStub)

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


def test_websearch_search_applies_source_policy(monkeypatch):
    async def run() -> None:
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
        assert all("content" in r for r in results)

    asyncio.run(run())


def test_orchestrator_dedupes_and_fetches_unique_results(monkeypatch):
    async def run() -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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
        assert all(s.get("source") == "web_fetch" for s in snippets)
        assert collected_snippets["value"] == snippets

    asyncio.run(run())


def test_orchestrator_respects_payload_domain_lists(monkeypatch):
    async def run() -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from app import orchestrator

        captured_snippets: Dict[str, List[Dict[str, str]]] = {}

        def fake_call_llm(payload, snippets):
            captured_snippets["value"] = list(snippets)
            return {"llm": "ok"}

        monkeypatch.setattr(orchestrator, "call_llm", fake_call_llm)

        instances: List[object] = []

        class FakeSearcher:
            def __init__(self, policy):
                self.policy = policy
                self.search_calls: List[str] = []
                self.fetch_calls: List[str] = []
                instances.append(self)

            async def search(self, query: str):
                self.search_calls.append(query)
                return [
                    {"url": "https://allowed.com/a", "title": "Allowed"},
                    {"url": "https://denied.com/b", "title": "Denied"},
                    {"url": "https://other.com/c", "title": "Other"},
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
            "allow_domains": r"allowed\.com",
            "deny_domains": [r"denied\.com"],
        }

        data = await orchestrator.orchestrate_llm_trip(payload)

        assert instances, "WebSearcher should have been instantiated"
        policy = instances[0].policy
        assert list(policy.allow_domains or []) == [r"allowed\.com"]
        assert list(policy.deny_domains or []) == [r"denied\.com"]

        # Only allowed domain results should pass through.
        fetch_calls = instances[0].fetch_calls
        assert fetch_calls == ["https://allowed.com/a"]

        snippets = data.get("snippets")
        assert snippets is not None
        assert snippets == [
            {
                "url": "https://allowed.com/a",
                "title": "Title for https://allowed.com/a",
                "text": "Snippet text",
                "source": "web_fetch",
            }
        ]
        assert captured_snippets["value"] == snippets

    asyncio.run(run())


def test_orchestrator_foundation_preserves_user_extras(monkeypatch):
    async def run() -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from app import orchestrator

        captured: Dict[str, Any] = {}

        def fake_call_llm(payload, snippets):
            captured["agent_context"] = payload.get("agent_context")
            return {"llm": "ok"}

        monkeypatch.setattr(orchestrator, "call_llm", fake_call_llm)
        monkeypatch.setattr(orchestrator, "WebSearcher", None)

        payload = {
            "origin": "Lisbon",
            "destinations": ["Porto"],
            "dates": {"start": "2025-03-01", "end": "2025-03-05"},
            "budget_total": 1800.0,
            "currency": "EUR",
            "party": {"adults": 2, "children": 1},
            "interests": ["food tours", "history"],
            "constraints": {"diet": ["vegetarian"], "max_flight_hours": 6},
        }

        data = await orchestrator.orchestrate_llm_trip(payload)

        agent_context = data.get("agent_context")
        assert agent_context is not None

        foundation = agent_context.get("foundation")
        assert foundation
        assert foundation.get("interests") == payload["interests"]
        assert foundation.get("constraints") == payload["constraints"]

        captured_ctx = captured.get("agent_context")
        assert captured_ctx is not None
        assert captured_ctx.get("foundation", {}).get("interests") == payload["interests"]
        assert captured_ctx.get("foundation", {}).get("constraints") == payload["constraints"]

    asyncio.run(run())


def test_orchestrator_llm_backfill_updates_destination(monkeypatch):
    async def run() -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        from app import orchestrator

        def fake_call_llm(payload, snippets):
            return {"llm": "ok"}

        monkeypatch.setattr(orchestrator, "call_llm", fake_call_llm)
        monkeypatch.setattr(orchestrator, "WebSearcher", None)

        captured_cities: Dict[str, Any] = {}

        def fake_llm_backfill(cities, foundation, model="gpt-4o-mini"):
            captured_cities["value"] = {"cities": list(cities), "model": model}
            return {
                "Paris": {
                    "highlights": ["LLM highlight"],
                    "experiences": ["LLM experience"],
                    "dining": ["LLM dining"],
                    "notes": ["Indicative: LLM note"],
                }
            }

        monkeypatch.setattr(orchestrator, "llm_backfill_city_details", fake_llm_backfill)

        payload = {
            "origin": "Lisbon",
            "destinations": ["Paris"],
            "dates": {"start": "2025-03-01", "end": "2025-03-05"},
            "budget_total": 1800.0,
            "currency": "EUR",
            "party": {"adults": 2, "children": 1},
            "prefs": {"objective": "balanced"},
        }

        data = await orchestrator.orchestrate_llm_trip(payload)

        assert captured_cities["value"]["cities"] == ["Paris"]
        agent_context = data.get("agent_context")
        assert agent_context is not None
        destinations = agent_context.get("destinations")
        assert destinations
        assert destinations[0]["highlights"] == ["LLM highlight"]
        assert destinations[0]["experiences"] == ["LLM experience"]
        assert destinations[0]["dining"] == ["LLM dining"]
        assert agent_context.get("llm_sources")
        assert any(src.get("model") == captured_cities["value"]["model"] for src in agent_context["llm_sources"])
        assert "llm://" in " ".join(data.get("source_links", []))
        costs = agent_context.get("costs")
        assert costs and set(["stays", "transport", "food", "total", "budget"]).issubset(costs.keys())

    asyncio.run(run())

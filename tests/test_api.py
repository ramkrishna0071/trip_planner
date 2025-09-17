from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.main import app


def _sample_payload() -> dict:
    return {
        "origin": "San Francisco",
        "destinations": ["Paris", "Amsterdam"],
        "dates": {"start": "2025-10-10", "end": "2025-10-20"},
        "budget_total": 4500,
        "currency": "USD",
        "party": {"adults": 2, "children": 1, "seniors": 0},
        "prefs": {"objective": "balanced"},
    }


def test_api_plan_endpoint(monkeypatch):
    client = TestClient(app)
    orchestrator = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr("app.main.orchestrate_llm_trip", orchestrator)

    response = client.post("/api/plan", json=_sample_payload())

    assert response.status_code == 200
    orchestrator.assert_awaited_once()
    called_payload = orchestrator.await_args.args[0]
    assert called_payload["purpose"] == "leisure"
    assert response.json() == {"status": "ok"}


def test_trip_llm_only_endpoint(monkeypatch):
    client = TestClient(app)
    orchestrator = AsyncMock(return_value={"bundle_count": 3})
    monkeypatch.setattr("app.main.orchestrate_llm_trip", orchestrator)

    response = client.post(
        "/trip/llm_only",
        json={
            "origin": "San Francisco",
            "purpose": "family vacation",
            "budget_total": 5000,
            "currency": "USD",
            "dates": {"start": "2025-10-10", "end": "2025-10-20"},
            "party": {"adults": 2, "children": 1, "seniors": 0},
            "constraints": {},
            "interests": [],
            "destinations": ["Paris", "Amsterdam"],
        },
    )

    assert response.status_code == 200
    orchestrator.assert_awaited_once()
    assert response.json() == {"bundle_count": 3}

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.main import app

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_frontend_payload(form_values: dict) -> dict:
    if shutil.which("node") is None:
        pytest.skip("Node.js is required to execute the frontend payload builder")

    script = f"""
import {{ buildRequestPayload }} from './trip_planner_frontend/src/lib/payload.js';
const formValues = {json.dumps(form_values)};
const payload = buildRequestPayload(formValues);
console.log(JSON.stringify(payload));
"""

    result = subprocess.run(  # noqa: S603
        ["node", "--input-type=module", "-e", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_frontend_payload_passes_through_api(monkeypatch):
    payload = _build_frontend_payload(
        {
            "origin": "Los Angeles",
            "destinations": ["Kyoto", "Tokyo"],
            "startDate": "2026-03-01",
            "endDate": "2026-03-11",
            "budget": 6800,
            "adults": 2,
            "children": 2,
            "seniors": 1,
            "objective": "family_friendly",
        }
    )

    orchestrator = AsyncMock(return_value={"status": "ok", "bundle_count": 2})
    monkeypatch.setattr("app.main.orchestrate_llm_trip", orchestrator)

    client = TestClient(app)
    response = client.post("/api/plan", json=payload)

    assert response.status_code == 200
    orchestrator.assert_awaited_once()

    called_payload = orchestrator.await_args.args[0]
    assert called_payload["origin"] == "Los Angeles"
    assert called_payload["dates"]["start"] == "2026-03-01"
    assert called_payload["dates"]["end"] == "2026-03-11"
    assert called_payload["purpose"] == "family vacation"
    assert called_payload["party"]["seniors"] == 1
    assert called_payload["prefs"]["objective"] == "family_friendly"

    assert response.json() == {"status": "ok", "bundle_count": 2}

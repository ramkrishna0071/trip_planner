import os

os.environ.setdefault("OPENAI_API_KEY", "test")

from app.schemas import TripRequest
from app.orchestrator import plan_trip

SAMPLE_PAYLOAD = {
    "origin": "San Francisco",
    "purpose": "family vacation",
    "budget_total": 4500,
    "currency": "USD",
    "dates": {"start": "2025-07-10", "end": "2025-07-20"},
    "party": {"adults": 2, "children": 1, "seniors": 0},
    "constraints": {"diet": ["vegetarian"], "max_flight_hours": 12},
    "interests": ["culture", "food", "museums", "kid-friendly"],
    "destinations": ["Paris", "Amsterdam", "Berlin"],
}


def test_trip_request_budget_total_round_trip():
    trip_req = TripRequest.model_validate(SAMPLE_PAYLOAD)

    assert trip_req.budget_total == SAMPLE_PAYLOAD["budget_total"]

    dumped = trip_req.model_dump()
    assert dumped["budget_total"] == SAMPLE_PAYLOAD["budget_total"]

    response = plan_trip(trip_req)
    echoed = response.query_echo
    assert echoed.budget_total == SAMPLE_PAYLOAD["budget_total"]

    echoed_dump = response.model_dump(mode="json")
    assert echoed_dump["query_echo"]["budget_total"] == SAMPLE_PAYLOAD["budget_total"]


def test_trip_request_accepts_legacy_budget_field():
    legacy_payload = {**SAMPLE_PAYLOAD}
    legacy_payload.pop("budget_total")
    legacy_payload["budget"] = 3200

    trip_req = TripRequest.model_validate(legacy_payload)
    assert trip_req.budget_total == 3200

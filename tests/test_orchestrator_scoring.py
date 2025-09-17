from app.orchestrator import plan_trip
from app.schemas import TripRequest, TripPrefs, Dates, Party


def _build_request(objective: str, **prefs_kwargs) -> TripRequest:
    prefs = TripPrefs(objective=objective, **prefs_kwargs)
    return TripRequest(
        origin="New York",
        destinations=["Paris", "Berlin"],
        dates=Dates(start="2024-06-01", end="2024-06-07"),
        budget_total=5200.0,
        currency="USD",
        party=Party(adults=2),
        prefs=prefs,
    )


def test_balanced_objective_prioritises_balanced_plan():
    response = plan_trip(_build_request("balanced"))
    assert response.options[0].label == "balanced"
    assert "Scores —" in response.options[0].summary
    assert response.options[0].scores["composite"] >= response.options[1].scores["composite"]


def test_cheapest_objective_promotes_budget_plan():
    response = plan_trip(_build_request("cheapest"))
    assert response.options[0].label == "cheapest"
    assert response.options[0].scores["cost"] >= response.options[1].scores["cost"]


def test_comfort_objective_prefers_fastest_plan():
    response = plan_trip(_build_request("comfort"))
    assert response.options[0].label == "comfort"
    comfort = next(opt for opt in response.options if opt.label == "comfort")
    cheapest = next(opt for opt in response.options if opt.label == "cheapest")
    assert comfort.scores["time"] >= cheapest.scores["time"]


def test_max_flight_hours_penalises_comfort_plan():
    response = plan_trip(_build_request("comfort", max_flight_hours=0.2))
    assert response.options[0].label != "comfort"
    comfort = next(opt for opt in response.options if opt.label == "comfort")
    assert comfort.scores["time"] < 0.8

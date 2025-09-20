"""Regression tests for destination scouting heuristics."""

from app.agents.destination_scout import _extract_points


def test_extract_points_retains_dont_miss_landmark():
    snippet = "Don't miss Disneyland Paris for a day of magic with the family."
    highlights, dining = _extract_points(snippet, "Paris")

    assert snippet in highlights
    assert dining == []


def test_extract_points_accepts_landmark_without_city_when_action_present():
    snippet = "Don't miss the sweeping views from Table Mountain at sunset."
    highlights, _ = _extract_points(snippet, "Cape Town")

    assert snippet in highlights

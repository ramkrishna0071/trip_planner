"""Utility agent that normalizes raw trip payloads."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List


def extract_foundation(payload: Any) -> Dict[str, Any]:
    """Return derived trip fundamentals from a raw payload or TripRequest.

    The goal of this helper is to expose a predictable structure that other
    agents (destination scouts, logistics planners, the LLM orchestrator) can
    build on top of without re-implementing basic calculations.  It accepts the
    "raw" request payload, a ``TripRequest`` instance, or anything that exposes
    ``model_dump`` similar to Pydantic models.
    """
    raw: Dict[str, Any]
    if hasattr(payload, "model_dump"):
        raw = payload.model_dump(mode="python")  # type: ignore[assignment]
    elif isinstance(payload, dict):
        raw = dict(payload)
    else:
        raise TypeError("Unsupported payload type for foundation extraction")

    origin = raw.get("origin", "")
    destinations: List[str] = [str(c) for c in raw.get("destinations", [])]
    currency = raw.get("currency", "USD")
    budget_total = float(raw.get("budget_total") or raw.get("budget") or 0.0)

    dates_raw = raw.get("dates", {}) or {}
    start = str(dates_raw.get("start", ""))
    end = str(dates_raw.get("end", start)) if dates_raw else start

    start_dt, end_dt = _safe_parse(start), _safe_parse(end)
    if start_dt and end_dt and end_dt < start_dt:
        # swap to avoid negative durations
        start_dt, end_dt = end_dt, start_dt
    duration_days = _duration_days(start_dt, end_dt)
    nights = max(0, duration_days - 1)
    n_stops = max(1, len(destinations))
    nights_allocation = _split_evenly(max(1, nights), n_stops)

    party = raw.get("party", {}) or {}
    adults = int(party.get("adults", 0))
    children = int(party.get("children", 0))
    seniors = int(party.get("seniors", 0))
    party_size = max(1, adults + children + seniors)

    prefs = raw.get("prefs", {}) or {}
    flexible_days = int(prefs.get("flexible_days", 0) or 0)
    max_flight_hours = prefs.get("max_flight_hours")

    per_day_budget = budget_total / duration_days if duration_days else 0.0
    per_person_per_day = budget_total / (duration_days * party_size) if duration_days and party_size else 0.0

    notes: List[str] = []
    if per_person_per_day and per_person_per_day < 60:
        notes.append("Budget per person per day is tight; prioritize value options.")
    if flexible_days:
        notes.append(f"User can flex {flexible_days} day(s) if needed.")
    if max_flight_hours:
        notes.append(f"Limit flight segments to around {max_flight_hours} hour(s).")

    season = _infer_season(start_dt)

    calendar: List[Dict[str, Any]] = []
    if start_dt:
        cursor = start_dt
        for idx, city in enumerate(destinations or [origin]):
            stay_nights = nights_allocation[idx] if idx < len(nights_allocation) else 1
            for _ in range(stay_nights or 1):
                calendar.append({
                    "date": cursor.isoformat(),
                    "city": city,
                })
                cursor += timedelta(days=1)

    foundation: Dict[str, Any] = {
        "origin": origin,
        "destinations": destinations,
        "currency": currency,
        "purpose": raw.get("purpose"),
        "dates": {
            "start": start,
            "end": end,
            "duration_days": duration_days,
            "nights": nights,
            "season": season,
        },
        "party": {
            "adults": adults,
            "children": children,
            "seniors": seniors,
            "size": party_size,
        },
        "budget": {
            "total": budget_total,
            "per_day": round(per_day_budget, 2) if per_day_budget else 0.0,
            "per_person_per_day": round(per_person_per_day, 2) if per_person_per_day else 0.0,
            "currency": currency,
        },
        "preferences": {
            "objective": prefs.get("objective", "balanced"),
            "flexible_days": flexible_days,
            "max_flight_hours": max_flight_hours,
            "diet": list(prefs.get("diet", [])) if prefs.get("diet") else [],
            "mobility": prefs.get("mobility", "normal"),
        },
        "constraints": raw.get("constraints", {}),
        "interests": list(raw.get("interests", [])) if raw.get("interests") else [],
        "nights_allocation": nights_allocation,
        "calendar": calendar,
        "notes": notes,
    }

    return foundation


def _safe_parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return None


def _duration_days(start: datetime | None, end: datetime | None) -> int:
    if not start or not end:
        return 1
    return (end - start).days + 1 if end >= start else 1


def _split_evenly(total: int, buckets: int) -> List[int]:
    if buckets <= 0:
        return []
    base = [total // buckets] * buckets
    remainder = total % buckets
    for i in range(remainder):
        base[i] += 1
    return base


def _infer_season(start: datetime | None) -> str | None:
    if not start:
        return None
    month = start.month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"

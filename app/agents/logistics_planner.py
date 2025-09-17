"""Logistics planning helpers."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List


def compute_logistics(foundation: Dict[str, Any]) -> Dict[str, Any]:
    """Propose transfer legs, buffers, and a light-touch schedule."""
    destinations: List[str] = list(foundation.get("destinations", []))
    nights_alloc = list(foundation.get("nights_allocation", []))
    dates = foundation.get("dates", {}) or {}
    start = dates.get("start")

    timeline: List[Dict[str, Any]] = []
    legs: List[Dict[str, Any]] = []
    transfer_buffers: Dict[str, float] = {}
    feasibility_notes: List[str] = []

    start_dt = _safe_parse(start)
    cursor = start_dt

    for idx, city in enumerate(destinations):
        stay_nights = nights_alloc[idx] if idx < len(nights_alloc) else 1
        if stay_nights <= 0:
            stay_nights = 1
        arrival = cursor.isoformat() if cursor else None
        departure_dt = cursor + timedelta(days=stay_nights) if cursor else None
        departure = departure_dt.isoformat() if departure_dt else None
        timeline.append({
            "city": city,
            "arrival": arrival,
            "departure": departure,
            "nights": stay_nights,
        })
        if departure_dt:
            cursor = departure_dt

    if len(destinations) <= 1:
        if destinations:
            feasibility_notes.append("Single-destination trip: no intercity transfers required.")
        return {
            "timeline": timeline,
            "legs": [],
            "transfer_buffers": transfer_buffers,
            "feasibility_notes": feasibility_notes,
        }

    for idx in range(len(destinations) - 1):
        frm = destinations[idx]
        to = destinations[idx + 1]
        stay_nights = nights_alloc[idx] if idx < len(nights_alloc) else 1
        depart_dt = start_dt + timedelta(days=sum(nights_alloc[: idx + 1])) if start_dt else None
        depart_date = depart_dt.date().isoformat() if depart_dt else None

        mode = _infer_mode(foundation, frm, to)
        duration_hr = _estimate_duration(mode)
        buffer_hr = _buffer_for(mode)
        cost_estimate = _cost_estimate(mode)

        leg = {
            "from": frm,
            "to": to,
            "mode": mode,
            "depart_date": depart_date,
            "duration_hr": duration_hr,
            "buffer_hr": buffer_hr,
            "cost_estimate": cost_estimate,
            "suggested_window": "morning" if mode in {"train", "bus"} else "midday",
        }
        legs.append(leg)
        transfer_buffers[f"{frm}->{to}"] = buffer_hr
        feasibility_notes.append(
            f"Plan {mode} from {frm} to {to} (depart {depart_date or 'TBD'}) allowing ~{buffer_hr}h buffer."
        )
        if stay_nights <= 1:
            feasibility_notes.append(f"Short stay in {frm}; pack light for quick transfer to {to}.")

    return {
        "timeline": timeline,
        "legs": legs,
        "transfer_buffers": transfer_buffers,
        "feasibility_notes": feasibility_notes,
    }


def _infer_mode(foundation: Dict[str, Any], frm: str, to: str) -> str:
    objective = foundation.get("preferences", {}).get("objective", "balanced")
    if objective == "comfort":
        return "flight"
    if objective == "cheapest":
        return "bus"
    if len(frm) > 10 or len(to) > 10:
        return "flight"
    return "train"


def _estimate_duration(mode: str) -> float:
    if mode == "flight":
        return 2.5
    if mode == "bus":
        return 5.0
    if mode == "car":
        return 4.0
    return 3.0


def _buffer_for(mode: str) -> float:
    if mode == "flight":
        return 2.0
    if mode == "bus":
        return 0.75
    if mode == "car":
        return 0.5
    return 1.0


def _cost_estimate(mode: str) -> float:
    if mode == "flight":
        return 160.0
    if mode == "bus":
        return 45.0
    if mode == "car":
        return 90.0
    return 85.0


def _safe_parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        try:
            return datetime.strptime(str(value), "%Y-%m-%d")
        except ValueError:
            return None

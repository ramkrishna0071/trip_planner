# app/orchestrator.py
from __future__ import annotations

from typing import List, Dict, Any
from collections.abc import Iterable
from datetime import datetime
import asyncio

from app.schemas import (
    TripRequest,
    TripResponse,
    PlanBundle,
    TravelLeg,
    Stay,
    DayPlan,
    AgentContext,
    TripPrefs,
)
from app.llm import call_llm  # import at module top to avoid circulars
from app.agents.foundation_agent import extract_foundation
from app.agents.destination_scout import expand_destinations
from app.agents.logistics_planner import compute_logistics

# If your websearch tool isn't ready, these imports will be skipped gracefully.
try:
    from app.tools.websearch import WebSearcher, SourcePolicy
except Exception:
    WebSearcher = None  # type: ignore
    class SourcePolicy:  # fallback stub
        def __init__(self, allow_domains=None, deny_domains=None, max_results=6, max_per_domain=2, recency_days=365):
            self.allow_domains = allow_domains or []
            self.deny_domains = deny_domains or []
            self.max_results = max_results
            self.max_per_domain = max_per_domain
            self.recency_days = recency_days
        def allowed(self, url: str) -> bool:
            from urllib.parse import urlparse
            host = urlparse(url).hostname or ""
            if any(host.endswith(d) for d in self.deny_domains):
                return False
            return (not self.allow_domains) or any(host.endswith(d) for d in self.allow_domains)

# ---------- baseline generic planner (no web, fast) ----------
def plan_trip(
    req: TripRequest,
    context: AgentContext | Dict[str, Any] | None = None,
) -> TripResponse:
    """Produce baseline bundles informed by optional agent context."""

    if isinstance(context, AgentContext):
        ctx: Dict[str, Any] = context.model_dump(mode="python")
    elif isinstance(context, dict):
        ctx = dict(context)
    elif context is None:
        ctx = {}
    else:
        ctx = dict(context)  # type: ignore[arg-type]

    foundation_ctx = ctx.get("foundation", {}) or {}
    logistics_ctx = ctx.get("logistics", {}) or {}
    dest_ctx_list = ctx.get("destinations", []) or []

    combined_notes: List[str] = []
    for bucket in (
        foundation_ctx.get("notes", []),
        ctx.get("notes", []),
        logistics_ctx.get("feasibility_notes", []),
    ):
        for item in bucket or []:
            if item not in combined_notes:
                combined_notes.append(item)

    trip_days = int(foundation_ctx.get("dates", {}).get("duration_days") or max(1, _days(req.dates.start, req.dates.end)))
    nights = int(foundation_ctx.get("dates", {}).get("nights") or max(1, trip_days - 1))
    n_stops = max(1, len(req.destinations))

    nights_allocation = list(foundation_ctx.get("nights_allocation", [])) if foundation_ctx else []
    if len(nights_allocation) != n_stops:
        nights_allocation = _split_nights(nights, n_stops)

    party_size = req.party.adults + req.party.children + req.party.seniors
    ppd_val = float(foundation_ctx.get("budget", {}).get("per_person_per_day") or _ppd(req.budget_total, trip_days, party_size))

    stays = [
        Stay(city=city, nights=n, style="hotel", budget_per_night=ppd_val * 0.4 * party_size)
        for city, n in zip(req.destinations, nights_allocation)
    ]

    dest_map: Dict[str, Dict[str, Any]] = {
        d.get("city"): d for d in dest_ctx_list if isinstance(d, dict) and d.get("city")
    }
    experiences = _day_plans(req.destinations, nights_allocation, dest_map)

    logistic_legs = logistics_ctx.get("legs") or []
    if logistic_legs:
        intercity = []
        for idx, leg in enumerate(logistic_legs):
            frm = leg.get("from") or leg.get("frm") or (req.destinations[idx] if idx < len(req.destinations) else req.destinations[0])
            to = leg.get("to") or (req.destinations[idx + 1] if idx + 1 < len(req.destinations) else req.destinations[-1])
            leg_payload = {
                "mode": leg.get("mode", "train"),
                "from": frm,
                "to": to,
                "date": leg.get("depart_date"),
                "duration_hr": leg.get("duration_hr"),
                "cost_estimate": leg.get("cost_estimate"),
            }
            intercity.append(TravelLeg(**leg_payload))
    else:
        intercity = _legs_for(req.destinations)

    base_cost_stay = sum(s.nights * s.budget_per_night for s in stays)
    base_cost_move = sum(l.cost_estimate or 0 for l in intercity)
    food_misc = trip_days * party_size * ppd_val * 0.35

    cheapest_stays = [
        Stay(**{**s.model_dump(), "style": "homestay", "budget_per_night": s.budget_per_night * 0.75})
        for s in stays
    ]
    cheapest = PlanBundle(
        label="cheapest",
        summary="Minimize cost using homestays and public transport.",
        total_cost=base_cost_move + sum(s.nights * s.budget_per_night for s in cheapest_stays) + food_misc * 0.9,
        currency=req.currency,
        transfers=max(1, len(intercity)),
        est_duration_days=trip_days,
        travel=[*intercity],
        stays=cheapest_stays,
        local_transport=["metro/day-pass", "bus", "rideshare (as needed)"],
        experience_plan=experiences,
        notes=["Indicative prices; replace with live quotes when adapters are connected."] + combined_notes,
        feasibility_notes=combined_notes,
        transfer_buffers=dict(logistics_ctx.get("transfer_buffers", {})),
    )

    comfort_legs = [
        (TravelLeg(mode="flight", **{"from": l.frm, "to": l.to}, duration_hr=1.0,
                   cost_estimate=(l.cost_estimate or 80) * 2.2) if l.mode in ("train", "bus") else l)
        for l in intercity
    ]
    comfort_stays = [
        Stay(**{**s.model_dump(), "style": "boutique", "budget_per_night": s.budget_per_night * 1.5})
        for s in stays
    ]
    comfort = PlanBundle(
        label="comfort",
        summary="Fewer transfers, faster hops, nicer stays.",
        total_cost=sum(s.nights * s.budget_per_night for s in comfort_stays)
                   + sum(l.cost_estimate or 0 for l in comfort_legs) + food_misc * 1.1,
        currency=req.currency,
        transfers=max(0, len(comfort_legs)),
        est_duration_days=trip_days,
        travel=comfort_legs,
        stays=comfort_stays,
        local_transport=["hotel pickup", "metro", "rideshare"],
        experience_plan=experiences,
        notes=["Upgrade selected hops to flights; swap stays to boutique."] + combined_notes,
        feasibility_notes=combined_notes,
        transfer_buffers=dict(logistics_ctx.get("transfer_buffers", {})),
    )

    balanced = PlanBundle(
        label="balanced",
        summary="Balanced cost, pace and comfort.",
        total_cost=base_cost_stay + base_cost_move + food_misc,
        currency=req.currency,
        transfers=len(intercity),
        est_duration_days=trip_days,
        travel=intercity,
        stays=stays,
        local_transport=["metro", "IC card", "occasional rideshare"],
        experience_plan=experiences,
        notes=["Good default; tune by preferences (diet, mobility)."] + combined_notes,
        feasibility_notes=combined_notes,
        transfer_buffers=dict(logistics_ctx.get("transfer_buffers", {})),
    )

    options = [balanced, cheapest, comfort]
    metrics = [_plan_metrics(bundle, req.budget_total) for bundle in options]
    base_scores = _build_normalized_scores(metrics)

    scored_options: List[PlanBundle] = []
    for bundle, metric_snapshot, normalized in zip(options, metrics, base_scores):
        adjusted = _apply_preferences(bundle, metric_snapshot, normalized, req.prefs)
        bundle.scores = {
            "cost": round(adjusted["cost"], 4),
            "time": round(adjusted["time"], 4),
            "experience": round(adjusted["experience"], 4),
            "composite": round(adjusted["composite"], 4),
            "budget_utilization": round(metric_snapshot["budget_utilization"], 4),
            "transit_hours": round(metric_snapshot["transit_hours"], 2),
            "experience_density": round(metric_snapshot["experience_density"], 2),
            "flight_hours": round(metric_snapshot["flight_hours"], 2),
            "avg_flex_hours": round(metric_snapshot["avg_flex_hours"], 2),
        }
        _attach_score_notes(bundle)
        scored_options.append(bundle)

    options = sorted(scored_options, key=lambda b: b.scores.get("composite", 0.0), reverse=True)

    agent_context_model = AgentContext.model_validate(ctx) if ctx else None
    return TripResponse(query_echo=req, options=options, agent_context=agent_context_model)

# ---------- LLM + (optional) web sources ----------
async def orchestrate_llm_trip(payload: Dict[str, Any],
                               allow_domains: List[str] | None = None,
                               deny_domains: List[str] | None = None) -> Dict[str, Any]:
    """Build queries, optionally web-search + fetch snippets, call LLM, return dict."""
    trip_req = TripRequest.model_validate(payload)
    foundation = extract_foundation(trip_req)

    if "interests" in payload:
        raw_interests = payload.get("interests")
        if isinstance(raw_interests, str):
            foundation["interests"] = [raw_interests]
        elif isinstance(raw_interests, Iterable):
            foundation["interests"] = [str(item) for item in raw_interests if item is not None]
        else:
            foundation["interests"] = [str(raw_interests)] if raw_interests is not None else []

    if "constraints" in payload:
        raw_constraints = payload.get("constraints")
        if isinstance(raw_constraints, dict):
            foundation["constraints"] = raw_constraints
        elif raw_constraints is None:
            foundation["constraints"] = {}
        else:
            try:
                foundation["constraints"] = dict(raw_constraints)
            except Exception:
                foundation["constraints"] = {"value": raw_constraints}

    resolved_allow = _normalize_domain_list(
        allow_domains if allow_domains is not None else payload.get("allow_domains")
    )
    resolved_deny = _normalize_domain_list(
        deny_domains if deny_domains is not None else payload.get("deny_domains")
    )

    # Build queries from payload
    queries: List[str] = []
    dests: List[str] = list(foundation.get("destinations", [])) or trip_req.destinations
    if len(dests) >= 2:
        for a, b in zip(dests[:-1], dests[1:]):
            queries.append(f"how to travel {a} to {b} train flight bus cost")
    for c in dests[:3]:
        queries += [f"top attractions {c} family friendly", f"{c} city pass prices"]
    if dests:
        origin = trip_req.origin
        queries.append(f"visa requirements {origin} to {dests[-1]} official")

    # Build snippets (empty if WebSearcher not available)
    snippets: List[Dict[str, str]] = []
    if WebSearcher is not None:
        policy = SourcePolicy(
            allow_domains=resolved_allow,
            deny_domains=resolved_deny,
            max_results=6,
            max_per_domain=2,
        )
        searcher = WebSearcher(policy)
        # SEARCH
        results: List[Dict[str, str]] = []
        for q in queries:
            try:
                hits = await searcher.search(q)
                results.extend(hits)
            except Exception:
                continue
        # DEDUPE / FILTER
        seen, per_domain, pruned = set(), {}, []
        for r in results:
            url = r.get("url", "")
            if not url or url in seen or not policy.allowed(url):
                continue
            dom = url.split("/")[2] if "://" in url else url
            if per_domain.get(dom, 0) >= policy.max_per_domain:
                continue
            seen.add(url)
            per_domain[dom] = per_domain.get(dom, 0) + 1
            pruned.append(r)
        # FETCH
        fetch_tasks = [searcher.fetch(r["url"]) for r in pruned[:policy.max_results]]
        docs = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for d in docs:
            if getattr(d, "url", None):
                snippets.append({"url": d.url, "title": getattr(d, "title", "") or "", "text": getattr(d, "text", "") or ""})

    destination_context = expand_destinations(foundation, snippets)
    logistics_context = compute_logistics(foundation)

    aggregated_notes = list({note: None for note in (
        *(foundation.get("notes", []) or []),
        *(destination_context.get("notes", []) or []),
        *(logistics_context.get("feasibility_notes", []) or []),
    )}.keys())

    agent_context: Dict[str, Any] = {
        "foundation": foundation,
        "destinations": destination_context.get("destinations", []),
        "logistics": logistics_context,
        "notes": aggregated_notes,
        "sources": destination_context.get("sources", []),
        "snippets": snippets,
    }

    baseline_plan = plan_trip(trip_req, agent_context)
    baseline_dump = baseline_plan.model_dump(mode="json")

    # Call LLM (your llm.py returns a dict already)
    try:
        enriched_payload = dict(payload)
        enriched_payload["agent_context"] = agent_context
        data = call_llm(enriched_payload, snippets)
    except Exception as exc:
        return {
            "baseline_plan": baseline_dump,
            "snippets": snippets,
            "agent_context": agent_context,
            "llm_error": str(exc),
        }

    if isinstance(data, dict):
        data.setdefault("baseline_plan", baseline_dump)
        data.setdefault("snippets", snippets)
        data.setdefault("agent_context", agent_context)
        return data

    return {
        "llm_raw": data,
        "baseline_plan": baseline_dump,
        "snippets": snippets,
        "agent_context": agent_context,
    }

# ---------- helpers ----------
def _normalize_domain_list(domains: Any | Iterable | None) -> List[str]:
    if domains is None:
        return []
    if isinstance(domains, str):
        return [domains]
    if isinstance(domains, Iterable):
        return [str(item) for item in domains if item is not None]
    return [str(domains)]


def _split_nights(total_nights: int, n_stops: int) -> List[int]:
    base = [total_nights // n_stops] * n_stops
    for i in range(total_nights % n_stops):
        base[i] += 1
    return base

def _ppd(budget_total: float, days: int, party_size: int) -> float:
    return max(1.0, budget_total / max(1, days) / max(1, party_size))

def _day_plans(
    cities: List[str],
    nights_each: List[int],
    dest_context: Dict[str, Dict[str, Any]] | None = None,
) -> List[DayPlan]:
    out: List[DayPlan] = []
    dest_context = dest_context or {}
    for city, n in zip(cities, nights_each):
        info = dest_context.get(city, {})
        must_do = info.get("highlights") or [f"Top sights in {city}"]
        experiences = info.get("experiences") or []
        dining = info.get("dining") or []
        for day in range(max(1, n)):
            hidden: List[str] = []
            if experiences:
                hidden.append(experiences[min(day, len(experiences) - 1)])
            if dining:
                hidden.append(dining[min(day, len(dining) - 1)])
            if not hidden:
                hidden = [f"Neighborhood food lane in {city}"]
            out.append(
                DayPlan(
                    city=city,
                    must_do=must_do[:3],
                    hidden_gem=hidden,
                    flex_hours=2 if experiences else 3,
                )
            )
    return out

def _legs_for(cities: List[str]) -> List[TravelLeg]:
    legs: List[TravelLeg] = []
    if len(cities) <= 1:
        return legs
    for a, b in zip(cities[:-1], cities[1:]):
        legs.append(TravelLeg(mode="train", **{"from": a, "to": b}, duration_hr=3.0, cost_estimate=80.0))
    return legs

def _days(start: str, end: str) -> int:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    return (e - s).days + 1


_DEFAULT_MODE_DURATION = {
    "train": 3.0,
    "bus": 4.0,
    "flight": 2.0,
    "car": 3.5,
    "rideshare": 1.2,
    "metro": 0.6,
    "walk": 0.4,
    "ferry": 2.5,
}

_STAY_STYLE_QUALITY = {
    "homestay": 0.45,
    "hotel": 0.65,
    "apartment": 0.7,
    "boutique": 0.85,
}

_OBJECTIVE_WEIGHTS = {
    "balanced": {"cost": 1.4, "time": 0.6, "experience": 1.2},
    "cheapest": {"cost": 1.8, "time": 0.6, "experience": 0.6},
    "comfort": {"cost": 0.8, "time": 1.6, "experience": 1.1},
    "family_friendly": {"cost": 1.0, "time": 0.9, "experience": 1.5},
}

_OBJECTIVE_COST_MIX = {
    "balanced": (0.4, 0.6),
    "cheapest": (0.25, 0.75),
    "comfort": (0.8, 0.2),
    "family_friendly": (0.5, 0.5),
}


def _plan_metrics(bundle: PlanBundle, budget_total: float) -> Dict[str, float]:
    transit_hours = 0.0
    flight_hours = 0.0
    for leg in bundle.travel:
        duration = leg.duration_hr
        if duration is None:
            duration = _DEFAULT_MODE_DURATION.get(leg.mode, 3.0)
        transit_hours += duration
        if leg.mode == "flight":
            flight_hours += duration

    if bundle.experience_plan:
        total_items = sum(len(day.must_do) + len(day.hidden_gem) for day in bundle.experience_plan)
        experience_density = total_items / max(1, len(bundle.experience_plan))
        avg_flex_hours = sum(day.flex_hours for day in bundle.experience_plan) / max(1, len(bundle.experience_plan))
    else:
        experience_density = 0.0
        avg_flex_hours = 0.0

    budget_utilization = bundle.total_cost / budget_total if budget_total else 1.0
    stay_quality = _avg_stay_quality(bundle.stays)

    return {
        "budget_utilization": budget_utilization,
        "transit_hours": transit_hours,
        "experience_density": experience_density,
        "flight_hours": flight_hours,
        "transfers": bundle.transfers,
        "stay_quality": stay_quality,
        "avg_flex_hours": avg_flex_hours,
        "under_budget_ratio": max(0.0, 1.0 - min(budget_utilization, 1.0)),
    }


def _build_normalized_scores(metrics: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not metrics:
        return []

    alignment_vals = [abs(m["budget_utilization"] - 1.0) for m in metrics]
    savings_vals = [m["budget_utilization"] for m in metrics]
    avg_util = sum(m["budget_utilization"] for m in metrics) / len(metrics)
    center_vals = [abs(m["budget_utilization"] - avg_util) for m in metrics]
    time_vals = [m["transit_hours"] + 0.5 * m["transfers"] for m in metrics]
    density_vals = [m["experience_density"] for m in metrics]
    quality_vals = [m["stay_quality"] for m in metrics]
    flex_vals = [m["avg_flex_hours"] for m in metrics]

    alignment_scores = _normalize(alignment_vals, invert=True)
    savings_scores = _normalize(savings_vals, invert=True)
    center_scores = _normalize(center_vals, invert=True)
    time_scores = _normalize(time_vals, invert=True)
    density_scores = _normalize(density_vals, invert=False)
    quality_scores = _normalize(quality_vals, invert=False)
    flex_scores = _normalize(flex_vals, invert=False)

    base: List[Dict[str, float]] = []
    for idx in range(len(metrics)):
        experience_score = _clamp(0.6 * density_scores[idx] + 0.4 * quality_scores[idx])
        base.append(
            {
                "cost_alignment": alignment_scores[idx],
                "cost_savings": savings_scores[idx],
                "cost_center": center_scores[idx],
                "time": time_scores[idx],
                "experience": experience_score,
                "quality": quality_scores[idx],
                "flex": flex_scores[idx],
            }
        )
    return base


def _apply_preferences(
    bundle: PlanBundle,
    metrics: Dict[str, float],
    base_scores: Dict[str, float],
    prefs: TripPrefs,
) -> Dict[str, float]:
    cost_mix = _OBJECTIVE_COST_MIX.get(prefs.objective, _OBJECTIVE_COST_MIX["balanced"])
    cost_score = _blend(base_scores["cost_alignment"], base_scores["cost_savings"], cost_mix)
    time_score = base_scores["time"]
    experience_score = base_scores["experience"]
    quality_score = base_scores["quality"]
    flex_score = base_scores["flex"]
    center_score = base_scores.get("cost_center", cost_score)

    weights = dict(_OBJECTIVE_WEIGHTS.get(prefs.objective, _OBJECTIVE_WEIGHTS["balanced"]))

    if prefs.objective == "cheapest":
        cost_score = _clamp(cost_score + metrics.get("under_budget_ratio", 0.0) * 0.7)
        time_score = min(time_score, 0.8)
    elif prefs.objective == "comfort":
        time_score = _clamp(time_score + quality_score * 0.15)
        experience_score = _clamp(experience_score + quality_score * 0.2)
    elif prefs.objective == "family_friendly":
        experience_score = _clamp(experience_score + flex_score * 0.25)
        time_score = _clamp(time_score + flex_score * 0.1)
    elif prefs.objective == "balanced":
        cost_score = _clamp(0.5 * cost_score + 0.5 * center_score)
        time_score = max(time_score, 0.4)

    if prefs.flexible_days:
        weights["time"] *= 1.0 / (1.0 + 0.05 * prefs.flexible_days)
        experience_score = _clamp(experience_score + min(0.15, 0.02 * prefs.flexible_days))

    if prefs.diet:
        weights["experience"] *= 1.0 + min(0.5, 0.1 * len(prefs.diet))

    if prefs.mobility == "step_free":
        penalty = min(0.6, 0.12 * metrics.get("transfers", 0.0))
        time_score = _clamp(time_score * (1.0 - penalty))
        weights["time"] *= 1.25
    elif prefs.mobility == "low_stairs":
        penalty = min(0.4, 0.08 * metrics.get("transfers", 0.0))
        time_score = _clamp(time_score * (1.0 - penalty))
        weights["time"] *= 1.1

    if prefs.max_flight_hours is not None:
        if metrics.get("flight_hours", 0.0) > prefs.max_flight_hours:
            over_ratio = (metrics["flight_hours"] - prefs.max_flight_hours) / max(prefs.max_flight_hours, 1.0)
            penalty = min(0.9, over_ratio * 0.5 + 0.2)
            time_score = _clamp(time_score * (1.0 - penalty))
            experience_score = _clamp(experience_score * (1.0 - penalty * 0.5))
            cost_score = _clamp(cost_score * (1.0 - penalty * 0.3))
            weights["time"] *= 1.1

    total_weight = sum(weights.values()) or 1.0
    composite = (
        weights["cost"] * cost_score
        + weights["time"] * time_score
        + weights["experience"] * experience_score
    ) / total_weight

    return {
        "cost": cost_score,
        "time": time_score,
        "experience": experience_score,
        "composite": composite,
    }


def _attach_score_notes(bundle: PlanBundle) -> None:
    if not bundle.scores:
        return

    score_text = (
        f"Scores — cost {bundle.scores.get('cost', 0.0):.2f}, "
        f"time {bundle.scores.get('time', 0.0):.2f}, "
        f"experience {bundle.scores.get('experience', 0.0):.2f}, "
        f"overall {bundle.scores.get('composite', 0.0):.2f}."
    )
    if "Scores —" not in bundle.summary:
        summary_base = bundle.summary.rstrip()
        if not summary_base.endswith("."):
            summary_base = f"{summary_base}."
        bundle.summary = f"{summary_base} {score_text}"

    note_text = (
        f"Metrics → transit_hours={bundle.scores.get('transit_hours', 0.0):.2f}, "
        f"budget_utilization={bundle.scores.get('budget_utilization', 0.0):.2f}, "
        f"experience_density={bundle.scores.get('experience_density', 0.0):.2f}"
    )
    if note_text not in bundle.notes:
        bundle.notes.append(note_text)


def _normalize(values: List[float], *, invert: bool) -> List[float]:
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val < 1e-9:
        return [1.0 for _ in values]
    scaled = [(val - min_val) / (max_val - min_val) for val in values]
    if invert:
        scaled = [1.0 - s for s in scaled]
    baseline = 0.2
    return [_clamp(baseline + (1.0 - baseline) * s) for s in scaled]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _avg_stay_quality(stays: List[Stay]) -> float:
    if not stays:
        return 0.6
    total = sum(_STAY_STYLE_QUALITY.get(stay.style, 0.6) for stay in stays)
    return total / len(stays)


def _blend(alignment: float, savings: float, mix: tuple[float, float]) -> float:
    a, b = mix
    denom = (a + b) or 1.0
    return _clamp((alignment * a + savings * b) / denom)

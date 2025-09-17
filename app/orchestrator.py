# app/orchestrator.py
from __future__ import annotations

from typing import List, Dict, Any
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
    if req.prefs.objective == "cheapest":
        options.sort(key=lambda o: o.total_cost)
    elif req.prefs.objective == "comfort":
        options.sort(key=lambda o: (o.transfers, -o.total_cost))
    else:
        options = [balanced, cheapest, comfort]

    agent_context_model = AgentContext.model_validate(ctx) if ctx else None
    return TripResponse(query_echo=req, options=options, agent_context=agent_context_model)

# ---------- LLM + (optional) web sources ----------
async def orchestrate_llm_trip(payload: Dict[str, Any],
                               allow_domains: List[str] | None = None,
                               deny_domains: List[str] | None = None) -> Dict[str, Any]:
    """Build queries, optionally web-search + fetch snippets, call LLM, return dict."""
    trip_req = TripRequest.model_validate(payload)
    foundation = extract_foundation(trip_req)

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
        policy = SourcePolicy(allow_domains=allow_domains, deny_domains=deny_domains, max_results=6, max_per_domain=2)
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

# app/orchestrator.py
from __future__ import annotations

from typing import List, Dict, Any
from datetime import datetime
import asyncio

from app.schemas import TripRequest, TripResponse, PlanBundle, TravelLeg, Stay, DayPlan
from app.llm import call_llm  # import at module top to avoid circulars

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
def plan_trip(req: TripRequest) -> TripResponse:
    trip_days = max(1, _days(req.dates.start, req.dates.end))
    party_size = req.party.adults + req.party.children + req.party.seniors
    nights = max(1, trip_days - 1)
    n_stops = max(1, len(req.destinations))
    nights_each = _split_nights(nights, n_stops)
    ppd_val = _ppd(req.budget, trip_days, party_size)

    stays = [
        Stay(city=city, nights=n, style="hotel", budget_per_night=ppd_val * 0.4 * party_size)
        for city, n in zip(req.destinations, nights_each)
    ]
    experiences = _day_plans(req.destinations, nights_each)
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
        notes=["Indicative prices; replace with live quotes when adapters are connected."]
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
        notes=["Upgrade selected hops to flights; swap stays to boutique."]
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
        notes=["Good default; tune by preferences (diet, mobility)."]
    )

    options = [balanced, cheapest, comfort]
    if req.prefs.objective == "cheapest":
        options.sort(key=lambda o: o.total_cost)
    elif req.prefs.objective == "comfort":
        options.sort(key=lambda o: (o.transfers, -o.total_cost))
    else:
        options = [balanced, cheapest, comfort]

    return TripResponse(query_echo=req, options=options)

# ---------- LLM + (optional) web sources ----------
async def orchestrate_llm_trip(payload: Dict[str, Any],
                               allow_domains: List[str] | None = None,
                               deny_domains: List[str] | None = None) -> Dict[str, Any]:
    """Build queries, optionally web-search + fetch snippets, call LLM, return dict."""
    # Build queries from payload
    queries: List[str] = []
    dests: List[str] = payload.get("destinations", []) or []
    if len(dests) >= 2:
        for a, b in zip(dests[:-1], dests[1:]):
            queries.append(f"how to travel {a} to {b} train flight bus cost")
    for c in dests[:3]:
        queries += [f"top attractions {c} family friendly", f"{c} city pass prices"]
    if dests:
        origin = payload.get("origin", "")
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

    # Call LLM (your llm.py returns a dict already)
    data = call_llm(payload, snippets)
    # Ensure it’s a dict for FastAPI
    return data

# ---------- helpers ----------
def _split_nights(total_nights: int, n_stops: int) -> List[int]:
    base = [total_nights // n_stops] * n_stops
    for i in range(total_nights % n_stops):
        base[i] += 1
    return base

def _ppd(budget: float, days: int, party_size: int) -> float:
    return max(1.0, budget / max(1, days) / max(1, party_size))

def _day_plans(cities: List[str], nights_each: List[int]) -> List[DayPlan]:
    out: List[DayPlan] = []
    for city, n in zip(cities, nights_each):
        for _ in range(n):
            out.append(DayPlan(
                city=city,
                must_do=[f"Top sights in {city}"],
                hidden_gem=[f"Neighborhood food lane in {city}"],
                flex_hours=2
            ))
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

# app/orchestrator.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple
from collections.abc import Iterable
from datetime import datetime, timedelta
from urllib.parse import urlencode, quote_plus
import asyncio
import logging

from app.schemas import (
    TripRequest,
    TripResponse,
    PlanBundle,
    TravelLeg,
    Stay,
    DayPlan,
    AgentContext,
    TripPrefs,
    BookingLink,
)
from app.llm import call_llm, llm_backfill_city_details  # import at module top to avoid circulars
from app.agents.foundation_agent import extract_foundation
from app.agents.destination_scout import expand_destinations
from app.agents.logistics_planner import compute_logistics

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
_level = os.getenv("TRIP_PLANNER_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _level, logging.INFO))
logger.propagate = False

# Search providers vary widely in the keys they expose for summarised content.
# Normalise here so downstream logic can treat them uniformly.
def _payload_snippet(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("content", "snippet", "text", "description", "answer", "raw_content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    highlights = payload.get("highlights")
    if isinstance(highlights, list):
        joined = " ".join([item.strip() for item in highlights if isinstance(item, str) and item.strip()])
        if joined:
            return joined
    return ""

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

    # Harmonise any agent context that upstream agents may have assembled.

    if isinstance(context, AgentContext):
        ctx: Dict[str, Any] = context.model_dump(mode="python")
    elif isinstance(context, dict):
        ctx = dict(context)
    elif context is None:
        ctx = {}
    else:
        ctx = dict(context)  # type: ignore[arg-type]

    # Break context into the buckets produced by earlier agents. Missing keys
    # are normal when adapters are offline, so default to empty structures.
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

    # Trip length and per-destination allocations feed most downstream estimates.
    trip_days = int(foundation_ctx.get("dates", {}).get("duration_days") or max(1, _days(req.dates.start, req.dates.end)))
    nights = int(foundation_ctx.get("dates", {}).get("nights") or max(1, trip_days - 1))
    n_stops = max(1, len(req.destinations))

    nights_allocation = list(foundation_ctx.get("nights_allocation", [])) if foundation_ctx else []
    if len(nights_allocation) != n_stops:
        nights_allocation = _split_nights(nights, n_stops)

    party_size = req.party.adults + req.party.children + req.party.seniors
    ppd_val = float(
        foundation_ctx.get("budget", {}).get("per_person_per_day")
        or _ppd(req.budget_total, trip_days, party_size)
    )
    stay_windows = _infer_stay_windows(req, nights_allocation, logistics_ctx, foundation_ctx)

    logger.info(
        "Planning trip for %s travellers from %s visiting %s between %s and %s with budget %.2f %s",
        party_size,
        req.origin,
        ", ".join(req.destinations) or req.origin,
        req.dates.start,
        req.dates.end,
        req.budget_total,
        req.currency,
    )
    logger.debug(
        "Context snapshot — notes:%d foundation:%s logistics_legs:%d",
        len(combined_notes),
        sorted(foundation_ctx.keys()),
        len(logistics_ctx.get("legs") or []),
    )

    # Build a simple stay blueprint for each stop. Bundle builders will mutate
    # these to offer cheaper or more comfortable mixes.
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
    activity_cost = trip_days * party_size * ppd_val * 0.15
    baseline_total = base_cost_stay + base_cost_move + food_misc + activity_cost
    logger.info(
        "Baseline composition — stays %.2f, transport %.2f, food %.2f, activities %.2f (total %.2f %s) against budget %.2f",
        base_cost_stay,
        base_cost_move,
        food_misc,
        activity_cost,
        baseline_total,
        req.currency,
        req.budget_total,
    )

    budget_delta = baseline_total - req.budget_total
    ctx["costs"] = {
        "currency": req.currency,
        "stays": round(base_cost_stay, 2),
        "transport": round(base_cost_move, 2),
        "food": round(food_misc, 2),
        "activities": round(activity_cost, 2),
        "other": 0.0,
        "total": round(baseline_total, 2),
        "budget": req.budget_total,
        "delta": round(budget_delta, 2),
        "status": "over" if budget_delta > 0 else "under" if budget_delta < 0 else "on_budget",
    }
    logger.info(
        "Budget comparison — baseline total %.2f vs budget %.2f (%s %.2f)",
        baseline_total,
        req.budget_total,
        "over by" if budget_delta > 0 else "under by" if budget_delta < 0 else "aligned with",
        abs(budget_delta),
    )

    _extend_unique(combined_notes, _suggest_date_shifts(req, trip_days, baseline_total))

    transfer_buffers = dict(logistics_ctx.get("transfer_buffers", {}))
    base_experiences = [day.model_copy(deep=True) for day in experiences]

    cheapest_stays = [
        Stay(**{**s.model_dump(), "style": "homestay", "budget_per_night": s.budget_per_night * 0.75})
        for s in stays
    ]

    balanced_other_cost = food_misc + activity_cost
    cheapest_other_cost = food_misc * 0.85 + activity_cost * 0.6
    comfort_other_cost = food_misc * 1.15 + activity_cost * 1.2

    # Construct a mix of bundles that share the baseline experiences but vary
    # in the level of spend and pace.
    options: List[PlanBundle] = [
        _build_plan_bundle(
            label="balanced",
            summary="Balanced cost, pace and comfort.",
            stays=[stay.model_copy(deep=True) for stay in stays],
            travel=[leg.model_copy(deep=True) for leg in intercity],
            other_cost=balanced_other_cost,
            req=req,
            trip_days=trip_days,
            combined_notes=combined_notes,
            base_notes=["Good default; tune by preferences (diet, mobility)."],
            local_transport=["metro", "IC card", "occasional rideshare"],
            experience_plan=[day.model_copy(deep=True) for day in base_experiences],
            transfer_buffers=transfer_buffers,
            transfer_override=len(intercity),
            stay_windows=stay_windows,
        ),
        _build_plan_bundle(
            label="cheapest",
            summary="Minimize cost using homestays and public transport.",
            stays=[stay.model_copy(deep=True) for stay in cheapest_stays],
            travel=[leg.model_copy(deep=True) for leg in intercity],
            other_cost=cheapest_other_cost,
            req=req,
            trip_days=trip_days,
            combined_notes=combined_notes,
            base_notes=[
                "Lean on homestays, flexible dates, and public transport to stretch the budget.",
                "Indicative prices; replace with live quotes when adapters are connected.",
            ],
            local_transport=["metro/day-pass", "bus", "rideshare (as needed)"],
            experience_plan=[day.model_copy(deep=True) for day in base_experiences],
            transfer_buffers=transfer_buffers,
            transfer_override=max(1, len(intercity)),
            stay_windows=stay_windows,
        ),
        _build_plan_bundle(
            label="comfort",
            summary="Fewer transfers, faster hops, nicer stays.",
            stays=[
                Stay(**{**s.model_dump(), "style": "boutique", "budget_per_night": s.budget_per_night * 1.5})
                for s in stays
            ],
            travel=[
                (
                    TravelLeg(
                        mode="flight",
                        **{"from": leg.frm, "to": leg.to},
                        duration_hr=1.0,
                        cost_estimate=(leg.cost_estimate or 80.0) * 2.2,
                    )
                    if leg.mode in ("train", "bus")
                    else leg.model_copy(deep=True)
                )
                for leg in intercity
            ],
            other_cost=comfort_other_cost,
            req=req,
            trip_days=trip_days,
            combined_notes=combined_notes,
            base_notes=["Upgrade selected hops to flights; swap stays to boutique."],
            local_transport=["hotel pickup", "metro", "rideshare"],
            experience_plan=[day.model_copy(deep=True) for day in base_experiences],
            transfer_buffers=transfer_buffers,
            transfer_override=len(intercity),
            stay_windows=stay_windows,
        ),
    ]

    metrics = [_plan_metrics(bundle, req.budget_total) for bundle in options]
    logger.debug("Computed metrics for %d bundles", len(metrics))
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
    options = _promote_preferred_option(options, req.prefs.objective)

    agent_context_model = AgentContext.model_validate(ctx) if ctx else None
    logger.info(
        "Generated %d candidate bundles; best composite %.2f",
        len(options),
        options[0].scores.get("composite", 0.0) if options else 0.0,
    )
    return TripResponse(query_echo=req, options=options, agent_context=agent_context_model)

# ---------- LLM + (optional) web sources ----------
async def orchestrate_llm_trip(
    payload: Dict[str, Any],
    allow_domains: List[str] | None = None,
    deny_domains: List[str] | None = None,
) -> Dict[str, Any]:
    """Build queries, optionally web-search + fetch snippets, call LLM, return dict."""
    logger.info(
        "Orchestration start: origin=%s, destinations=%s, dates=%s-%s, budget=%s", 
        payload.get("origin"),
        payload.get("destinations"),
        payload.get("dates", {}).get("start") if isinstance(payload.get("dates"), dict) else None,
        payload.get("dates", {}).get("end") if isinstance(payload.get("dates"), dict) else None,
        payload.get("budget_total") or payload.get("budget"),
    )
    foundation = extract_foundation(payload)
    logger.debug("Foundation context keys: %s", sorted(foundation.keys()))
    trip_req = TripRequest.model_validate(payload)

    # Normalise any free-form user preferences before they travel through agents.
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

    # Build queries from payload. These will be surfaced in logs so operators can
    # understand what the orchestrator attempted to fetch from the web.
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

    logger.info("Assembled %d search queries", len(queries))

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
                logger.info("Search query '%s' produced %d hits", q, len(hits))
                results.extend(hits)
            except Exception:
                logger.warning("Search failed for query '%s'", q, exc_info=True)
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
        fetch_targets = pruned[:policy.max_results]
        docs = await asyncio.gather(*[searcher.fetch(r["url"]) for r in fetch_targets], return_exceptions=True)
        failed_urls: List[str] = []
        fallback_count = 0
        for payload, doc in zip(fetch_targets, docs):
            if isinstance(doc, Exception):
                failed_urls.append(payload.get("url", ""))
                logger.warning("Fetcher raised for %s", payload.get("url"), exc_info=True)
                doc = None
            if getattr(doc, "url", None):
                snippets.append(
                    {
                        "url": doc.url,
                        "title": getattr(doc, "title", "") or "",
                        "text": getattr(doc, "text", "") or "",
                        "source": "web_fetch",
                    }
                )
                continue
            snippet_text = _payload_snippet(payload)
            if snippet_text:
                snippets.append(
                    {
                        "url": payload.get("url", ""),
                        "title": payload.get("title", "") or "",
                        "text": snippet_text,
                        "source": "search_summary",
                    }
                )
                fallback_count += 1
            else:
                failed_urls.append(payload.get("url", ""))
        if fallback_count:
            logger.info("Populated %d snippet(s) from search result summaries", fallback_count)
        if failed_urls:
            dedup_failures = sorted({url for url in failed_urls if url})
            logger.warning("Unable to assemble snippets for %d url(s): %s", len(dedup_failures), ", ".join(dedup_failures))
        logger.info(
            "Collected %d unique snippets from %d raw hits (allowed domains: %s)",
            len(snippets),
            len(results),
            ", ".join(resolved_allow) if resolved_allow else "*",
        )
    else:
        logger.info("WebSearcher unavailable; continuing with heuristic planning only")

    # Derive richer context from specialised agents.
    destination_context = expand_destinations(foundation, snippets)

    llm_sources: List[Dict[str, Any]] = []
    heuristic_cities: List[str] = destination_context.get("heuristic_cities", []) or []
    remaining_heuristics = set(heuristic_cities)
    if heuristic_cities:
        missing_details: List[str] = []
        for city in heuristic_cities:
            dest_entry = next(
                (d for d in destination_context.get("destinations", []) if d.get("city") == city),
                {},
            )
            missing_fields: List[str] = []
            if not dest_entry.get("highlights"):
                missing_fields.append("highlights")
            if not dest_entry.get("experiences"):
                missing_fields.append("experiences")
            if not dest_entry.get("dining"):
                missing_fields.append("dining")
            if missing_fields:
                missing_details.append(f"{city}: {', '.join(missing_fields)}")
        if missing_details:
            logger.warning("Missing destination intel from web snippets — %s", "; ".join(missing_details))

        backfill_model = (
            payload.get("llm_backfill_model")
            or payload.get("llm_model")
            or os.getenv("TRIP_PLANNER_FALLBACK_MODEL")
            or "gpt-4o-mini"
        )
        backfill_data = llm_backfill_city_details(heuristic_cities, foundation, model=backfill_model)
        if backfill_data:
            dest_notes = destination_context.setdefault("notes", []) or []
            for city, data in backfill_data.items():
                dest_entry = next(
                    (d for d in destination_context.get("destinations", []) if d.get("city") == city),
                    None,
                )
                if not dest_entry:
                    continue
                filled_fields: List[str] = []
                for key in ("highlights", "experiences", "dining"):
                    values = data.get(key)
                    if values:
                        dest_entry[key] = values
                        filled_fields.append(key)
                if filled_fields:
                    dest_entry["source"] = "llm"
                    remaining_heuristics.discard(city)
                    logger.info(
                        "LLM backfill (%s) provided %s for %s",
                        backfill_model,
                        ", ".join(filled_fields),
                        city,
                    )
                if data.get("notes"):
                    _extend_unique(dest_notes, data["notes"])
                    if not filled_fields:
                        filled_fields.append("notes")
                if filled_fields:
                    llm_sources.append({"city": city, "model": backfill_model, "fields": filled_fields})
            if llm_sources:
                updated_cities = ", ".join(sorted({entry["city"] for entry in llm_sources}))
                info_note = f"LLM fallback ({backfill_model}) supplied destination intel for: {updated_cities}."
                _extend_unique(destination_context.setdefault("notes", []), [info_note])
                dest_source_bucket = destination_context.setdefault("sources", [])
                for entry in llm_sources:
                    dest_source_bucket.append(
                        {
                            "city": entry["city"],
                            "title": f"{entry['city']} fallback via {backfill_model}",
                            "url": f"llm://{backfill_model}",
                            "fields": ", ".join(entry["fields"]),
                        }
                    )
                logger.info("LLM backfill (%s) provided destination data for %s", backfill_model, updated_cities)
        else:
            logger.warning(
                "LLM backfill unavailable; retaining heuristic destination highlights for %s",
                ", ".join(heuristic_cities),
            )
            _extend_unique(
                destination_context.setdefault("notes", []),
                [
                    "Heuristic destination highlights retained due to unavailable LLM backfill.",
                ],
            )

    destination_context["heuristic_cities"] = sorted(remaining_heuristics)
    if remaining_heuristics:
        heuristic_sources = destination_context.setdefault("sources", [])
        for city in sorted(remaining_heuristics):
            heuristic_sources.append(
                {
                    "city": city,
                    "title": f"{city} heuristic scaffold",
                    "url": "heuristic://template",
                    "fields": "highlights, experiences",
                }
            )

    logistics_context = compute_logistics(foundation)

    aggregated_notes = list({note: None for note in (
        *(foundation.get("notes", []) or []),
        *(destination_context.get("notes", []) or []),
        *(logistics_context.get("feasibility_notes", []) or []),
    )}.keys())

    # Promote the snippet URLs so downstream consumers can surface citations.
    source_links = sorted({snip["url"] for snip in snippets if snip.get("url")})
    extra_source_urls = [src.get("url") for src in destination_context.get("sources", []) if src.get("url")]
    if extra_source_urls:
        source_links = sorted({*source_links, *extra_source_urls})
    if llm_sources:
        source_links = sorted({*source_links, *(f"llm://{entry['model']}" for entry in llm_sources)})

    agent_context: Dict[str, Any] = {
        "foundation": foundation,
        "destinations": destination_context.get("destinations", []),
        "logistics": logistics_context,
        "notes": aggregated_notes,
        "sources": destination_context.get("sources", []),
        "snippets": snippets,
        "source_links": source_links,
        "llm_sources": llm_sources,
    }

    # Produce the deterministic baseline plan. The LLM later layers narrative.
    baseline_plan = plan_trip(trip_req, agent_context)
    if baseline_plan.agent_context and getattr(baseline_plan.agent_context, "costs", None):
        agent_context["costs"] = baseline_plan.agent_context.costs
    baseline_dump = baseline_plan.model_dump(mode="json")
    baseline_dump["source_links"] = source_links
    logger.info(
        "Baseline planner produced %d bundles; top label: %s",
        len(baseline_plan.options),
        baseline_plan.options[0].label if baseline_plan.options else "n/a",
    )

    # Call LLM (your llm.py returns a dict already)
    try:
        enriched_payload = dict(payload)
        enriched_payload["agent_context"] = agent_context
        logger.info("Calling LLM with %d snippets", len(snippets))
        data = call_llm(enriched_payload, snippets)
        if isinstance(data, dict):
            logger.info(
                "LLM response received with keys: %s",
                ", ".join(sorted(data.keys())),
            )
    except Exception as exc:
        logger.exception("LLM call failed: %s", exc)
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
        data.setdefault("source_links", source_links)
        return data

    return {
        "llm_raw": data,
        "baseline_plan": baseline_dump,
        "snippets": snippets,
        "agent_context": agent_context,
        "source_links": source_links,
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


def _extend_unique(target: List[str], notes: Iterable[str]) -> None:
    for note in notes:
        if note and note not in target:
            target.append(note)


def _merge_notes(primary: Iterable[str], secondary: Iterable[str]) -> List[str]:
    merged: List[str] = []
    for bucket in (primary, secondary):
        for note in bucket:
            if note and note not in merged:
                merged.append(note)
    return merged


def _build_plan_bundle(
    *,
    label: str,
    summary: str,
    stays: List[Stay],
    travel: List[TravelLeg],
    other_cost: float,
    req: TripRequest,
    trip_days: int,
    combined_notes: List[str],
    base_notes: List[str] | None,
    local_transport: List[str],
    experience_plan: List[DayPlan],
    transfer_buffers: Dict[str, float],
    transfer_override: int | None = None,
    stay_windows: List[Dict[str, str]] | None = None,
) -> PlanBundle:
    working_notes: List[str] = list(base_notes or [])
    stay_total = sum(stay.nights * stay.budget_per_night for stay in stays)
    travel_total = sum((leg.cost_estimate or 0.0) for leg in travel)
    base_total = stay_total + travel_total + other_cost
    logger.info(
        "Preparing %s bundle with base cost %.2f %s (budget %.2f)",
        label,
        base_total,
        req.currency,
        req.budget_total,
    )

    budget_room = req.budget_total - base_total
    travel_adjustments, travel_constraint = _optimize_travel_time(
        travel, label, budget_room, req.currency
    )
    if travel_adjustments:
        _extend_unique(working_notes, travel_adjustments)
        travel_total = sum((leg.cost_estimate or 0.0) for leg in travel)
        base_total = stay_total + travel_total + other_cost
        budget_room = req.budget_total - base_total
    if travel_constraint:
        _extend_unique(working_notes, [travel_constraint])

    adjusted_stays, adjusted_travel, total_cost, alignment_note, limitation_note = _align_bundle_components(
        stays,
        travel,
        other_cost,
        req,
        label,
    )

    if alignment_note:
        _extend_unique(working_notes, [alignment_note])

    notes = _merge_notes(working_notes, combined_notes)
    feasibility = _merge_notes(
        combined_notes,
        ([limitation_note] if limitation_note else []),
    )

    transfers = transfer_override if transfer_override is not None else len(adjusted_travel)
    if transfers == 0 and adjusted_travel:
        transfers = len(adjusted_travel)

    booking_links = _compose_booking_links(
        adjusted_stays,
        adjusted_travel,
        stay_windows or [],
        experience_plan,
        req,
    )

    bundle = PlanBundle(
        label=label,
        summary=summary,
        total_cost=total_cost,
        currency=req.currency,
        transfers=transfers,
        est_duration_days=trip_days,
        travel=adjusted_travel,
        stays=adjusted_stays,
        local_transport=list(local_transport),
        experience_plan=experience_plan,
        notes=notes,
        feasibility_notes=feasibility,
        transfer_buffers=transfer_buffers,
        booking_links=booking_links,
    )
    final_gap = (
        (bundle.total_cost - req.budget_total) / req.budget_total * 100.0
        if req.budget_total
        else 0.0
    )
    logger.info(
        "Finalised %s bundle at %.2f %s (%+.2f%% vs budget)",
        label,
        bundle.total_cost,
        req.currency,
        final_gap,
    )
    return bundle


def _compose_booking_links(
    stays: List[Stay],
    travel: List[TravelLeg],
    stay_windows: List[Dict[str, str]],
    experience_plan: List[DayPlan],
    req: TripRequest,
) -> List[BookingLink]:
    links: List[BookingLink] = []
    seen: set[tuple[str, str, str]] = set()

    def _append(category: str, label: str, url: str, details: str | None = None) -> None:
        if not url:
            return
        key = (category, label, url)
        if key in seen:
            return
        seen.add(key)
        links.append(BookingLink(category=category, label=label, url=url, details=details))

    window_by_city = {}
    for idx, stay in enumerate(stays):
        window = stay_windows[idx] if idx < len(stay_windows) else {}
        window_by_city.setdefault(stay.city, window)
        check_in = window.get("check_in")
        check_out = window.get("check_out")
        url = _build_hotel_url(stay.city, check_in, check_out, req)
        details = _format_date_range(check_in, check_out, suffix=f" • {stay.nights} night(s)")
        _append("stay", f"Stay in {stay.city}", url, details)

    for leg in travel:
        depart_date = _coerce_date(leg.date) or window_by_city.get(leg.frm, {}).get("check_out")
        url = _build_transport_url(leg, depart_date)
        category = leg.mode if leg.mode in {"flight", "train", "bus"} else "transport"
        details = f"Depart {depart_date}" if depart_date else None
        _append(category, f"{leg.mode.title()} {leg.frm} → {leg.to}", url, details)

    for stay in stays:
        window = window_by_city.get(stay.city, {})
        arrival = window.get("check_in")
        departure = window.get("check_out")
        url = _build_city_pass_url(stay.city, arrival, departure)
        details = _format_date_range(arrival, departure)
        _append("local_pass", f"{stay.city} city & transit passes", url, details)

    experience_queries: Dict[str, List[str]] = {}
    for day in experience_plan:
        if not day.city:
            continue
        queries = experience_queries.setdefault(day.city, [])
        for item in day.must_do[:1]:
            cleaned = item.strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
    for city, items in experience_queries.items():
        window = window_by_city.get(city, {})
        arrival = window.get("check_in")
        departure = window.get("check_out")
        for item in items[:2]:
            url = _build_sight_url(city, item, arrival, departure)
            details = _format_date_range(arrival, departure)
            _append("sightseeing", item, url, details)

    return links


def _build_hotel_url(city: str, check_in: str | None, check_out: str | None, req: TripRequest) -> str:
    base = "https://www.booking.com/searchresults.html"
    adults = max(req.party.adults + req.party.seniors, 1)
    children = max(req.party.children, 0)
    params = {
        "ss": city,
        "group_adults": adults,
        "group_children": children,
        "no_rooms": 1,
    }
    if check_in:
        params["checkin"] = check_in
    if check_out:
        params["checkout"] = check_out
    return f"{base}?{urlencode(params)}"


def _build_transport_url(leg: TravelLeg, depart_date: str | None) -> str:
    if leg.mode == "flight":
        query = f"flights from {leg.frm} to {leg.to}"
        if depart_date:
            query += f" on {depart_date}"
        return f"https://www.google.com/travel/flights?q={quote_plus(query)}"

    base = "https://www.omio.com/search"
    params = {
        "departure": leg.frm,
        "arrival": leg.to,
    }
    if depart_date:
        params["date"] = depart_date
    if leg.mode in {"train", "bus"}:
        params["mode"] = leg.mode
    return f"{base}?{urlencode(params)}"


def _build_city_pass_url(city: str, arrival: str | None, departure: str | None) -> str:
    params = {"q": f"{city} city pass"}
    if arrival:
        params["date_from"] = arrival
    if departure:
        params["date_to"] = departure
    return f"https://www.getyourguide.com/s/?{urlencode(params)}"


def _build_sight_url(city: str, experience: str, arrival: str | None, departure: str | None) -> str:
    query = f"{experience} {city} tickets"
    params = {"q": query}
    if arrival:
        params["date_from"] = arrival
    if departure:
        params["date_to"] = departure
    return f"https://www.getyourguide.com/s/?{urlencode(params)}"


def _format_date_range(start: str | None, end: str | None, *, suffix: str | None = None) -> str | None:
    if not start and not end:
        return suffix.lstrip() if suffix else None
    parts = []
    if start:
        parts.append(start)
    if end:
        parts.append(end)
    details = " → ".join(parts)
    if suffix:
        details = f"{details}{suffix}"
    return details


def _infer_stay_windows(
    req: TripRequest,
    nights_allocation: List[int],
    logistics_ctx: Dict[str, Any],
    foundation_ctx: Dict[str, Any],
) -> List[Dict[str, str]]:
    windows: List[Dict[str, str]] = []
    timeline = []
    if isinstance(logistics_ctx, dict):
        timeline = logistics_ctx.get("timeline") or []
    for entry in timeline:
        city = entry.get("city")
        if not city:
            continue
        check_in = _coerce_date(entry.get("arrival"))
        check_out = _coerce_date(entry.get("departure"))
        windows.append({"city": city, "check_in": check_in, "check_out": check_out})

    if len(windows) == len(req.destinations):
        return windows

    fallback_start = foundation_ctx.get("dates", {}).get("start") or req.dates.start
    start_dt = _parse_date(fallback_start)
    cursor = start_dt
    fallback: List[Dict[str, str]] = []
    for city, nights in zip(req.destinations, nights_allocation):
        nights = max(int(nights or 1), 1)
        check_in = cursor.date().isoformat() if cursor else None
        cursor = cursor + timedelta(days=nights) if cursor else None
        check_out = cursor.date().isoformat() if cursor else None
        fallback.append({"city": city, "check_in": check_in, "check_out": check_out})
    if len(fallback) < len(req.destinations):
        remaining = req.destinations[len(fallback) :]
        for city in remaining:
            fallback.append({"city": city, "check_in": None, "check_out": None})
    return fallback


def _coerce_date(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value)
    if "T" in text:
        text = text.split("T", 1)[0]
    parsed = _parse_date(text)
    return parsed.date().isoformat() if parsed else text if text else None


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
def _align_bundle_components(
    stays: List[Stay],
    travel: List[TravelLeg],
    other_cost: float,
    req: TripRequest,
    label: str,
) -> Tuple[List[Stay], List[TravelLeg], float, str | None, str | None]:
    budget_total = req.budget_total
    currency = req.currency
    if budget_total <= 0:
        note = "Budget missing; left heuristic pricing in place."
        logger.info("Budget not provided; skipping alignment for %s bundle", label)
        total = sum(stay.nights * stay.budget_per_night for stay in stays) + sum(
            (leg.cost_estimate or 0.0) for leg in travel
        ) + other_cost
        return stays, travel, total, note, None

    raw_total = sum(stay.nights * stay.budget_per_night for stay in stays) + sum(
        (leg.cost_estimate or 0.0) for leg in travel
    ) + other_cost
    if raw_total <= 0:
        return stays, travel, 0.0, None, None

    raw_scale = budget_total / raw_total
    tolerance = 0.08 if label == "cheapest" else 0.05
    if abs(1.0 - raw_scale) <= tolerance:
        note = (
            f"Costs sit within {abs(1.0 - raw_scale) * 100:.1f}% of the {budget_total:,.0f} {currency} budget."
        )
        logger.info("%s bundle already within tolerance of budget", label)
        return stays, travel, raw_total, note, None

    if label == "cheapest":
        target_scale = min(max(raw_scale, 0.65), 1.05)
    elif label == "comfort":
        target_scale = min(max(raw_scale, 0.85), 1.2)
    else:
        target_scale = min(max(raw_scale, 0.8), 1.1)

    adjusted_stays, adjusted_travel, adjusted_other_cost = _apply_budget_scale(
        stays, travel, other_cost, target_scale, label
    )
    adjusted_total = sum(
        stay.nights * stay.budget_per_night for stay in adjusted_stays
    ) + sum((leg.cost_estimate or 0.0) for leg in adjusted_travel) + adjusted_other_cost

    gap_after = budget_total - adjusted_total
    gap_ratio_after = gap_after / budget_total
    alignment_note = (
        f"Aligned spend to {adjusted_total:,.0f} {currency} ({gap_ratio_after:+.1%} vs budget)."
    )

    limitation_note: str | None = None
    if abs(gap_ratio_after) > 0.12:
        if gap_after < 0:
            limitation_note = (
                "Even after cost trims the plan remains above budget; consider removing a stop or shifting travel to shoulder-season dates."
            )
        else:
            limitation_note = (
                "Plan sits well under budget; add experiences or extend nights to make the most of the allocation."
            )
    elif (target_scale in (0.65, 1.05, 0.85, 1.2, 0.8, 1.1)) and gap_after < 0:
        limitation_note = (
            "Supplier pricing caps prevented full alignment; nudging departure by a few days may unlock better rates."
        )

    logger.info(
        "Budget alignment for %s bundle — raw_scale %.2f → applied %.2f, final gap %.2f%%",
        label,
        raw_scale,
        target_scale,
        gap_ratio_after * 100.0,
    )
    return adjusted_stays, adjusted_travel, adjusted_total, alignment_note, limitation_note


def _apply_budget_scale(
    stays: List[Stay],
    travel: List[TravelLeg],
    other_cost: float,
    scale: float,
    label: str,
) -> Tuple[List[Stay], List[TravelLeg], float]:
    if scale <= 0:
        scale = 1.0

    if scale >= 1.0:
        stay_factor = scale * (1.05 if label == "comfort" else 1.02 if label == "balanced" else 1.0)
        travel_factor = scale * (1.08 if label == "comfort" else 1.02)
        other_factor = 1.0 + (scale - 1.0) * 0.6
    else:
        stay_factor = max(0.55, scale * (0.95 if label == "cheapest" else 0.9))
        travel_factor = max(0.5, scale * (0.9 if label == "cheapest" else 0.85))
        other_factor = 0.85 + 0.15 * scale

    adjusted_stays = [
        stay.model_copy(update={"budget_per_night": max(18.0, stay.budget_per_night * stay_factor)})
        for stay in stays
    ]
    adjusted_travel: List[TravelLeg] = []
    for leg in travel:
        cost = leg.cost_estimate
        if cost is not None:
            cost = max(0.0, cost * travel_factor)
        adjusted_travel.append(leg.model_copy(update={"cost_estimate": cost}))

    adjusted_other_cost = max(0.0, other_cost * other_factor)
    return adjusted_stays, adjusted_travel, adjusted_other_cost


def _suggest_date_shifts(req: TripRequest, trip_days: int, baseline_total: float) -> List[str]:
    suggestions: List[str] = []
    try:
        start_dt = datetime.strptime(req.dates.start, "%Y-%m-%d")
        end_dt = datetime.strptime(req.dates.end, "%Y-%m-%d")
    except Exception:
        logger.debug("Unable to parse trip dates for suggestions", exc_info=True)
        return suggestions

    total_span = (end_dt - start_dt).days + 1

    if start_dt.weekday() >= 4:
        suggestions.append(
            "Shifting departure to a Tue/Wed start often lowers fares and crowds compared to a weekend getaway."
        )
    if baseline_total > req.budget_total * 1.1 and req.budget_total > 0:
        suggestions.append(
            "Sliding the trip into the shoulder season (±7-10 days) can reduce stay rates by 15%+."
        )
    if trip_days < len(req.destinations) * 2:
        suggestions.append(
            "Add a flex day between cities so experiences feel less rushed and transfers stay manageable."
        )
    if start_dt.month in {6, 7, 8}:
        suggestions.append(
            "Summer peak pricing is high; a late-spring or early-fall window could unlock better value experiences."
        )
    if total_span >= 10:
        suggestions.append(
            "Include a lighter day mid-trip to balance energy levels on itineraries longer than a week."
        )

    logger.info("Date suggestions generated: %s", suggestions)
    return suggestions


def _optimize_travel_time(
    travel: List[TravelLeg],
    label: str,
    budget_room: float,
    currency: str,
) -> Tuple[List[str], str | None]:
    if not travel:
        return [], None

    durations = [
        leg.duration_hr if leg.duration_hr is not None else _DEFAULT_MODE_DURATION.get(leg.mode, 3.0)
        for leg in travel
    ]
    idx = max(range(len(travel)), key=lambda i: durations[i])
    longest = travel[idx]
    baseline_duration = durations[idx]

    notes: List[str] = []
    if baseline_duration < 4.5:
        return notes, None

    constraint_note: str | None = None
    if label in {"balanced", "comfort"} and budget_room >= -150.0:
        previous_cost = longest.cost_estimate or 0.0
        upgraded_cost = previous_cost * 1.35 if previous_cost else 160.0
        upgraded_leg = longest.model_copy(
            update={
                "mode": "flight",
                "duration_hr": max(1.0, baseline_duration * 0.45),
                "cost_estimate": upgraded_cost,
            }
        )
        travel[idx] = upgraded_leg
        saved_hours = baseline_duration - upgraded_leg.duration_hr
        delta_cost = upgraded_cost - previous_cost
        notes.append(
            f"Upgraded {longest.frm}→{longest.to} to a short flight, saving ~{saved_hours:.1f}h for about {delta_cost:,.0f} {currency}."
        )
        logger.info(
            "Applied travel-time upgrade for %s bundle leg %s→%s (saved %.1fh, delta %.2f)",
            label,
            longest.frm,
            longest.to,
            saved_hours,
            delta_cost,
        )
    elif label == "cheapest" and budget_room > 0 and baseline_duration > 6.0:
        previous_cost = longest.cost_estimate or 60.0
        premium_cost = previous_cost * 1.15
        faster_duration = max(2.5, baseline_duration * 0.7)
        travel[idx] = longest.model_copy(
            update={
                "duration_hr": faster_duration,
                "cost_estimate": premium_cost,
            }
        )
        notes.append(
            f"Reserved a faster {longest.mode} on {longest.frm}→{longest.to}, trimming travel by ~{baseline_duration - faster_duration:.1f}h with a modest upgrade."
        )
        logger.info(
            "Applied faster ground option on cheapest bundle leg %s→%s (saved %.1fh, delta %.2f)",
            longest.frm,
            longest.to,
            baseline_duration - faster_duration,
            premium_cost - previous_cost,
        )
    else:
        constraint_note = (
            f"Longest hop {longest.frm}→{longest.to} remains {baseline_duration:.1f}h; move travel by a day or opt for an overnight service to ease transit."
        )
        logger.info(
            "Unable to shorten longest leg %s→%s for %s bundle due to budget limits",
            longest.frm,
            longest.to,
            label,
        )

    return notes, constraint_note


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
        if metrics.get("flight_hours", 0.0) > 0:
            time_score = max(time_score, 0.4)
            penalty = min(0.3, 0.1 * metrics.get("flight_hours", 0.0))
            time_score = _clamp(time_score * (1.0 - penalty))
        else:
            time_score = max(time_score, 0.65)

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


def _promote_preferred_option(options: List[PlanBundle], preferred_label: str | None) -> List[PlanBundle]:
    if not options or not preferred_label:
        return options
    try:
        target_idx = next((idx for idx, opt in enumerate(options) if opt.label == preferred_label), None)
    except Exception:
        return options
    if target_idx is None or target_idx == 0:
        return options
    best_score = options[0].scores.get("composite", 0.0)
    preferred_score = options[target_idx].scores.get("composite", 0.0)
    threshold = 0.04
    if preferred_score >= best_score or best_score - preferred_score <= threshold:
        preferred = options[target_idx]
        reordered = [preferred] + options[:target_idx] + options[target_idx + 1 :]
        return reordered
    return options


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

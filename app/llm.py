# app/llm.py
import os
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback when openai missing in tests
    OpenAI = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
_level = os.getenv("TRIP_PLANNER_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _level, logging.INFO))
logger.propagate = False

# Load .env file if present
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if api_key and OpenAI is not None:
    _client = OpenAI(api_key=api_key)
else:  # pragma: no cover - exercised indirectly in tests without API key
    _client = None
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; returning stubbed responses from call_llm")
    else:
        logger.warning("openai package unavailable; returning stubbed responses from call_llm")

SYSTEM_PROMPT = """You are a travel-planning agent.
Use ONLY the provided web snippets.
Return JSON with two keys:
  1) trip_plan
  2) sources (array of {title,url,why_used}).
If unsure, mark facts as 'indicative' and do not invent citations.
Return ONLY valid JSON.
"""

CITY_BACKFILL_SYSTEM = """You are a cautious travel research assistant.
When web scraping fails, provide conservative fallback guidance.
Respond ONLY in JSON with the schema:
  {"cities": {"CITY": {"highlights": [], "experiences": [], "dining": [], "notes": []}}}
- highlights: 2-4 iconic, family-friendly attractions per city.
- experiences: activities or pacing tips aligned with the party profile.
- dining: vegetarian-friendly or flexible dining ideas (0-3 entries).
- notes: caveats when reliable data is unavailable.
Do not fabricate pricing or availability.
Leave arrays empty when uncertain.
"""

CITY_BACKFILL_TEMPLATE = """We are missing verified web intel for these cities: {cities}.
Trip context summary:
- origin: {origin}
- dates: {dates}
- party: {party}
- interests: {interests}
- dietary: {diet}

Provide concise bullet-style strings (<=120 characters each).
If information is generic knowledge, mark it as indicative (e.g., "Indicative: ...").
"""

USER_TEMPLATE = """User profile:
origin: {origin}
purpose: {purpose}
budget_total: {budget_total} {currency}
dates: {dates}
party: {party}
constraints: {constraints}
interests: {interests}
destinations: {destinations}

Web snippets:
{snippets}
"""

def _build_snippets(snips: List[Dict[str, str]]) -> str:
    """Format snippets for inclusion in the prompt."""
    parts = []
    for i, d in enumerate(snips, 1):
        parts.append(
            f"[{i}] TITLE: {d.get('title','')}\nURL: {d.get('url','')}\nTEXT: {(d.get('text','') or '')[:800]}"
        )
    return "\n\n".join(parts)

def call_llm(payload: Dict[str, Any], snippets: List[Dict[str, str]], model: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Call OpenAI LLM and return parsed JSON dict."""
    user_prompt = USER_TEMPLATE.format(
        origin=payload.get("origin"),
        purpose=payload.get("purpose"),
        budget_total=payload.get("budget_total"),
        currency=payload.get("currency"),
        dates=payload.get("dates"),
        party=payload.get("party"),
        constraints=payload.get("constraints"),
        interests=payload.get("interests"),
        destinations=payload.get("destinations"),
        snippets=_build_snippets(snippets),
    )

    if _client is None:
        logger.info(
            "Skipping LLM call — returning budget-friendly stub payload (missing client or API key)."
        )
        # The rest of the orchestrator can continue operating with deterministic
        # data even when the hosted model is not reachable.
        return {
            "llm": "skipped",
            "reason": "missing_openai_client",
            "echo": {
                "origin": payload.get("origin"),
                "destinations": payload.get("destinations"),
            },
        }

    logger.info(
        "Invoking LLM model %s with %d snippets", model, len(snippets)
    )
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)  # convert string to dict
        logger.info("LLM JSON payload parsed successfully with keys: %s", ", ".join(sorted(parsed.keys())))
        return parsed
    except Exception:
        logger.warning("LLM response was not valid JSON; returning raw text")
        return {"error": "Invalid JSON from model", "raw": raw}


def _summarise_party(party: Dict[str, Any] | None) -> str:
    if not isinstance(party, dict):
        return "unspecified"
    adults = party.get("adults")
    children = party.get("children")
    seniors = party.get("seniors")
    bits: List[str] = []
    if adults:
        bits.append(f"{adults} adults")
    if children:
        bits.append(f"{children} children")
    if seniors:
        bits.append(f"{seniors} seniors")
    return ", ".join(bits) if bits else "unspecified"


def _summarise_dates(dates: Dict[str, Any] | None) -> str:
    if not isinstance(dates, dict):
        return "unspecified"
    start = dates.get("start") or dates.get("begin")
    end = dates.get("end") or dates.get("finish")
    if start and end:
        return f"{start} to {end}"
    return start or end or "unspecified"


def llm_backfill_city_details(
    cities: List[str],
    foundation: Dict[str, Any],
    *,
    model: str = "gpt-4o-mini",
) -> Dict[str, Dict[str, List[str]]]:
    """Use the hosted LLM to backfill destination intel when web data is missing."""

    clean_cities = [c for c in (cities or []) if c]
    if not clean_cities:
        return {}

    if _client is None:  # pragma: no cover - exercised via stub in tests
        logger.info(
            "Skipping LLM backfill for cities %s (missing client or API key)",
            ", ".join(clean_cities),
        )
        return {}

    origin = foundation.get("origin")
    dates = foundation.get("dates")
    party = foundation.get("party")
    interests = foundation.get("interests") or []
    constraints = foundation.get("constraints") or {}
    diet = constraints.get("diet") or foundation.get("diet") or []

    user_prompt = CITY_BACKFILL_TEMPLATE.format(
        cities=", ".join(clean_cities),
        origin=origin or "unspecified",
        dates=_summarise_dates(dates),
        party=_summarise_party(party),
        interests=", ".join(interests) if interests else "none stated",
        diet=", ".join(diet) if diet else "none stated",
    )

    logger.info(
        "Invoking LLM model %s for destination backfill (%d cities)",
        model,
        len(clean_cities),
    )

    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CITY_BACKFILL_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = resp.choices[0].message.content
    try:
        payload = json.loads(raw)
    except Exception:
        logger.warning("LLM city backfill returned non-JSON payload; ignoring", exc_info=True)
        return {}

    city_block = payload.get("cities")
    if not isinstance(city_block, dict):
        return {}

    normalised: Dict[str, Dict[str, List[str]]] = {}
    for city in clean_cities:
        data = city_block.get(city) or city_block.get(city.lower()) or {}
        if not isinstance(data, dict):
            data = {}
        highlights = [item for item in data.get("highlights", []) if isinstance(item, str)]
        experiences = [item for item in data.get("experiences", []) if isinstance(item, str)]
        dining = [item for item in data.get("dining", []) if isinstance(item, str)]
        notes = [item for item in data.get("notes", []) if isinstance(item, str)]
        normalised[city] = {
            "highlights": highlights[:4],
            "experiences": experiences[:5],
            "dining": dining[:3],
            "notes": notes[:3],
        }

    return normalised

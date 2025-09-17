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

# app/llm.py
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import json

# Load .env file if present
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in your .env file or export it.")

# Initialize OpenAI client
_client = OpenAI(api_key=api_key)

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
        return json.loads(raw)  # convert string to dict
    except Exception:
        return {"error": "Invalid JSON from model", "raw": raw}

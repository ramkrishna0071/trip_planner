"""Destination scouting utilities."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
_level = os.getenv("TRIP_PLANNER_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _level, logging.INFO))
logger.propagate = False

_ACTION_WORDS: Tuple[str, ...] = (
    "visit",
    "explore",
    "discover",
    "experience",
    "enjoy",
    "tour",
    "stroll",
    "wander",
    "check out",
    "head to",
    "stop by",
    "make time for",
    "plan a trip",
    "don't miss",
    "do not miss",
    "must-see",
    "must see",
)

_POI_KEYWORDS: Tuple[str, ...] = (
    "museum",
    "gallery",
    "monument",
    "palace",
    "castle",
    "park",
    "temple",
    "cathedral",
    "square",
    "market",
    "beach",
    "bridge",
    "church",
    "plaza",
    "landmark",
    "observatory",
    "garden",
    "disneyland",
    "resort",
    "attraction",
)


def expand_destinations(foundation: Dict[str, Any], snippets: Iterable[Dict[str, str]] | None = None) -> Dict[str, Any]:
    """Return per-city highlights using web snippets when available."""
    snippets = list(snippets or [])
    destinations: List[str] = [d for d in foundation.get("destinations", []) if d]

    city_highlights: Dict[str, Dict[str, List[str]]] = {
        city: {"highlights": [], "experiences": [], "dining": []}
        for city in destinations
    }
    snippet_sources: List[Dict[str, str]] = []

    for snip in snippets:
        text = (snip.get("text") or "").strip()
        if not text:
            continue
        title = snip.get("title", "")
        url = snip.get("url", "")
        lowered = text.lower()
        for city in destinations:
            if city.lower() not in lowered and city.lower() not in title.lower():
                continue
            highlights, dining = _extract_points(text, city)
            if highlights:
                city_highlights[city]["highlights"].extend(highlights)
            if dining:
                city_highlights[city]["dining"].extend(dining)
            snippet_sources.append({"city": city, "title": title, "url": url})

    fallback_notes: List[str] = []
    heuristic_cities: List[str] = []
    if not snippets:
        logger.warning("No web snippets available; using heuristic destination highlights")
        fallback_notes.append("Web search unavailable; used heuristics for experiences.")

    season = foundation.get("dates", {}).get("season")
    if season:
        fallback_notes.append(f"Trip begins in {season}; align outdoor activities accordingly.")

    interests = foundation.get("interests", [])

    expanded: List[Dict[str, Any]] = []
    for city in destinations:
        info = city_highlights.get(city, {"highlights": [], "experiences": [], "dining": []})
        highlights = _unique(info.get("highlights", []))
        dining = _unique(info.get("dining", []))

        if not highlights:
            highlights = _fallback_highlights(city, interests)
            heuristic_cities.append(city)
            logger.warning("Missing snippet coverage for %s; applying heuristic highlights", city)
        experiences = _build_experiences(city, highlights, interests)
        if dining:
            experiences.append(f"Sample local flavours: {', '.join(dining[:2])}.")

        expanded.append({
            "city": city,
            "highlights": highlights[:5],
            "experiences": experiences[:5],
            "dining": dining[:3],
            "source": "web" if city not in heuristic_cities else "heuristic",
        })

    notes = fallback_notes
    return {
        "destinations": expanded,
        "sources": snippet_sources,
        "notes": notes,
        "heuristic_cities": heuristic_cities,
    }


def _extract_points(text: str, city: str) -> Tuple[List[str], List[str]]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    highlights: List[str] = []
    dining: List[str] = []
    city_lower = city.lower()
    for sentence in sentences:
        clean = sentence.strip().replace("\n", " ")
        if len(clean) < 40:
            continue

        lower = clean.lower()
        contains_city = city_lower in lower
        contains_dining = any(
            keyword in lower for keyword in ("restaurant", "cafe", "bar", "food", "dining")
        )
        contains_action = any(action in lower for action in _ACTION_WORDS)
        contains_poi = any(keyword in lower for keyword in _POI_KEYWORDS)
        has_capitalised_landmark = bool(_capitalised_tokens(clean, city_lower))

        if not contains_city:
            # Allow sentences that call out named landmarks when paired with
            # action-oriented travel language (e.g. "Don't miss Disneyland").
            if not (
                (contains_action and (has_capitalised_landmark or contains_poi))
                or (contains_poi and has_capitalised_landmark)
            ):
                continue

        if contains_dining:
            dining.append(clean)
        else:
            highlights.append(clean)

    return highlights[:4], dining[:3]


def _capitalised_tokens(sentence: str, city_lower: str) -> List[str]:
    """Return capitalised tokens that look like landmark names."""

    tokens = re.findall(r"\b[A-Za-z][\w']*\b", sentence)
    capitalised: List[str] = []
    for index, token in enumerate(tokens):
        if index == 0:
            # The first word is almost always capitalised; skip it to avoid
            # treating leading adjectives as points of interest.
            continue
        if not token[0].isupper():
            continue
        lower_token = token.lower()
        if lower_token == city_lower:
            continue
        if token in {"The", "A", "An"}:
            continue
        capitalised.append(token)
    return capitalised


def _fallback_highlights(city: str, interests: Iterable[str]) -> List[str]:
    base = [
        f"Historic walking tour of {city}",
        f"Visit the central market in {city}",
        f"Sunset viewpoint overlooking {city}",
    ]
    interest_map = {
        "food": f"Try a guided street-food crawl in {city}",
        "museums": f"Reserve tickets for a signature museum in {city}",
        "outdoors": f"Add a day trip to nearby nature around {city}",
        "nightlife": f"Plan an evening in the creative districts of {city}",
        "kid-friendly": f"Include an interactive science or play museum in {city}",
    }
    for interest in interests:
        key = interest.lower()
        if key in interest_map:
            base.append(interest_map[key])
    return base


def _build_experiences(city: str, highlights: Iterable[str], interests: Iterable[str]) -> List[str]:
    experiences: List[str] = []
    seen = set()
    for highlight in highlights:
        trimmed = highlight.strip()
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        experiences.append(trimmed)
    if "relaxation" in {i.lower() for i in interests}:
        experiences.append(f"Set aside a spa or thermal-bath session in {city}.")
    return experiences


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        trimmed = value.strip()
        if not trimmed or trimmed.lower() in seen:
            continue
        seen.add(trimmed.lower())
        out.append(trimmed)
    return out

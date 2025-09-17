# Trip Planner Orchestrator

This repository contains a lightweight travel-planning orchestrator that combines
heuristic budgeting with optional web and LLM enrichment. It can be run as an
API (via FastAPI) or exercised locally to inspect orchestration logs.

## Prerequisites

- Python 3.11 or newer.
- [uv](https://github.com/astral-sh/uv) (recommended) or `pip` for dependency
  management.
- An OpenAI API key stored in `OPENAI_API_KEY` if you want to call the hosted
  model. Without it, the system falls back to deterministic stub responses.
- A Tavily API key (`TAVILY_API_KEY`) for live web search. When absent the
  orchestrator retains heuristic city highlights and logs the fallback path.
- Optional: credentials for your preferred web-search adapter. The default
  implementation does not require authentication, but you can plug in your own
  tool by editing `app/tools/websearch.py`.

## Setup

Clone the repository and install dependencies:

```bash
# Create an isolated environment (uv manages this automatically)
uv sync

# Alternatively, using pip
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running the service

Launch the API server with uvicorn:

```bash
uv run uvicorn app.main:app --reload
```

The primary endpoint is `POST /api/plan`, which accepts the `TripRequest`
schema from `app/schemas.py`. Example payload:

```json
{
  "origin": "San Francisco",
  "destinations": ["Paris", "Amsterdam", "Berlin"],
  "dates": {"start": "2025-10-10", "end": "2025-10-20"},
  "budget_total": 4500,
  "currency": "USD",
  "party": {"adults": 2, "children": 1},
  "prefs": {"objective": "balanced"}
}
```

The response now includes `source_links`, `snippets`, and `agent_context`
entries so you can attribute recommendations and inspect intermediate state.
Each bundle’s cost ledger tracks stays, intercity transport, food, activities,
and remaining misc headroom so you can compare the spend against the requested
budget at a glance.

> **Tip:** The React frontend under `trip_planner_frontend/` targets
> `http://localhost:8000/api/plan` by default. Set
> `VITE_API_BASE_URL=http://localhost:8000` in
> `trip_planner_frontend/.env.local` while the FastAPI server is running to
> see live orchestration output instead of the bundled sample itinerary.

If you need to restrict browser origins, configure
`TRIP_PLANNER_ALLOWED_ORIGINS` (comma-separated list). The default `*` keeps
local development simple by allowing the Vite dev server to call the API.

## Inspecting orchestration logs

Run the debugging helper to see step-by-step orchestration output:

```bash
uv run python debug_orchestrator.py
```

Key log lines include:

- Search activity (queries issued, hit counts, allowed domains).
- Baseline bundle summaries with cost composition.
- LLM invocation status, including the keys returned by the model or stub.

Set the environment variable `TRIP_PLANNER_LOG_LEVEL=DEBUG` to increase
verbosity (see `debug_orchestrator.py` for an example of adjusting log levels).

## Testing

Execute the asynchronous test suite with:

```bash
uv run pytest
```

All new features should include coverage in `tests/` where practical.

## Repository structure

- `app/orchestrator.py` – central orchestration logic and bundle scoring.
- `app/llm.py` – wrapper around the OpenAI client with graceful fallbacks.
- `app/agents/` – specialised heuristics for foundation, destination, and logistics insights.
- `debug_orchestrator.py` – quick entry point for manual experiments.
- `trip_planner_frontend/` – React + Vite workspace that mirrors conversational planners and surfaces booking links alongside orchestrator bundles. See its README for setup instructions.

Happy trip planning!

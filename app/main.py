from fastapi import FastAPI
from app.orchestrator import orchestrate_llm_trip
from fastapi import Body
app = FastAPI(title="Trip Planner Agentic API")

@app.post("/trip/llm_only")
async def trip_llm_only(
    origin: str = Body(...),
    purpose: str = Body(...),
    budget_total: float = Body(...),
    currency: str = Body("USD"),
    dates: dict = Body(...),
    party: dict = Body(...),
    constraints: dict = Body({}),
    interests: list = Body([]),
    destinations: list = Body(...),
    allow_domains: list[str] | None = Body(None),
    deny_domains: list[str] | None = Body(None)
):
    payload = {
        "origin": origin,
        "purpose": purpose,
        "budget_total": budget_total,
        "currency": currency,
        "dates": dates,
        "party": party,
        "constraints": constraints,
        "interests": interests,
        "destinations": destinations
    }
    return await orchestrate_llm_trip(payload, allow_domains, deny_domains)

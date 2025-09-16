# debug_orchestrator.py
import asyncio
import json

from app.orchestrator import orchestrate_llm_trip


async def main():
    payload = {
        "origin": "San Francisco",
        "purpose": "family vacation",
        "budget_total": 4500,
        "currency": "USD",
        "dates": {"start": "2025-07-10", "end": "2025-07-20"},
        "party": {"adults": 2, "children": 1, "seniors": 0},
        "constraints": {"diet": ["vegetarian"], "max_flight_hours": 12},
        "interests": ["culture", "food", "museums", "kid-friendly"],
        "destinations": ["Paris", "Amsterdam", "Berlin"],
        "allow_domains": [
            "parisinfo.com",
            "iamsterdam.com",
            "visitberlin.de",
            "bahn.com",
            "sncf-connect.com",
        ],
        "deny_domains": [
            ".*coupon.*",
            ".*blogspot.*",
            ".*ai-generated.*",
        ],
    }

    # Call orchestrator directly
    result = await orchestrate_llm_trip(payload)
    print("➡️ Orchestrator returned:\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

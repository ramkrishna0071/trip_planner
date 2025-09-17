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
        "dates": {"start": "2025-10-10", "end": "2025-10-20"},
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
            "airfrance.com",
            "klm.com",
            "delta.com",
            "tripadvisor.com",
            "lonelyplanet.com",
            "culturetrip.com",
            "fodor.com",
            "timeout.com",
            "theinfatuation.com",
            "yelp.com",
            "familyvacationcritic.com",
            "travelandleisure.com",
            "nationalgeographic.com/travel",
            "smithsonianmag.com/travel",
            "roadtrippers.com",
            "atlasobscura.com",
            "travelawaits.com",
            "familytravel.org",
            "familytraveller.com",
            "familyvacationhub.com",
            "familyvacationcritic.com",
            "airbnb.com",
            "vrbo.com",
            "booking.com",
            "expedia.com",
            "hotels.com",
            "trip.com",
            "kayak.com",
            "skyscanner.com",
            "google.com/travel",
            "rome2rio.com",
            "omio.com",
            "getyourguide.com",
            "viator.com",
            "klook.com",
            "musement.com",
            "thediscoverer.com",
            "travelzoo.com",
            "secretflying.com",
            "thepointsguy.com",
            "flyertalk.com",
            "awardwallet.com",
            "seatguru.com",
            "travelcodex.com",
            "theflightdeal.com",
            "airfarewatchdog.com",
            "scottscheapflights.com",
            "travelocity.com",
            "priceline.com",
            "agoda.com",
            "trivago.com",
            "hotwire.com",
            "lastminute.com",
            "ebookers.com",
            "travelpirates.com",
            "holidaypirates.com",
            "travelzoo.com",
            "secretflying.com",
            "thepointsguy.com",
            "flyertalk.com",
            "awardwallet.com"
        ],
        "deny_domains": [
            ".*coupon.*",
            ".*blogspot.*",
            ".*ai-generated.*",
            ".*pinterest.*"
        ],
    }

    # Call orchestrator directly
    result = await orchestrate_llm_trip(payload)
    print("➡️ Orchestrator returned:\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

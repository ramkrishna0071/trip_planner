import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# --- test payload ---
payload = {
    "origin": "San Francisco",
    "purpose": "family vacation",
    "budget_total": 4500,
    "currency": "USD",
    "dates": {
        "start": "2025-07-10",
        "end": "2025-07-20"
    },
    "party": {
        "adults": 2,
        "children": 1,
        "seniors": 0
    },
    "constraints": {
        "diet": ["vegetarian"],
        "max_flight_hours": 12
    },
    "interests": ["culture", "food", "museums", "kid-friendly"],
    "destinations": ["Paris", "Amsterdam", "Berlin"],
    "allow_domains": [
        "parisinfo.com",
        "iamsterdam.com",
        "visitberlin.de",
        "bahn.com",
        "sncf-connect.com"
    ],
    "deny_domains": [
        ".*coupon.*",
        ".*blogspot.*",
        ".*ai-generated.*"
    ]
}

def run_test():
    url = f"{BASE_URL}/trip/llm_only"
    headers = {"Content-Type": "application/json"}

    print(f"➡️ Sending POST {url}")
    print(json.dumps(payload, indent=2))

    resp = requests.post(url, headers=headers, json=payload)

    print(f"\n⬅️ Status: {resp.status_code}")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    run_test()

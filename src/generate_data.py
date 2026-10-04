import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.config import ADDRESS_CSV, CLAIMS_CSV, CUSTOMER_CSV, DATA_DIR

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_CLAIMS = 5000
FRAUD_RATE = 0.08

PRODUCT_LINES = [
    "Latitude 5540", "Latitude 7440", "Precision 5680",
    "OptiPlex 7010", "XPS 15 9530", "Inspiron 16 5630",
    "PowerEdge R760", "Vostro 3420",
]

COMPONENTS = {
    "GPU": (400, 950),
    "Motherboard": (350, 800),
    "Battery": (50, 150),
    "LCD Screen": (200, 500),
    "SSD": (100, 300),
    "RAM Module": (40, 120),
    "Keyboard": (30, 80),
    "Fan Assembly": (20, 60),
}

CHANNELS = ["Chat", "Phone", "Web Portal", "Email"]

FRAUD_DESCRIPTIONS = [
    "Device not turning on. Screen completely black. Need immediate replacement.",
    "System crashes randomly. GPU failure detected. Please send replacement ASAP.",
    "Laptop overheating and shutting down. Motherboard issue. Urgent replacement needed.",
    "Battery swollen and dangerous. Need urgent replacement shipped immediately.",
    "Display cracked internally. No physical damage. Need LCD replacement urgently.",
]

LEGIT_DESCRIPTIONS = [
    "Getting error code 2000-0333 during diagnostics. Fan seems louder than usual.",
    "Blue screen WHEA_UNCORRECTABLE_ERROR appearing after BIOS update to version 1.12.",
    "Battery health showing 23% in Dell Power Manager after 14 months of use.",
    "Intermittent flickering on external monitor via USB-C dock model WD19S.",
    "SSD SMART status showing caution. Reallocated sector count increasing weekly.",
    "Keyboard key 'E' registering double input. Happens roughly every 10th press.",
    "Trackpad not responding after driver update. Tried rolling back without success.",
    "System fan making grinding noise at 3000+ RPM. Started 2 weeks ago.",
]


def generate_claims(n):
    records = []
    customer_pool = [f"CUST-{random.randint(10000, 99999)}" for _ in range(int(n * 0.7))]
    address_pool = [f"ADDR-{random.randint(1000, 9999)}" for _ in range(int(n * 0.5))]

    for i in range(n):
        is_fraud = 1 if random.random() < FRAUD_RATE else 0

        if is_fraud:
            customer_id = random.choice(customer_pool[:50])
            address_id = random.choice(address_pool[:20])
            component = random.choice(["GPU", "Motherboard", "LCD Screen"])
            description = random.choice(FRAUD_DESCRIPTIONS)
            diagnostic_run = random.choice([False, False, False, True])
            diagnostic_result = "NOT_RUN" if not diagnostic_run else "PASS"
            device_age = random.randint(30, 300)
            channel = random.choice(["Chat", "Web Portal"])
        else:
            customer_id = random.choice(customer_pool)
            address_id = random.choice(address_pool)
            component = random.choice(list(COMPONENTS.keys()))
            description = random.choice(LEGIT_DESCRIPTIONS)
            diagnostic_run = random.choice([True, True, True, False])
            diagnostic_result = random.choice(["FAIL", "FAIL", "PASS"]) if diagnostic_run else "NOT_RUN"
            device_age = random.randint(60, 900)
            channel = random.choice(CHANNELS)

        comp_min, comp_max = COMPONENTS[component]
        purchase_date = datetime.now() - timedelta(days=device_age)

        records.append({
            "claim_id": f"CLM-2024-{str(i + 1).zfill(5)}",
            "claim_date": fake.date_time_between(start_date="-90d", end_date="now"),
            "service_tag": fake.bothify(text="???#??#").upper(),
            "customer_id": customer_id,
            "customer_name": fake.name(),
            "email_domain": random.choice(["gmail.com", "yahoo.com", "outlook.com", "company.com"]),
            "shipping_address_id": address_id,
            "shipping_state": fake.state_abbr(),
            "shipping_zip": fake.zipcode(),
            "product_line": random.choice(PRODUCT_LINES),
            "device_purchase_date": purchase_date.date(),
            "warranty_end_date": (purchase_date + timedelta(days=1095)).date(),
            "device_age_days": device_age,
            "component_requested": component,
            "component_value_usd": round(random.uniform(comp_min, comp_max), 2),
            "claim_description": description,
            "diagnostic_run": diagnostic_run,
            "diagnostic_result": diagnostic_result,
            "support_channel": channel,
            "agent_id": f"AGT-{random.randint(1000, 1200)}",
            "claim_status": "Pending",
            "is_fraud": is_fraud,
        })

    return pd.DataFrame(records)


def enrich_customer_history(claims_df):
    history = []
    for cust_id, c in claims_df.groupby("customer_id"):
        history.append({
            "customer_id": cust_id,
            "total_claims_lifetime": len(c),
            "claims_last_90_days": len(c),
            "claims_last_30_days": int((c["claim_date"] > datetime.now() - timedelta(days=30)).sum()),
            "unique_devices_claimed": c["service_tag"].nunique(),
            "unique_addresses_used": c["shipping_address_id"].nunique(),
            "avg_component_value": round(c["component_value_usd"].mean(), 2),
            "max_component_value": round(c["component_value_usd"].max(), 2),
            "pct_high_value_claims": round((c["component_value_usd"] > 300).sum() / len(c), 2),
            "previous_rejections": random.randint(0, 3),
            "days_since_last_claim": random.randint(1, 180),
            "account_age_days": random.randint(30, 1500),
        })
    return pd.DataFrame(history)


def enrich_address_signals(claims_df):
    signals = []
    for addr_id, a in claims_df.groupby("shipping_address_id"):
        signals.append({
            "shipping_address_id": addr_id,
            "unique_customers_at_address": a["customer_id"].nunique(),
            "unique_service_tags": a["service_tag"].nunique(),
            "total_claims_at_address": len(a),
            "is_residential": random.choice([True, True, True, False]),
            "address_fraud_rate": round(a["is_fraud"].mean(), 2),
        })
    return pd.DataFrame(signals)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    claims = generate_claims(NUM_CLAIMS)
    customer_history = enrich_customer_history(claims)
    address_signals = enrich_address_signals(claims)

    claims.to_csv(CLAIMS_CSV, index=False)
    customer_history.to_csv(CUSTOMER_CSV, index=False)
    address_signals.to_csv(ADDRESS_CSV, index=False)

    print(f"Generated {len(claims)} claims")
    print(f"  Fraud: {claims['is_fraud'].sum()}")
    print(f"  Legit: {len(claims) - claims['is_fraud'].sum()}")
    print(f"Customer profiles: {len(customer_history)}")
    print(f"Address profiles:  {len(address_signals)}")

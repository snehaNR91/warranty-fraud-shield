import pandas as pd

from src.config import ADDRESS_CSV, CLAIMS_CSV, CUSTOMER_CSV, DATA_DIR

SOURCES = ["", " (1)"]

CATEGORY_BY_PREFIX = {
    "Latitude": "Laptop", "Inspiron": "Laptop", "Vostro": "Laptop", "XPS": "Laptop",
    "OptiPlex": "Desktop", "PowerEdge": "Server", "Precision": "Workstation",
}


def load_claims():
    frames = []
    for i, suffix in enumerate(SOURCES):
        df = pd.read_csv(DATA_DIR / f"claims{suffix}.csv", parse_dates=["claim_date"])
        df["source_batch"] = i + 1
        df["claim_id"] = df["claim_id"].str.replace("CLM-2024-", f"CLM-B{i + 1}-", regex=False)
        frames.append(df)
    claims = pd.concat(frames, ignore_index=True)

    category = claims["product_line"].str.split().str[0].map(CATEGORY_BY_PREFIX)
    claims["product_category"] = claims["product_category"].fillna(category).fillna("Other")
    claims["warranty_type"] = claims["warranty_type"].fillna("Unknown")
    return claims.sort_values("claim_date").reset_index(drop=True)


def build_customer_history(claims):
    now = claims["claim_date"].max()
    prior_age = pd.concat(
        pd.read_csv(DATA_DIR / f"customer_history{s}.csv", usecols=["customer_id", "account_age_days"])
        for s in SOURCES
    ).groupby("customer_id")["account_age_days"].max()

    g = claims.groupby("customer_id")
    hist = pd.DataFrame({
        "total_claims_lifetime": g.size(),
        "claims_last_90_days": g["claim_date"].apply(lambda d: (d > now - pd.Timedelta(days=90)).sum()),
        "claims_last_30_days": g["claim_date"].apply(lambda d: (d > now - pd.Timedelta(days=30)).sum()),
        "unique_devices_claimed": g["service_tag"].nunique(),
        "unique_addresses_used": g["shipping_address_id"].nunique(),
        "avg_component_value": g["component_value_usd"].mean().round(2),
        "max_component_value": g["component_value_usd"].max().round(2),
        "pct_high_value_claims": g["component_value_usd"].apply(lambda v: (v > 300).mean()).round(2),
        "previous_rejections": g["claim_status"].apply(lambda s: (s == "Rejected").sum()),
        "days_since_last_claim": (now - g["claim_date"].max()).dt.days,
        "account_age_days": prior_age,
        "customer_fraud_ratio": g["is_fraud"].mean().round(2),
    })
    return hist.reset_index()


def build_address_signals(claims):
    residential = pd.concat(
        pd.read_csv(DATA_DIR / f"address_signals{s}.csv", usecols=["shipping_address_id", "is_residential"])
        for s in SOURCES
    ).drop_duplicates("shipping_address_id").set_index("shipping_address_id")["is_residential"]

    g = claims.groupby("shipping_address_id")
    addr = pd.DataFrame({
        "unique_customers_at_address": g["customer_id"].nunique(),
        "unique_service_tags": g["service_tag"].nunique(),
        "total_claims_at_address": g.size(),
        "is_residential": residential,
        "address_fraud_rate": g["is_fraud"].mean().round(2),
        "pct_high_value_at_address": g["component_value_usd"].apply(lambda v: (v > 300).mean()).round(2),
        "unique_products_at_address": g["product_line"].nunique(),
        "avg_days_between_claims": g["claim_date"].apply(
            lambda d: d.sort_values().diff().dt.total_seconds().mean() / 86400).round(1),
    })
    return addr.rename_axis("shipping_address_id").reset_index()


if __name__ == "__main__":
    claims = load_claims()
    customers = build_customer_history(claims)
    addresses = build_address_signals(claims)

    claims.to_csv(CLAIMS_CSV, index=False)
    customers.to_csv(CUSTOMER_CSV, index=False)
    addresses.to_csv(ADDRESS_CSV, index=False)

    print(f"Combined claims: {len(claims)} (fraud: {claims['is_fraud'].sum()})")
    print(claims.groupby("source_batch")["is_fraud"].agg(["size", "sum"]))
    print(f"Customer profiles: {len(customers)}")
    print(f"Address profiles:  {len(addresses)}")

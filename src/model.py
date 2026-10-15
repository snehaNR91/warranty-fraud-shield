"""Train the Gradient Boosting fraud model and save it with its feature list and label encoders."""

import logging
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import ADDRESS_CSV, CLAIMS_CSV, CUSTOMER_CSV, MODEL_DIR, MODEL_PKL

logger = logging.getLogger(__name__)

FEATURES = [
    "device_age_days",
    "component_value_usd",
    "component_encoded",
    "channel_encoded",
    "product_encoded",
    "diagnostic_not_run",
    "diagnostic_passed",
    "description_length",
    "warranty_remaining_days",
    "total_claims_lifetime",
    "claims_last_90_days",
    "claims_last_30_days",
    "unique_devices_claimed",
    "unique_addresses_used",
    "avg_component_value",
    "max_component_value",
    "pct_high_value_claims",
    "previous_rejections",
    "days_since_last_claim",
    "account_age_days",
    "unique_customers_at_address",
    "total_claims_at_address",
    "address_fraud_rate",
]
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_model() -> None:
    """Train on the combined data, log test-set metrics and write the model bundle to ``MODEL_PKL``."""
    claims = pd.read_csv(CLAIMS_CSV)
    df = claims.merge(pd.read_csv(CUSTOMER_CSV), on="customer_id", how="left")
    df = df.merge(pd.read_csv(ADDRESS_CSV), on="shipping_address_id", how="left")

    df["description_length"] = df["claim_description"].str.len()
    df["diagnostic_not_run"] = (df["diagnostic_result"] == "NOT_RUN").astype(int)
    df["diagnostic_passed"] = (df["diagnostic_result"] == "PASS").astype(int)
    df["warranty_remaining_days"] = (pd.to_datetime(df["warranty_end_date"]) - pd.to_datetime(df["claim_date"])).dt.days

    le_component, le_channel, le_product = LabelEncoder(), LabelEncoder(), LabelEncoder()
    df["component_encoded"] = le_component.fit_transform(df["component_requested"])
    df["channel_encoded"] = le_channel.fit_transform(df["support_channel"])
    df["product_encoded"] = le_product.fit_transform(df["product_line"])

    x_all = df[FEATURES].fillna(0)
    y_all = df["is_fraud"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
    )

    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    logger.info("ROC AUC: %.4f", roc_auc_score(y_test, y_prob))

    importance = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    logger.info("Top 10 features:\n%s", importance.head(10).to_string(index=False))

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": FEATURES,
                "le_component": le_component,
                "le_channel": le_channel,
                "le_product": le_product,
            },
            f,
        )
    logger.info("Model saved to %s", MODEL_PKL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    train_model()

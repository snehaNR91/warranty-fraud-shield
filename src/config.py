from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

CLAIMS_CSV = DATA_DIR / "combined_claims.csv"
CUSTOMER_CSV = DATA_DIR / "combined_customer_history.csv"
ADDRESS_CSV = DATA_DIR / "combined_address_signals.csv"
MODEL_PKL = MODEL_DIR / "fraud_model.pkl"

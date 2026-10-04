from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

CLAIMS_CSV = DATA_DIR / "claims.csv"
CUSTOMER_CSV = DATA_DIR / "customer_history.csv"
ADDRESS_CSV = DATA_DIR / "address_signals.csv"
MODEL_PKL = MODEL_DIR / "fraud_model.pkl"

# AI Shield - Warranty / RMA Fraud Detection

```
warranty-fraud-shield/
  app.py               Streamlit dashboard (Dashboard, Claim Analyzer, Model Insights)
  requirements.txt
  data/                Generated CSVs (claims, customer_history, address_signals)
  models/              Trained model (fraud_model.pkl)
  src/
    config.py          Shared paths
    combine_data.py    Merges the two source datasets into data/combined_*.csv
    generate_data.py   Synthetic data generator (optional; writes to data/combined_*.csv)
    model.py           Model training
```

## Run (from this folder)

```
pip install -r requirements.txt
python -m src.combine_data
python -m src.model
streamlit run app.py
```

## Investigation Agent (LLM)

The "🕵️ Investigation Agent" page (`src/investigation_agent.py`) uses an LLM with tool calling to investigate a
flagged claim and draft a recommendation for human approval. Configure it in `.streamlit/secrets.toml`
(git-ignored) or environment variables:

```
GENAI_URL     = "https://aia.gateway.dell.com/genai/dev/v1"   # OpenAI-compatible endpoint
GENAI_MODEL   = "gpt-oss-120b"                                # any model that supports tool calling
TOKEN_URL     = "https://www.dell.com/di/api/v3/oauth/token"  # OAuth2 client-credentials endpoint
CLIENT_ID     = "..."
CLIENT_SECRET = "..."
```

The gateway is on the Dell network, so the agent only works where that network is reachable.

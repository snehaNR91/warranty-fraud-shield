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

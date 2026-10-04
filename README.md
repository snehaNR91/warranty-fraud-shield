# AI Shield - Warranty / RMA Fraud Detection

```
warranty-fraud-shield/
  app.py               Streamlit dashboard (Dashboard, Claim Analyzer, Model Insights)
  requirements.txt
  data/                Generated CSVs (claims, customer_history, address_signals)
  models/              Trained model (fraud_model.pkl)
  src/
    config.py          Shared paths
    generate_data.py   Synthetic data generator
    model.py           Model training
```

## Run (from this folder)

```
pip install -r requirements.txt
python -m src.generate_data
python -m src.model
streamlit run app.py
```

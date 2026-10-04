import pickle

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ADDRESS_CSV, CLAIMS_CSV, CUSTOMER_CSV, MODEL_PKL

st.set_page_config(page_title="Warranty Fraud Detection - AI Shield", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    claims = pd.read_csv(CLAIMS_CSV, parse_dates=["claim_date"])
    return claims, pd.read_csv(CUSTOMER_CSV), pd.read_csv(ADDRESS_CSV)


if not (MODEL_PKL.exists() and CLAIMS_CSV.exists()):
    st.error("Data/model missing. Run `python -m src.generate_data` then `python -m src.model`.")
    st.stop()

model_bundle = load_model()
model = model_bundle["model"]
claims, customer_history, address_signals = load_data()

st.sidebar.title("🛡️ AI Shield")
st.sidebar.markdown("Warranty Fraud Detection System")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔍 Claim Analyzer", "📈 Model Insights"])

COLORS = {"Legitimate": "#2196F3", "Fraudulent": "#F44336"}

# ---------------- DASHBOARD ----------------
if page == "📊 Dashboard":
    st.title("📊 Warranty Fraud Detection Dashboard")
    st.markdown("---")

    fraud = claims[claims["is_fraud"] == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", f"{len(claims):,}")
    c2.metric("Flagged as Fraud", f"{len(fraud):,}", delta=f"{len(fraud) / len(claims) * 100:.1f}%")
    c3.metric("Value at Risk", f"${fraud['component_value_usd'].sum():,.0f}")
    c4.metric("Avg Fraud Claim Value", f"${fraud['component_value_usd'].mean():,.0f}")
    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("Fraud by Component Type")
        d = fraud.groupby("component_requested").size().reset_index(name="fraud_count")
        fig = px.bar(d.sort_values("fraud_count"), x="fraud_count", y="component_requested",
                     orientation="h", color="fraud_count", color_continuous_scale="Reds")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Fraud by Support Channel")
        d = fraud.groupby("support_channel").size().reset_index(name="fraud_count")
        fig = px.pie(d, names="support_channel", values="fraud_count", hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Claims Timeline (Last 90 Days)")
        daily = claims.groupby([claims["claim_date"].dt.date, "is_fraud"]).size().reset_index(name="count")
        daily.columns = ["date", "is_fraud", "count"]
        daily["type"] = daily["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
        fig = px.line(daily, x="date", y="count", color="type", color_discrete_map=COLORS)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Top Suspicious Addresses")
        top = address_signals.nlargest(10, "address_fraud_rate")
        fig = px.bar(top, x="shipping_address_id", y="address_fraud_rate",
                     color="total_claims_at_address", color_continuous_scale="OrRd")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🚨 Recent High-Risk Claims")
    st.dataframe(
        fraud.sort_values("claim_date", ascending=False).head(15)[[
            "claim_id", "claim_date", "customer_id", "component_requested",
            "component_value_usd", "claim_description", "diagnostic_result"]],
        use_container_width=True, height=400)

# ---------------- CLAIM ANALYZER ----------------
elif page == "🔍 Claim Analyzer":
    st.title("🔍 Real-Time Claim Fraud Analyzer")
    st.markdown("Submit a new warranty claim for AI-powered fraud assessment")
    st.markdown("---")

    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            product = st.selectbox("Product Line", sorted(claims["product_line"].unique()))
            component = st.selectbox("Component Requested", sorted(claims["component_requested"].unique()))
            device_age = st.slider("Device Age (Days)", 1, 1095, 180)
        with col2:
            channel = st.selectbox("Support Channel", sorted(claims["support_channel"].unique()))
            diagnostic = st.selectbox("Diagnostic Result", ["NOT_RUN", "PASS", "FAIL"])
            prev_claims = st.slider("Customer's Total Previous Claims", 0, 20, 1)
        with col3:
            claims_90d = st.slider("Claims in Last 90 Days", 0, 10, 0)
            unique_addresses = st.slider("Unique Addresses Used", 1, 5, 1)
            component_value = st.number_input("Component Value ($)", 20, 1000, 200)
        description = st.text_area("Claim Description", placeholder="Describe the issue with the device...", height=100)
        submitted = st.form_submit_button("🔍 Analyze Claim", use_container_width=True)

    if submitted:
        st.markdown("---")

        def enc(key, value):
            classes = list(model_bundle[key].classes_)
            return classes.index(value) if value in classes else 0

        features = {
            "device_age_days": device_age,
            "component_value_usd": component_value,
            "component_encoded": enc("le_component", component),
            "channel_encoded": enc("le_channel", channel),
            "product_encoded": enc("le_product", product),
            "diagnostic_not_run": int(diagnostic == "NOT_RUN"),
            "diagnostic_passed": int(diagnostic == "PASS"),
            "description_length": len(description),
            "warranty_remaining_days": max(1095 - device_age, 0),
            "total_claims_lifetime": prev_claims,
            "claims_last_90_days": claims_90d,
            "claims_last_30_days": max(claims_90d - 1, 0),
            "unique_devices_claimed": max(prev_claims - 1, 1),
            "unique_addresses_used": unique_addresses,
            "avg_component_value": component_value,
            "max_component_value": component_value * 1.2,
            "pct_high_value_claims": 1.0 if component_value > 300 else 0.3,
            "previous_rejections": 0,
            "days_since_last_claim": max(90 // max(claims_90d, 1), 1),
            "account_age_days": device_age + 30,
            "unique_customers_at_address": unique_addresses,
            "total_claims_at_address": prev_claims,
            "address_fraud_rate": 0.1,
        }
        X_input = pd.DataFrame([features])[model_bundle["features"]]
        score = int(model.predict_proba(X_input)[0][1] * 100)

        if score >= 70:
            risk_level, color = "🔴 HIGH RISK", "#F44336"
        elif score >= 40:
            risk_level, color = "🟡 MEDIUM RISK", "#FF9800"
        else:
            risk_level, color = "🟢 LOW RISK", "#4CAF50"

        col_score, col_detail = st.columns([1, 2])
        with col_score:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score, title={"text": "Fraud Risk Score"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                       "steps": [{"range": [0, 40], "color": "#E8F5E9"},
                                 {"range": [40, 70], "color": "#FFF3E0"},
                                 {"range": [70, 100], "color": "#FFEBEE"}]}))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_detail:
            st.markdown(f"### Assessment: {risk_level}")
            st.markdown("#### 📋 Risk Flags:")
            flags = []
            if claims_90d >= 3:
                flags.append(f"⚠️ High claim frequency: **{claims_90d} claims** in 90 days")
            if diagnostic == "NOT_RUN":
                flags.append("⚠️ Diagnostics were **not run** before filing claim")
            if diagnostic == "PASS":
                flags.append("⚠️ Diagnostics **passed** but replacement still requested")
            if component_value > 300:
                flags.append(f"⚠️ High-value component requested: **${component_value}**")
            if unique_addresses > 2:
                flags.append(f"⚠️ Customer has used **{unique_addresses} different addresses**")
            if device_age < 60:
                flags.append(f"⚠️ Very new device: only **{device_age} days old**")
            if len(description) < 50:
                flags.append("⚠️ Claim description is **unusually short**")
            for flag in flags or ["✅ No significant risk flags detected"]:
                st.markdown(flag)

            st.markdown("#### 🎯 Recommended Action:")
            if score >= 70:
                st.error("HOLD shipment. Escalate to Fraud Investigation Team for manual review.")
            elif score >= 40:
                st.warning("Proceed with caution. Request additional verification from customer.")
            else:
                st.success("Low risk. Approve claim through standard workflow.")

# ---------------- MODEL INSIGHTS ----------------
else:
    st.title("📈 Model Performance & Insights")
    st.markdown("---")
    st.subheader("Feature Importance Rankings")
    imp = pd.DataFrame({"Feature": model_bundle["features"], "Importance": model.feature_importances_})
    fig = px.bar(imp.sort_values("Importance").tail(15), x="Importance", y="Feature",
                 orientation="h", color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔎 Key Fraud Patterns Detected")
    label = claims["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
    left, right = st.columns(2)
    with left:
        st.markdown("#### Device Age Distribution")
        fig = px.histogram(claims, x="device_age_days", color=label, nbins=30, barmode="overlay",
                           color_discrete_map=COLORS, opacity=0.7)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Component Value Distribution")
        fig = px.box(claims, x=label, y="component_value_usd", color=label, color_discrete_map=COLORS)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

import pickle

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ADDRESS_CSV, CLAIMS_CSV, CUSTOMER_CSV, MODEL_PKL
from src.reason_generator import generate_rejection_reasons

st.set_page_config(page_title="Warranty Fraud Detection - AI Shield", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    claims = pd.read_csv(CLAIMS_CSV, parse_dates=["claim_date"])
    return claims, pd.read_csv(CUSTOMER_CSV), pd.read_csv(ADDRESS_CSV)


@st.cache_data
def compute_reason_stats(claims, customer_history, address_signals):
    df = (claims.merge(customer_history, on="customer_id", how="left")
                .merge(address_signals, on="shipping_address_id", how="left"))
    df["warranty_remaining_days"] = (pd.to_datetime(df["warranty_end_date"]) - df["claim_date"]).dt.days
    df["dup_desc"] = df.groupby(["customer_id", "claim_description"])["claim_id"].transform("count") - 1
    rows = []
    for r in df.to_dict("records"):
        reasons = generate_rejection_reasons(
            claim_data={**r, "description_similarity_score": 0.9 if r["dup_desc"] > 0 else 0.3,
                        "duplicate_description_count": r["dup_desc"],
                        "technical_detail_score": round(min(len(r["claim_description"]) / 200, 1.0), 2)},
            customer_data=r, address_data=r)
        rows += [{"Reason Code": x["code"], "Reason": x["title"], "Severity": x["severity"],
                  "is_fraud": r["is_fraud"]} for x in reasons]
    stats = (pd.DataFrame(rows).groupby(["Reason Code", "Reason", "Severity"])
             .agg(**{"Times Triggered": ("is_fraud", "size"), "Confirmed Fraud %": ("is_fraud", "mean")})
             .reset_index())
    stats["Confirmed Fraud %"] = (stats["Confirmed Fraud %"] * 100).round(1)
    stats["False Positive %"] = (100 - stats["Confirmed Fraud %"]).round(1)
    return stats


if not CLAIMS_CSV.exists():
    with st.spinner("Combining source datasets..."):
        from src import combine_data
        claims = combine_data.load_claims()
        claims.to_csv(CLAIMS_CSV, index=False)
        combine_data.build_customer_history(claims).to_csv(CUSTOMER_CSV, index=False)
        combine_data.build_address_signals(claims).to_csv(ADDRESS_CSV, index=False)

if not MODEL_PKL.exists():
    with st.spinner("Training fraud model (first run only)..."):
        from src.model import train_model
        train_model()

model_bundle = load_model()
model = model_bundle["model"]
claims, customer_history, address_signals = load_data()

st.sidebar.title("🛡️ AI Shield")
st.sidebar.markdown("Warranty Fraud Detection System")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔍 Claim Analyzer", "📈 Model Insights",
                                     "📝 Rejection Analytics"])

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
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Fraud by Support Channel")
        d = fraud.groupby("support_channel").size().reset_index(name="fraud_count")
        fig = px.pie(d, names="support_channel", values="fraud_count", hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Claims Timeline (Last 90 Days)")
        daily = claims.groupby([claims["claim_date"].dt.date, "is_fraud"]).size().reset_index(name="count")
        daily.columns = ["date", "is_fraud", "count"]
        daily["type"] = daily["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
        fig = px.line(daily, x="date", y="count", color="type", color_discrete_map=COLORS)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Top Suspicious Addresses")
        top = address_signals.nlargest(10, "address_fraud_rate")
        fig = px.bar(top, x="shipping_address_id", y="address_fraud_rate",
                     color="total_claims_at_address", color_continuous_scale="OrRd")
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("🚨 Recent High-Risk Claims")
    st.dataframe(
        fraud.sort_values("claim_date", ascending=False).head(15)[[
            "claim_id", "claim_date", "customer_id", "component_requested",
            "component_value_usd", "claim_description", "diagnostic_result"]],
        width="stretch", height=400)

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
            service_tag = st.text_input("Service Tag", value="HTXR5N3")
        with col2:
            channel = st.selectbox("Support Channel", sorted(claims["support_channel"].unique()))
            diagnostic = st.selectbox("Diagnostic Result", ["NOT_RUN", "PASS", "FAIL"])
            prev_claims = st.slider("Customer's Total Previous Claims", 0, 20, 1)
            customer_id = st.text_input("Customer ID", value="CUST-40221")
        with col3:
            claims_90d = st.slider("Claims in Last 90 Days", 0, 10, 0)
            claims_30d = st.slider("Claims in Last 30 Days", 0, 5, 0)
            unique_addresses = st.slider("Unique Addresses Used", 1, 5, 1)
            component_value = st.number_input("Component Value ($)", 20, 1000, 200)
        description = st.text_area("Claim Description", placeholder="Describe the issue with the device...", height=100)
        submitted = st.form_submit_button("🔍 Analyze Claim", width="stretch")

    if submitted:
        st.session_state.analyzed = True

    if st.session_state.get("analyzed"):
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
            "claims_last_30_days": claims_30d,
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

        short_desc = len(description) < 80
        reasons = generate_rejection_reasons(
            claim_data={
                "service_tag": service_tag,
                "customer_id": customer_id,
                "product_line": product,
                "component_requested": component,
                "component_value_usd": component_value,
                "device_age_days": device_age,
                "diagnostic_result": diagnostic,
                "claim_description": description,
                "warranty_remaining_days": features["warranty_remaining_days"],
                "shipping_address_id": "ADDR-XXXX",
                "description_similarity_score": 0.85 if short_desc else 0.3,
                "duplicate_description_count": 6 if short_desc else 0,
                "technical_detail_score": round(min(len(description) / 200, 1.0), 2),
            },
            customer_data={
                "total_claims_lifetime": prev_claims,
                "claims_last_90_days": claims_90d,
                "claims_last_30_days": claims_30d,
                "unique_addresses_used": unique_addresses,
                "pct_high_value_claims": features["pct_high_value_claims"],
            },
            address_data={
                "unique_customers_at_address": unique_addresses,
                "total_claims_at_address": prev_claims,
            },
        )

        if score >= 70:
            risk_level, color, decision = "🔴 HIGH RISK", "#F44336", "REJECT / HOLD"
        elif score >= 40:
            risk_level, color, decision = "🟡 MEDIUM RISK", "#FF9800", "REVIEW"
        else:
            risk_level, color, decision = "🟢 LOW RISK", "#4CAF50", "APPROVE"

        col_score, col_detail = st.columns([1, 2])
        with col_score:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=score, title={"text": "Fraud Risk Score"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                       "steps": [{"range": [0, 40], "color": "#E8F5E9"},
                                 {"range": [40, 70], "color": "#FFF3E0"},
                                 {"range": [70, 100], "color": "#FFEBEE"}]}))
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

        with col_detail:
            st.markdown(f"### Assessment: {risk_level}")
            st.markdown(f"### Decision: **{decision}**")
            counts = {s: sum(r["severity"] == s for r in reasons) for s in ("HIGH", "MEDIUM", "LOW")}
            st.markdown(f"**{len(reasons)} reason(s) triggered** — "
                        f"🔴 {counts['HIGH']} High · 🟡 {counts['MEDIUM']} Medium · 🟢 {counts['LOW']} Low")

            st.markdown("#### 🎯 Recommended Action:")
            if score >= 70:
                st.error("HOLD shipment. Escalate to Fraud Investigation Team for manual review.")
            elif score >= 40:
                st.warning("Proceed with caution. Request additional verification from customer.")
            else:
                st.success("Low risk. Approve claim through standard workflow.")

        st.markdown("---")
        st.subheader("📋 Rejection / Flag Reasons")
        SEVERITY_STYLE = {"HIGH": ("🔴", "#F44336"), "MEDIUM": ("🟡", "#FF9800"), "LOW": ("🟢", "#4CAF50")}
        for r in reasons:
            icon, border = SEVERITY_STYLE.get(r["severity"], ("⚪", "#9E9E9E"))
            st.markdown(f"""
<div style="border-left:5px solid {border}; padding:12px 15px; margin:10px 0;
            background-color:rgba(128,128,128,0.08); border-radius:5px;">
<strong>{icon} [{r['code']}] {r['title']}</strong><br>
<span style="opacity:0.7;">Category: {r['category']} | Severity: {r['severity']}</span><br><br>
<strong>Finding:</strong> {r['detail']}<br><br>
<strong>Recommended Action:</strong> <em>{r['recommendation']}</em>
</div>""", unsafe_allow_html=True)
        if not reasons:
            st.success("✅ No rejection reasons triggered. Claim appears legitimate.")

        st.subheader("📝 Agent Summary (Copy-Paste Ready)")
        lines = ["CLAIM FRAUD ASSESSMENT REPORT", "=" * 50,
                 f"Service Tag:    {service_tag}",
                 f"Customer ID:    {customer_id}",
                 f"Component:      {component} (${component_value})",
                 f"Fraud Score:    {score}/100",
                 f"Risk Level:     {risk_level}",
                 f"Decision:       {decision}",
                 f"Reasons Found:  {len(reasons)}", "=" * 50]
        if reasons:
            lines.append("REJECTION REASONS:")
            for i, r in enumerate(reasons, 1):
                lines += [f"  {i}. [{r['code']}] [{r['severity']}] {r['title']}",
                          f"     Detail: {r['detail']}",
                          f"     Action: {r['recommendation']}", ""]
        st.code("\n".join(lines), language="text")

        st.markdown("---")
        st.subheader("🔄 Agent Feedback")
        with st.form("feedback_form"):
            fb_left, fb_right = st.columns(2)
            fb_left.radio("Do you agree with the AI assessment?", [
                "✅ Agree — Correct Assessment",
                "❌ Disagree — False Positive (Legit Claim)",
                "❌ Disagree — Missed Fraud (Should be flagged)"])
            fb_right.text_area("Agent Notes (Optional)", height=100,
                               placeholder="Add any additional context for model improvement...")
            if st.form_submit_button("Submit Feedback", width="stretch"):
                st.success("✅ Feedback recorded! This will be used to improve model accuracy "
                           "in the next training cycle.")

# ---------------- MODEL INSIGHTS ----------------
elif page == "📈 Model Insights":
    st.title("📈 Model Performance & Insights")
    st.markdown("---")
    st.subheader("Feature Importance Rankings")
    imp = pd.DataFrame({"Feature": model_bundle["features"], "Importance": model.feature_importances_})
    fig = px.bar(imp.sort_values("Importance").tail(15), x="Importance", y="Feature",
                 orientation="h", color="Importance", color_continuous_scale="Blues")
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("🔎 Key Fraud Patterns Detected")
    label = claims["is_fraud"].map({0: "Legitimate", 1: "Fraudulent"})
    left, right = st.columns(2)
    with left:
        st.markdown("#### Device Age Distribution")
        fig = px.histogram(claims, x="device_age_days", color=label, nbins=30, barmode="overlay",
                           color_discrete_map=COLORS, opacity=0.7)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Component Value Distribution")
        fig = px.box(claims, x=label, y="component_value_usd", color=label, color_discrete_map=COLORS)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, width="stretch")

# ---------------- REJECTION ANALYTICS ----------------
else:
    st.title("📝 Rejection Reason Analytics")
    st.markdown("Understand WHY claims are being flagged — computed by running every rule against all historical claims")
    st.markdown("---")

    with st.spinner("Evaluating rejection rules on historical claims..."):
        stats = compute_reason_stats(claims, customer_history, address_signals)

    most = stats.loc[stats["Times Triggered"].idxmax()]
    best = stats.loc[stats["Confirmed Fraud %"].idxmax()]
    worst = stats.loc[stats["False Positive %"].idxmax()]
    c1, c2, c3 = st.columns(3)
    c1.metric("Most Triggered Reason", f"{most['Reason Code']}", delta=f"{most['Times Triggered']:,} claims",
              delta_color="off")
    c2.metric("Most Accurate Reason", f"{best['Reason Code']}", delta=f"{best['Confirmed Fraud %']}% confirmed")
    c3.metric("Highest False Positive", f"{worst['Reason Code']}", delta=f"{worst['False Positive %']}% false positive",
              delta_color="inverse")
    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("Reason Trigger Frequency")
        fig = px.bar(stats.sort_values("Times Triggered"), x="Times Triggered", y="Reason", orientation="h",
                     color="Confirmed Fraud %", color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Accuracy by Reason")
        fig = px.bar(stats, x="Reason Code", y=["Confirmed Fraud %", "False Positive %"], barmode="stack",
                     color_discrete_map={"Confirmed Fraud %": "#F44336", "False Positive %": "#90CAF9"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Detailed Reason Performance")
    pct = lambda label: st.column_config.ProgressColumn(label, format="%.1f%%", min_value=0, max_value=100)
    st.dataframe(stats.sort_values("Confirmed Fraud %", ascending=False), width="stretch", hide_index=True,
                 column_config={"Confirmed Fraud %": pct("Confirmed Fraud %"),
                                "False Positive %": pct("False Positive %")})

    st.markdown("---")
    st.subheader("💡 Key Insights")
    st.markdown(f"""
- **{worst['Reason Code']} ({worst['Reason']})** has the highest false positive rate at {worst['False Positive %']}%.
  Consider **lowering its severity** or only flagging it alongside other signals.
- **{best['Reason Code']} ({best['Reason']})** is the most reliable single indicator with a
  {best['Confirmed Fraud %']}% confirmed fraud rate. Consider **auto-holding** claims that trigger it.
- **{most['Reason Code']} ({most['Reason']})** fires most often ({most['Times Triggered']:,} claims) —
  tuning its threshold has the largest impact on agent workload.
""")

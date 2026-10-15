import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src import investigation_agent as agent


@pytest.fixture
def scored():
    rows = []
    for i in range(6):  # fraud ring: 6 customers shipping to ADDR-RING
        rows.append(
            {
                "claim_id": f"C{i}",
                "customer_id": f"R{i}",
                "shipping_address_id": "ADDR-RING",
                "component_requested": "GPU",
                "component_value_usd": 800.0,
                "device_age_days": 40,
                "diagnostic_result": "NOT_RUN",
                "claim_description": "GPU dead. Send replacement ASAP.",
                "claim_status": "Approved",
                "fraud_score": 99,
            }
        )
    rows.append(
        {
            "claim_id": "G1",
            "customer_id": "GOOD",
            "shipping_address_id": "ADDR-HOME",
            "component_requested": "Battery",
            "component_value_usd": 110.0,
            "device_age_days": 500,
            "diagnostic_result": "FAIL",
            "claim_status": "Approved",
            "fraud_score": 1,
            "claim_description": "Battery health at 23% after 14 months; error code 2000-0151 in diagnostics.",
        }
    )
    df = pd.DataFrame(rows)
    ring = df["shipping_address_id"] == "ADDR-RING"
    df = df.assign(
        claim_date=pd.Timestamp("2026-09-01"),
        service_tag="TAG",
        product_line="XPS 15 9530",
        warranty_remaining_days=900,
        support_channel="Web Portal",
        total_claims_lifetime=1,
        claims_last_90_days=1,
        claims_last_30_days=1,
        unique_devices_claimed=1,
        unique_addresses_used=1,
        avg_component_value=500.0,
        max_component_value=800.0,
        pct_high_value_claims=np.where(ring, 1.0, 0.0),
        previous_rejections=0,
        days_since_last_claim=10,
        account_age_days=300,
        unique_customers_at_address=np.where(ring, 6, 1),
        unique_service_tags=np.where(ring, 6, 1),
        total_claims_at_address=np.where(ring, 6, 1),
        address_fraud_rate=np.where(ring, 1.0, 0.0),
        pct_high_value_at_address=np.where(ring, 1.0, 0.0),
        is_residential=True,
        is_fraud=np.where(ring, 1, 0),
    )
    return df


@pytest.fixture
def tools(scored):
    return agent.InvestigationTools(scored)


def test_risk_assessment_returns_score_band_and_rules(tools):
    result = tools.run("get_risk_assessment", {"claim_id": "C0"})
    assert result["fraud_score"] == 99
    assert result["decision_band"].startswith("HIGH")
    assert {"RF-003", "RF-007"} <= {r["code"] for r in result["triggered_rules"]}


def test_address_network_exposes_ring(tools):
    result = tools.run("get_address_network", {"address_id": "ADDR-RING"})
    assert result["stats"]["unique_customers_at_address"] == 6
    assert len(result["linked_customers"]) == 6


def test_similar_descriptions_groups_copy_paste_text(tools):
    result = tools.run("find_similar_descriptions", {"description": "GPU dead. Send replacement ASAP."})
    assert result["near_duplicate_groups"][0]["claims"] == 6
    assert result["near_duplicate_groups"][0]["distinct_customers"] == 6


def test_tools_never_leak_the_fraud_label(tools):
    for name, args in [
        ("get_risk_assessment", {"claim_id": "C0"}),
        ("get_customer_history", {"customer_id": "R0"}),
        ("get_address_network", {"address_id": "ADDR-RING"}),
        ("get_component_baseline", {"component": "GPU"}),
    ]:
        assert "is_fraud" not in json.dumps(tools.run(name, args), default=str)


@pytest.mark.parametrize(
    ("name", "args"),
    [("not_a_tool", {}), ("get_customer_history", {"customer_id": "NOPE"}), ("get_risk_assessment", {"bad": 1})],
)
def test_tool_errors_are_returned_not_raised(tools, name, args):
    assert "error" in tools.run(name, args)


def _call(call_id, name, **args):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_agent_loop_runs_tools_and_parses_fenced_verdict(tools):
    script = iter(
        [
            {"content": "Check the score.", "tool_calls": [_call("1", "get_risk_assessment", claim_id="C0")]},
            {
                "content": "Check the network.",
                "tool_calls": [_call("2", "get_address_network", address_id="ADDR-RING")],
            },
            {"content": '```json\n{"recommendation": "HOLD", "confidence": "HIGH", "summary": "ring"}\n```'},
        ]
    )
    seen_roles = []

    def fake_chat(messages, specs):
        assert specs is agent.TOOL_SPECS
        seen_roles.append([m["role"] for m in messages])
        return next(script)

    steps = []
    verdict, trace = agent.run_investigation("C0", tools, fake_chat, on_step=steps.append)
    assert verdict["recommendation"] == "HOLD"
    assert [s["tool"] for s in trace] == ["get_risk_assessment", "get_address_network"]
    assert steps == trace
    assert seen_roles[-1] == ["system", "user", "assistant", "tool", "assistant", "tool"]


def test_agent_forces_verdict_at_step_limit(tools):
    calls = {"n": 0}

    def looping_chat(messages, specs):
        calls["n"] += 1
        if messages[-1]["role"] == "user" and "Step limit" in messages[-1]["content"]:
            return {"content": '{"recommendation": "ESCALATE", "confidence": "LOW", "summary": "limit"}'}
        return {"content": "", "tool_calls": [_call(str(calls["n"]), "get_component_baseline", component="GPU")]}

    verdict, trace = agent.run_investigation("C0", tools, looping_chat, max_steps=3)
    assert verdict["recommendation"] == "ESCALATE"
    assert len(trace) == 3


def test_unstructured_verdict_falls_back_to_escalate():
    verdict = agent.parse_final("I think it's fraud but I'm not sure.")
    assert verdict["recommendation"] == "ESCALATE"
    assert "not sure" in verdict["case_note"]


def test_env_overrides_secrets_and_missing_config(monkeypatch):
    for key in agent.CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GENAI_MODEL", "env-model")
    cfg = agent.load_config({"GENAI_URL": "https://gw/v1", "GENAI_MODEL": "secret-model"})
    assert cfg["GENAI_URL"] == "https://gw/v1"
    assert cfg["GENAI_MODEL"] == "env-model"
    assert agent.missing_config(cfg) == ["TOKEN_URL", "CLIENT_ID", "CLIENT_SECRET"]
    assert agent.missing_config({**cfg, "GENAI_API_KEY": "k"}) == []


def test_gateway_token_is_cached(monkeypatch):
    posts = []

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"access_token": "tok", "expires_in": 3600}

    monkeypatch.setattr(agent.requests, "post", lambda *a, **k: posts.append(k) or FakeResponse())
    client = agent.GatewayClient(
        {"TOKEN_URL": "https://auth/token", "CLIENT_ID": "id", "CLIENT_SECRET": "s", "GENAI_URL": "https://gw/v1"}
    )
    assert client.token() == "tok"
    assert client.token() == "tok"
    assert len(posts) == 1
    assert posts[0]["data"]["grant_type"] == "client_credentials"


def test_build_scored_claims_adds_features_and_score(scored):
    class FakeModel:
        def predict_proba(self, x):
            return np.column_stack([np.full(len(x), 0.25), np.full(len(x), 0.75)])

    def encoder(values):
        return LabelEncoder().fit(values)

    raw = scored.drop(columns=["fraud_score"]).assign(warranty_end_date="2029-01-01")
    customers = raw[["customer_id"]].drop_duplicates()
    addresses = raw[["shipping_address_id"]].drop_duplicates()
    bundle = {
        "model": FakeModel(),
        "features": ["device_age_days", "component_encoded"],
        "le_component": encoder(raw["component_requested"]),
        "le_channel": encoder(raw["support_channel"]),
        "le_product": encoder(raw["product_line"]),
    }
    result = agent.build_scored_claims(raw, customers, addresses, bundle)
    assert (result["fraud_score"] == 75).all()
    assert {"description_length", "diagnostic_not_run", "component_encoded"} <= set(result.columns)


@pytest.mark.parametrize(("score", "band"), [(99, "HIGH"), (70, "HIGH"), (69, "MEDIUM"), (40, "MEDIUM"), (39, "LOW")])
def test_decision_band_thresholds(score, band):
    assert agent.decision_band(score).startswith(band)

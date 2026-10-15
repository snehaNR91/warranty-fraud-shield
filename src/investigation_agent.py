"""Agentic fraud investigation for flagged warranty claims.

An LLM plans which evidence to gather, calls read-only investigation tools over the claims data, and drafts a
recommendation, case note and customer message for a human to approve. Nothing is held or sent automatically.

The LLM is reached through an OpenAI-compatible ``/chat/completions`` endpoint (e.g. the Dell AIA gateway),
authenticated with OAuth2 client credentials or a static bearer token.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import requests

from src.reason_generator import generate_rejection_reasons

try:  # trust the OS certificate store (e.g. Dell's internal root CA) for HTTPS calls to the gateway
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover - optional dependency
    pass

logger = logging.getLogger(__name__)

Message = dict[str, Any]
ChatFn = Callable[[list[Message], list[dict[str, Any]]], Message]

CONFIG_KEYS = (
    "GENAI_URL",
    "GENAI_MODEL",
    "TOKEN_URL",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "GENAI_SCOPE",
    "GENAI_API_KEY",
    "GENAI_CA_BUNDLE",
)
MAX_STEPS = 8
HIGH_RISK, MEDIUM_RISK = 70, 40
HTTP_TIMEOUT_S = 30
CHAT_TIMEOUT_S = 120
TOKEN_REFRESH_MARGIN_S = 60
SIMILARITY_THRESHOLD = 0.75

SYSTEM_PROMPT = """You are a senior warranty fraud investigator at Dell Technologies.
A claim has been flagged for review. Investigate it before any replacement part ships.

How to work:
- Gather evidence with the tools. Start with get_risk_assessment, then follow the leads it gives you
  (customer history, shipping-address network, similar claim descriptions). Use 3-6 tool calls.
- Only state facts returned by tools. Never invent numbers, names or history.
- Weigh both sides: note evidence that the claim may be genuine as well as evidence of fraud.

Finish with ONLY a JSON object (no markdown fences) with these keys:
  "recommendation": "HOLD" | "RELEASE" | "ESCALATE",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "summary": one-sentence conclusion,
  "key_evidence": list of 3-6 short evidence bullets, each citing a number from the tools,
  "counter_evidence": list of 0-3 bullets that point towards the claim being genuine,
  "next_steps": list of 2-4 concrete actions for the fraud team,
  "case_note": a concise internal case note (4-6 sentences) for the CRM,
  "customer_message": a polite, non-accusatory message to the customer requesting the needed verification.
"""


# ─────────────────────────── configuration & auth ───────────────────────────
def load_config(secrets: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Load gateway settings from Streamlit secrets, overridden by environment variables when set.

    The environment is read first because Streamlit copies root-level secrets into ``os.environ`` on access.

    Args:
        secrets: Mapping such as ``st.secrets``; may be ``None`` or raise if no secrets file exists.

    Returns:
        Settings keyed by :data:`CONFIG_KEYS`; missing values are empty strings.
    """
    env = {k: os.environ.get(k, "").strip() for k in CONFIG_KEYS}
    cfg: dict[str, str] = {}
    for key in CONFIG_KEYS:
        try:
            cfg[key] = str(secrets[key]).strip() if secrets is not None and key in secrets else ""
        except Exception:  # noqa: BLE001 - Streamlit raises if no secrets file is present
            cfg[key] = ""
        cfg[key] = env[key] or cfg[key]
    return cfg


def missing_config(cfg: Mapping[str, str]) -> list[str]:
    """Return the names of required settings that are not configured."""
    missing = [k for k in ("GENAI_URL", "GENAI_MODEL") if not cfg.get(k)]
    if not cfg.get("GENAI_API_KEY"):
        missing += [k for k in ("TOKEN_URL", "CLIENT_ID", "CLIENT_SECRET") if not cfg.get(k)]
    return missing


class GatewayClient:
    """Minimal OpenAI-compatible client with OAuth2 client-credentials token caching."""

    def __init__(self, cfg: Mapping[str, str]) -> None:
        self.cfg = cfg
        self.verify: str | bool = cfg.get("GENAI_CA_BUNDLE") or True
        self._token: str | None = None
        self._expires = 0.0

    @property
    def base_url(self) -> str:
        """Gateway base URL without a trailing slash."""
        return self.cfg["GENAI_URL"].rstrip("/")

    def token(self) -> str:
        """Return a valid bearer token, requesting a new one when the cached token is about to expire."""
        if self.cfg.get("GENAI_API_KEY"):
            return self.cfg["GENAI_API_KEY"]
        if self._token and time.time() < self._expires - TOKEN_REFRESH_MARGIN_S:
            return self._token
        data = {
            "grant_type": "client_credentials",
            "client_id": self.cfg["CLIENT_ID"],
            "client_secret": self.cfg["CLIENT_SECRET"],
        }
        if self.cfg.get("GENAI_SCOPE"):
            data["scope"] = self.cfg["GENAI_SCOPE"]
        response = requests.post(self.cfg["TOKEN_URL"], data=data, timeout=HTTP_TIMEOUT_S, verify=self.verify)
        response.raise_for_status()
        body = response.json()
        self._token = body["access_token"]
        self._expires = time.time() + float(body.get("expires_in", 3600))
        logger.info("Obtained gateway token (expires in %ss)", body.get("expires_in"))
        return self._token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}", "Content-Type": "application/json"}

    def list_models(self) -> list[str]:
        """Return the sorted model names available on the gateway."""
        response = requests.get(
            f"{self.base_url}/models", headers=self._headers(), timeout=HTTP_TIMEOUT_S, verify=self.verify
        )
        response.raise_for_status()
        body = response.json()
        items = body.get("data", []) if isinstance(body, dict) else body
        return sorted(m.get("id") or m.get("model", "") for m in items)

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        """Send one chat-completions request with tool definitions and return the assistant message."""
        payload = {
            "model": self.cfg["GENAI_MODEL"],
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=CHAT_TIMEOUT_S,
            verify=self.verify,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]


# ─────────────────────────── data preparation ───────────────────────────
def build_scored_claims(
    claims: pd.DataFrame, customer_history: pd.DataFrame, address_signals: pd.DataFrame, bundle: dict[str, Any]
) -> pd.DataFrame:
    """Join the three tables, compute model features and score every historical claim (0-100).

    Args:
        claims: Claim-level records.
        customer_history: Per-customer behaviour features.
        address_signals: Per-address network features.
        bundle: Trained model bundle (model, feature list and label encoders) from ``src.model``.

    Returns:
        The joined frame with model features and an integer ``fraud_score`` column.
    """
    df = claims.merge(customer_history, on="customer_id", how="left").merge(
        address_signals, on="shipping_address_id", how="left"
    )
    df["description_length"] = df["claim_description"].str.len()
    df["diagnostic_not_run"] = (df["diagnostic_result"] == "NOT_RUN").astype(int)
    df["diagnostic_passed"] = (df["diagnostic_result"] == "PASS").astype(int)
    df["warranty_remaining_days"] = (pd.to_datetime(df["warranty_end_date"]) - pd.to_datetime(df["claim_date"])).dt.days
    for col, key in (
        ("component_requested", "le_component"),
        ("support_channel", "le_channel"),
        ("product_line", "le_product"),
    ):
        classes = {c: i for i, c in enumerate(bundle[key].classes_)}
        df[key.replace("le_", "") + "_encoded"] = df[col].map(classes).fillna(0).astype(int)
    probabilities = bundle["model"].predict_proba(df[bundle["features"]].fillna(0))[:, 1]
    df["fraud_score"] = (probabilities * 100).astype(int)
    return df


def decision_band(score: int) -> str:
    """Map a 0-100 fraud score to the decision band used across the app."""
    if score >= HIGH_RISK:
        return "HIGH (REJECT / HOLD)"
    return "MEDIUM (REVIEW)" if score >= MEDIUM_RISK else "LOW (APPROVE)"


# ─────────────────────────── investigation tools ───────────────────────────
def _tool(name: str, description: str, param: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {param: {"type": "string"}}, "required": [param]},
        },
    }


TOOL_SPECS = [
    _tool(
        "get_risk_assessment",
        "ML fraud score (0-100), decision band, claim details and every triggered "
        "rejection rule for the claim under investigation.",
        "claim_id",
    ),
    _tool(
        "get_customer_history",
        "Customer profile (claim counts, addresses used, high-value share, rejections) "
        "and their most recent claims.",
        "customer_id",
    ),
    _tool(
        "get_address_network",
        "Who else ships to a shipping address: address statistics, linked customer "
        "accounts and how many other addresses those accounts use. Use to detect fraud rings / drop "
        "addresses.",
        "address_id",
    ),
    _tool(
        "find_similar_descriptions",
        "Find other claims whose description is identical or near-identical to "
        "the given text (copy-paste / scripted submissions).",
        "description",
    ),
    _tool(
        "get_component_baseline",
        "Typical value, device age, diagnostic outcomes and channels for a component "
        "across all claims, to judge whether this claim is unusual.",
        "component",
    ),
]
TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOL_SPECS)


def _round(value: float, digits: int = 1) -> float:
    return round(float(value), digits)


class InvestigationTools:
    """Read-only tools over the scored claims. The fraud label (``is_fraud``) is never exposed to the agent."""

    def __init__(self, scored: pd.DataFrame) -> None:
        self.df = scored
        self.by_id = scored.set_index("claim_id")

    def run(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name; errors are returned to the model rather than raised."""
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, name)(**args)
        except Exception as exc:  # noqa: BLE001 - surface any tool failure to the model
            logger.warning("Tool %s failed: %s", name, exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    def get_risk_assessment(self, claim_id: str) -> dict[str, Any]:
        """Return the ML score, decision band, key claim fields and triggered rejection rules."""
        if claim_id not in self.by_id.index:
            return {"error": f"claim {claim_id} not found"}
        claim = self.by_id.loc[claim_id]
        record = claim.to_dict()
        reasons = generate_rejection_reasons(
            claim_data={
                **record,
                "claim_id": claim_id,
                "description_similarity_score": 0.3,
                "technical_detail_score": _round(min(len(claim["claim_description"]) / 200, 1.0), 2),
            },
            customer_data=record,
            address_data=record,
        )
        fields = (
            "claim_date",
            "service_tag",
            "customer_id",
            "shipping_address_id",
            "product_line",
            "component_requested",
            "component_value_usd",
            "device_age_days",
            "warranty_remaining_days",
            "diagnostic_result",
            "support_channel",
            "claim_description",
        )
        return {
            "claim_id": claim_id,
            "fraud_score": int(claim["fraud_score"]),
            "decision_band": decision_band(int(claim["fraud_score"])),
            "claim": {k: claim[k].item() if hasattr(claim[k], "item") else claim[k] for k in fields},
            "triggered_rules": [
                {"code": r["code"], "severity": r["severity"], "title": r["title"], "finding": r["detail"]}
                for r in reasons
            ],
        }

    def get_customer_history(self, customer_id: str) -> dict[str, Any]:
        """Return the customer's behaviour profile, status counts and most recent claims."""
        rows = self.df[self.df["customer_id"] == customer_id]
        if rows.empty:
            return {"error": f"customer {customer_id} not found"}
        profile = rows.iloc[0]
        recent = rows.sort_values("claim_date", ascending=False).head(8)
        profile_fields = (
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
        )
        return {
            "customer_id": customer_id,
            "profile": {k: _round(profile[k], 2) for k in profile_fields},
            "claim_status_counts": rows["claim_status"].value_counts().to_dict(),
            "recent_claims": [
                {
                    "claim_id": r.claim_id,
                    "date": str(r.claim_date)[:10],
                    "component": r.component_requested,
                    "value_usd": _round(r.component_value_usd, 0),
                    "address": r.shipping_address_id,
                    "diagnostic": r.diagnostic_result,
                    "status": r.claim_status,
                }
                for r in recent.itertuples()
            ],
        }

    def get_address_network(self, address_id: str) -> dict[str, Any]:
        """Return address statistics and the customer accounts linked to the address."""
        rows = self.df[self.df["shipping_address_id"] == address_id]
        if rows.empty:
            return {"error": f"address {address_id} not found"}
        addr = rows.iloc[0]
        per_customer = rows.groupby("customer_id").agg(claims_here=("claim_id", "size"))
        per_customer["addresses_used_overall"] = (
            self.df[self.df["customer_id"].isin(per_customer.index)]
            .groupby("customer_id")["shipping_address_id"]
            .nunique()
        )
        top = per_customer.sort_values("claims_here", ascending=False).head(8)
        return {
            "address_id": address_id,
            "stats": {
                "total_claims_at_address": int(addr["total_claims_at_address"]),
                "unique_customers_at_address": int(addr["unique_customers_at_address"]),
                "unique_service_tags": int(addr["unique_service_tags"]),
                "historical_fraud_rate": _round(addr["address_fraud_rate"], 2),
                "pct_high_value_at_address": _round(addr["pct_high_value_at_address"], 2),
                "is_residential": bool(addr["is_residential"]),
                "total_value_shipped_usd": _round(rows["component_value_usd"].sum(), 0),
            },
            "components_shipped": rows["component_requested"].value_counts().head(5).to_dict(),
            "linked_customers": [
                {
                    "customer_id": cid,
                    "claims_here": int(r.claims_here),
                    "addresses_used_overall": int(r.addresses_used_overall),
                }
                for cid, r in top.iterrows()
            ],
        }

    def find_similar_descriptions(self, description: str) -> dict[str, Any]:
        """Return groups of claims whose description closely matches ``description``."""
        text = description.strip().lower()
        unique = self.df["claim_description"].dropna().unique()
        ranked = sorted(((difflib.SequenceMatcher(None, text, d.lower()).ratio(), d) for d in unique), reverse=True)
        matches = []
        for ratio, desc in ranked[:5]:
            if ratio < SIMILARITY_THRESHOLD:
                break
            rows = self.df[self.df["claim_description"] == desc]
            matches.append(
                {
                    "description": desc,
                    "similarity": _round(ratio, 2),
                    "claims": len(rows),
                    "distinct_customers": int(rows["customer_id"].nunique()),
                    "distinct_addresses": int(rows["shipping_address_id"].nunique()),
                    "status_counts": rows["claim_status"].value_counts().to_dict(),
                }
            )
        note = (
            f"{sum(m['claims'] for m in matches)} claims share near-identical wording"
            if matches
            else "No near-duplicates found"
        )
        return {"query": description, "near_duplicate_groups": matches, "note": note}

    def get_component_baseline(self, component: str) -> dict[str, Any]:
        """Return typical value, device age, diagnostic and channel mix for a component."""
        rows = self.df[self.df["component_requested"].str.lower() == component.lower()]
        if rows.empty:
            return {"error": f"component {component} not found"}
        return {
            "component": component,
            "claims": len(rows),
            "value_usd": {
                "median": _round(rows["component_value_usd"].median(), 0),
                "p90": _round(rows["component_value_usd"].quantile(0.9), 0),
            },
            "device_age_days": {
                "median": _round(rows["device_age_days"].median(), 0),
                "p10": _round(rows["device_age_days"].quantile(0.1), 0),
            },
            "diagnostic_mix": rows["diagnostic_result"].value_counts(normalize=True).round(2).to_dict(),
            "channel_mix": rows["support_channel"].value_counts(normalize=True).round(2).to_dict(),
        }


# ─────────────────────────── agent loop ───────────────────────────
def parse_final(text: str | None) -> dict[str, Any]:
    """Parse the agent's final JSON verdict, tolerating markdown fences; falls back to ESCALATE."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        logger.warning("Agent returned an unstructured verdict")
        return {
            "recommendation": "ESCALATE",
            "confidence": "LOW",
            "summary": "The agent did not return a structured verdict; raw output shown below.",
            "case_note": text,
        }


def run_investigation(
    claim_id: str,
    tools: InvestigationTools,
    chat_fn: ChatFn,
    on_step: Callable[[dict[str, Any]], None] | None = None,
    max_steps: int = MAX_STEPS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the plan → act (tool call) → observe loop for one claim.

    Args:
        claim_id: Claim to investigate.
        tools: Investigation tools the agent may call.
        chat_fn: Function sending messages + tool specs to the LLM and returning the assistant message.
        on_step: Optional callback invoked after each tool call (used for live UI updates).
        max_steps: Maximum number of LLM turns before a verdict is forced.

    Returns:
        The parsed verdict and the list of executed steps (thought, tool, args, result).
    """
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Investigate warranty claim {claim_id}. " "Should the replacement part be held or released?",
        },
    ]
    trace: list[dict[str, Any]] = []
    for _ in range(max_steps):
        msg = chat_fn(messages, TOOL_SPECS)
        calls = msg.get("tool_calls") or []
        messages.append(
            {"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})}
        )
        if not calls:
            logger.info("Investigation of %s finished after %d tool calls", claim_id, len(trace))
            return parse_final(msg.get("content")), trace
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = tools.run(name, args)
            step = {"thought": msg.get("content") or "", "tool": name, "args": args, "result": result}
            trace.append(step)
            logger.info("Agent step %d: %s(%s)", len(trace), name, args)
            if on_step:
                on_step(step)
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", name), "content": json.dumps(result, default=str)}
            )
    messages.append({"role": "user", "content": "Step limit reached. Give your final JSON verdict now."})
    return parse_final(chat_fn(messages, TOOL_SPECS).get("content")), trace

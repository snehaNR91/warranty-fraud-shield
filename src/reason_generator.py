# reason_generator.py

"""Rule engine: evaluates a claim against the 12 coded rejection rules defined in ``src.reason_engine``."""

from collections.abc import Mapping
from typing import Any

from src.reason_engine import REJECTION_REASONS


def generate_rejection_reasons(
    claim_data: Mapping[str, Any], customer_data: Mapping[str, Any], address_data: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Analyze a claim against all fraud rules and return applicable rejection reasons ranked by severity.

    Args:
        claim_data: Claim fields (service tag, component, value, device age, diagnostics, description, ...).
        customer_data: Customer history (claim counts, addresses used, share of high-value claims).
        address_data: Shipping-address signals (customers and claims at the address).

    Returns:
        Triggered reasons (code, category, severity, title, detail, recommendation), HIGH severity first.
    """
    triggered_reasons = []

    # --- Rule 1: Claim frequency (30 days) ---
    if customer_data.get("claims_last_30_days", 0) >= 2:
        reason = REJECTION_REASONS["HIGH_CLAIM_FREQUENCY_30D"].copy()
        reason["detail"] = reason["template"].format(
            count=customer_data["claims_last_30_days"],
            threshold=1,
            avg=1.5,
        )
        triggered_reasons.append(reason)

    # --- Rule 2: Claim frequency (90 days) ---
    if customer_data.get("claims_last_90_days", 0) >= 3:
        reason = REJECTION_REASONS["HIGH_CLAIM_FREQUENCY_90D"].copy()
        reason["detail"] = reason["template"].format(
            count=customer_data["claims_last_90_days"],
            multiplier=round(customer_data["claims_last_90_days"] / 1.2, 1),
        )
        triggered_reasons.append(reason)

    # --- Rule 3: Diagnostics not run ---
    if claim_data.get("diagnostic_result") == "NOT_RUN":
        reason = REJECTION_REASONS["DIAGNOSTICS_NOT_RUN"].copy()
        reason["detail"] = reason["template"].format(
            service_tag=claim_data.get("service_tag", "N/A"),
            component=claim_data.get("component_requested", "N/A"),
        )
        triggered_reasons.append(reason)

    # --- Rule 4: Diagnostics passed but claiming failure ---
    if claim_data.get("diagnostic_result") == "PASS":
        reason = REJECTION_REASONS["DIAGNOSTICS_PASSED_BUT_CLAIMING"].copy()
        reason["detail"] = reason["template"].format(
            service_tag=claim_data.get("service_tag", "N/A"),
            component=claim_data.get("component_requested", "N/A"),
        )
        triggered_reasons.append(reason)

    # --- Rule 5: High-value component pattern ---
    if customer_data.get("pct_high_value_claims", 0) >= 0.7:
        reason = REJECTION_REASONS["HIGH_VALUE_COMPONENT_PATTERN"].copy()
        reason["detail"] = reason["template"].format(
            threshold=300,
            pct=int(customer_data["pct_high_value_claims"] * 100),
            component=claim_data.get("component_requested", "N/A"),
            value=claim_data.get("component_value_usd", 0),
        )
        triggered_reasons.append(reason)

    # --- Rule 6: Component value vs device age ---
    if claim_data.get("component_value_usd", 0) > 300 and claim_data.get("device_age_days", 999) < 90:
        reason = REJECTION_REASONS["COMPONENT_VALUE_EXCEEDS_DEVICE"].copy()
        reason["detail"] = reason["template"].format(
            component=claim_data.get("component_requested", "N/A"),
            value=claim_data.get("component_value_usd", 0),
            age=claim_data.get("device_age_days", 0),
            product=claim_data.get("product_line", "N/A"),
        )
        triggered_reasons.append(reason)

    # --- Rule 7: Multiple customers at same address ---
    if address_data.get("unique_customers_at_address", 1) >= 3:
        reason = REJECTION_REASONS["MULTIPLE_CUSTOMERS_SAME_ADDRESS"].copy()
        reason["detail"] = reason["template"].format(
            num_customers=address_data["unique_customers_at_address"],
            address_id=claim_data.get("shipping_address_id", "N/A"),
            total_claims=address_data.get("total_claims_at_address", 0),
        )
        triggered_reasons.append(reason)

    # --- Rule 8: Multiple addresses for single customer ---
    if customer_data.get("unique_addresses_used", 1) >= 3:
        reason = REJECTION_REASONS["MULTIPLE_ADDRESSES_SINGLE_CUSTOMER"].copy()
        reason["detail"] = reason["template"].format(
            customer_id=claim_data.get("customer_id", "N/A"),
            num_addresses=customer_data["unique_addresses_used"],
            total_claims=customer_data.get("total_claims_lifetime", 0),
        )
        triggered_reasons.append(reason)

    # --- Rule 9: Templated description ---
    if claim_data.get("description_similarity_score", 0) > 0.8:
        reason = REJECTION_REASONS["TEMPLATED_DESCRIPTION"].copy()
        reason["detail"] = reason["template"].format(
            similarity=int(claim_data.get("description_similarity_score", 0) * 100),
            match_count=claim_data.get("duplicate_description_count", 0),
        )
        triggered_reasons.append(reason)

    # --- Rule 10: Vague description ---
    desc_len = len(claim_data.get("claim_description", ""))
    if desc_len < 50:
        reason = REJECTION_REASONS["VAGUE_DESCRIPTION"].copy()
        reason["detail"] = reason["template"].format(
            length=desc_len,
            tech_score=claim_data.get("technical_detail_score", 0.1),
        )
        triggered_reasons.append(reason)

    # --- Rule 11: New device, high value claim ---
    if claim_data.get("device_age_days", 999) < 60 and claim_data.get("component_value_usd", 0) > 300:
        reason = REJECTION_REASONS["NEW_DEVICE_HIGH_VALUE_CLAIM"].copy()
        reason["detail"] = reason["template"].format(
            age=claim_data.get("device_age_days", 0),
            component=claim_data.get("component_requested", "N/A"),
            failure_rate=0.3,
        )
        triggered_reasons.append(reason)

    # --- Rule 12: Warranty expired ---
    warranty_remaining = claim_data.get("warranty_remaining_days", 999)
    if warranty_remaining <= 0:
        reason = REJECTION_REASONS["EXPIRED_WARRANTY_ATTEMPT"].copy()
        reason["detail"] = reason["template"].format(
            service_tag=claim_data.get("service_tag", "N/A"),
            days_expired=abs(warranty_remaining),
            days_left=0,
        )
        triggered_reasons.append(reason)
    elif warranty_remaining <= 30:
        reason = REJECTION_REASONS["EXPIRED_WARRANTY_ATTEMPT"].copy()
        reason["detail"] = reason["template"].format(
            service_tag=claim_data.get("service_tag", "N/A"),
            days_expired=0,
            days_left=warranty_remaining,
        )
        triggered_reasons.append(reason)

    # Sort by severity
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    triggered_reasons.sort(key=lambda x: severity_order.get(x["severity"], 3))

    return triggered_reasons

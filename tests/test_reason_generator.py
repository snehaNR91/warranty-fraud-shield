import pytest

from src.reason_engine import REJECTION_REASONS
from src.reason_generator import generate_rejection_reasons

CLEAN_CLAIM = {
    "service_tag": "ABC1234",
    "customer_id": "CUST-1",
    "product_line": "Latitude 5540",
    "component_requested": "Battery",
    "component_value_usd": 120,
    "device_age_days": 400,
    "diagnostic_result": "FAIL",
    "claim_description": "Battery health at 23% in Dell Power Manager after 14 months; error 2000-0151 in diagnostics.",
    "warranty_remaining_days": 600,
    "shipping_address_id": "ADDR-1",
    "description_similarity_score": 0.2,
}
CLEAN_CUSTOMER = {
    "claims_last_30_days": 0,
    "claims_last_90_days": 1,
    "unique_addresses_used": 1,
    "pct_high_value_claims": 0.0,
    "total_claims_lifetime": 1,
}
CLEAN_ADDRESS = {"unique_customers_at_address": 1, "total_claims_at_address": 1}


def codes(claim=None, customer=None, address=None):
    reasons = generate_rejection_reasons(
        {**CLEAN_CLAIM, **(claim or {})}, {**CLEAN_CUSTOMER, **(customer or {})}, {**CLEAN_ADDRESS, **(address or {})}
    )
    return [r["code"] for r in reasons]


def test_catalogue_has_twelve_unique_codes():
    all_codes = [r["code"] for r in REJECTION_REASONS.values()]
    assert len(all_codes) == 12
    assert len(set(all_codes)) == 12


def test_clean_claim_triggers_nothing():
    assert codes() == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"customer": {"claims_last_30_days": 2}}, "RF-001"),
        ({"customer": {"claims_last_90_days": 3}}, "RF-002"),
        ({"claim": {"diagnostic_result": "NOT_RUN"}}, "RF-003"),
        ({"claim": {"diagnostic_result": "PASS"}}, "RF-004"),
        ({"customer": {"pct_high_value_claims": 0.7}}, "RF-005"),
        ({"claim": {"component_value_usd": 500, "device_age_days": 80}}, "RF-006"),
        ({"address": {"unique_customers_at_address": 3}}, "RF-007"),
        ({"customer": {"unique_addresses_used": 3}}, "RF-008"),
        ({"claim": {"description_similarity_score": 0.85}}, "RF-009"),
        ({"claim": {"claim_description": "broken"}}, "RF-010"),
        ({"claim": {"component_value_usd": 500, "device_age_days": 30}}, "RF-011"),
        ({"claim": {"warranty_remaining_days": 10}}, "RF-012"),
    ],
)
def test_each_rule_fires_at_its_threshold(overrides, expected):
    assert expected in codes(**overrides)


def test_thresholds_are_not_triggered_just_below():
    assert (
        codes(
            customer={"claims_last_30_days": 1, "claims_last_90_days": 2, "unique_addresses_used": 2},
            address={"unique_customers_at_address": 2},
        )
        == []
    )


def test_reasons_are_sorted_high_first_and_filled_in():
    reasons = generate_rejection_reasons(
        {**CLEAN_CLAIM, "diagnostic_result": "NOT_RUN", "claim_description": "broken"},
        {**CLEAN_CUSTOMER, "claims_last_30_days": 3},
        CLEAN_ADDRESS,
    )
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    severities = [order[r["severity"]] for r in reasons]
    assert severities == sorted(severities)
    assert all("{" not in r["detail"] for r in reasons)
    assert "ABC1234" in next(r["detail"] for r in reasons if r["code"] == "RF-003")

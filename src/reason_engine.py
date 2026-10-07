# reason_engine.py

REJECTION_REASONS = {
    # --- Claim Velocity ---
    "HIGH_CLAIM_FREQUENCY_30D": {
        "code": "RF-001",
        "category": "Claim Velocity",
        "severity": "HIGH",
        "title": "Excessive claims in 30 days",
        "template": (
            "Customer has filed {count} claims in the last 30 days, "
            "which exceeds the threshold of {threshold}. "
            "Average for this product line is {avg} claims per year."
        ),
        "recommendation": (
            "Hold shipment. Request customer to provide "
            "proof of defect via photo/video before proceeding."
        ),
    },
    "HIGH_CLAIM_FREQUENCY_90D": {
        "code": "RF-002",
        "category": "Claim Velocity",
        "severity": "MEDIUM",
        "title": "Elevated claims in 90 days",
        "template": (
            "Customer has filed {count} claims in the last 90 days. "
            "This is {multiplier}x higher than the average customer."
        ),
        "recommendation": (
            "Proceed with additional verification. "
            "Request diagnostics report before shipping part."
        ),
    },

    # --- Diagnostic Mismatch ---
    "DIAGNOSTICS_NOT_RUN": {
        "code": "RF-003",
        "category": "Diagnostic Mismatch",
        "severity": "MEDIUM",
        "title": "Diagnostics not executed",
        "template": (
            "No Dell diagnostic report found for Service Tag {service_tag}. "
            "Customer is requesting a {component} replacement "
            "without running standard diagnostics."
        ),
        "recommendation": (
            "Request customer to run Dell SupportAssist diagnostics "
            "and share the validation code before approving."
        ),
    },
    "DIAGNOSTICS_PASSED_BUT_CLAIMING": {
        "code": "RF-004",
        "category": "Diagnostic Mismatch",
        "severity": "HIGH",
        "title": "Diagnostics passed but replacement requested",
        "template": (
            "Dell diagnostics PASSED for Service Tag {service_tag}, "
            "but customer is requesting replacement of {component}. "
            "No hardware fault detected by automated tests."
        ),
        "recommendation": (
            "Escalate to Level 2 support for manual remote diagnosis "
            "before approving replacement."
        ),
    },

    # --- High Value Targeting ---
    "HIGH_VALUE_COMPONENT_PATTERN": {
        "code": "RF-005",
        "category": "Value Targeting",
        "severity": "HIGH",
        "title": "Pattern of high-value component requests",
        "template": (
            "Customer has requested high-value components (>${threshold}) "
            "in {pct}% of all claims. Current request: {component} "
            "valued at ${value}."
        ),
        "recommendation": (
            "Require return of defective part BEFORE shipping "
            "replacement (advance exchange suspension)."
        ),
    },
    "COMPONENT_VALUE_EXCEEDS_DEVICE": {
        "code": "RF-006",
        "category": "Value Targeting",
        "severity": "MEDIUM",
        "title": "Component value disproportionate to device age",
        "template": (
            "Requested {component} (${value}) represents a significant "
            "portion of device value for a {age}-day-old {product}."
        ),
        "recommendation": (
            "Verify warranty entitlement and consider "
            "whole-unit replacement evaluation."
        ),
    },

    # --- Address / Identity Anomalies ---
    "MULTIPLE_CUSTOMERS_SAME_ADDRESS": {
        "code": "RF-007",
        "category": "Identity Anomaly",
        "severity": "HIGH",
        "title": "Multiple customer accounts linked to same address",
        "template": (
            "{num_customers} different customer accounts have shipped "
            "warranty parts to address {address_id}. "
            "Total claims at this address: {total_claims}."
        ),
        "recommendation": (
            "Flag address for fraud ring investigation. "
            "Hold all pending shipments to this address."
        ),
    },
    "MULTIPLE_ADDRESSES_SINGLE_CUSTOMER": {
        "code": "RF-008",
        "category": "Identity Anomaly",
        "severity": "MEDIUM",
        "title": "Customer using multiple shipping addresses",
        "template": (
            "Customer {customer_id} has used {num_addresses} different "
            "shipping addresses across {total_claims} claims."
        ),
        "recommendation": (
            "Verify customer identity. Request government-issued "
            "ID or original purchase invoice."
        ),
    },

    # --- Description Anomalies ---
    "TEMPLATED_DESCRIPTION": {
        "code": "RF-009",
        "category": "Description Anomaly",
        "severity": "HIGH",
        "title": "Claim description matches known fraud templates",
        "template": (
            "Claim description has a {similarity}% similarity match "
            "with {match_count} other flagged claims. "
            "Suspected copy-paste or scripted submission."
        ),
        "recommendation": (
            "Require a live video call with customer to "
            "visually verify the reported defect."
        ),
    },
    "VAGUE_DESCRIPTION": {
        "code": "RF-010",
        "category": "Description Anomaly",
        "severity": "LOW",
        "title": "Claim description lacks technical detail",
        "template": (
            "Description is only {length} characters with a technical "
            "detail score of {tech_score}/1.0. Genuine hardware "
            "failures typically include error codes or specific symptoms."
        ),
        "recommendation": (
            "Ask customer to provide specific error codes, "
            "screenshots, or diagnostic validation codes."
        ),
    },

    # --- Device Age Anomaly ---
    "NEW_DEVICE_HIGH_VALUE_CLAIM": {
        "code": "RF-011",
        "category": "Device Anomaly",
        "severity": "MEDIUM",
        "title": "High-value claim on very new device",
        "template": (
            "Device is only {age} days old. {component} failures at "
            "this age are statistically rare ({failure_rate}% "
            "expected failure rate for this batch)."
        ),
        "recommendation": (
            "Offer depot repair instead of part dispatch. "
            "Request the unit be sent in for inspection."
        ),
    },
    "EXPIRED_WARRANTY_ATTEMPT": {
        "code": "RF-012",
        "category": "Device Anomaly",
        "severity": "HIGH",
        "title": "Warranty expired or nearly expired",
        "template": (
            "Warranty for Service Tag {service_tag} expired "
            "{days_expired} days ago or expires within {days_left} days. "
            "Possible rush claim before coverage ends."
        ),
        "recommendation": (
            "Strictly verify entitlement. Do not override "
            "warranty dates without manager approval."
        ),
    },
}
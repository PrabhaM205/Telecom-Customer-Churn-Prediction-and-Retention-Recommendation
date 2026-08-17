# tests/test_recommendation.py
"""
Tests for the Guardrail Agent's policy engine
(src.recommendation.retention_rules.evaluate_offer_policy) and the
deterministic fallback offer generator
(src.recommendation.offer_engine.generate_fallback_offer).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recommendation.retention_rules import (
    evaluate_offer_policy,
    evaluate_retention_rules,
    get_tenure_tier,
)
from src.recommendation.offer_engine import generate_fallback_offer


def _customer(tenure_months=1, segment=""):
    return {"Tenure Months": tenure_months, "customer_segment": segment}


def _base_offer(**overrides):
    offer = {
        "offer_type": "bill_credit",
        "offer_text": "A modest bill credit.",
        "reason": "Targets month-to-month churn risk.",
        "discount_percent": 5,
        "duration_months": 3,
        "minimum_term_months": 0,
        "incentive_types": ["bill_credit"],
        "policy_basis": "Section 2.1",
        "customer_message": "Thanks for being a customer.",
    }
    offer.update(overrides)
    return offer


# ---------------------------------------------------------------------------
# Tenure tiers
# ---------------------------------------------------------------------------

def test_tenure_tier_boundaries():
    assert get_tenure_tier(0)["name"] == "0-6 months"
    assert get_tenure_tier(5.9)["name"] == "0-6 months"
    assert get_tenure_tier(6)["name"] == "6-24 months"
    assert get_tenure_tier(23.9)["name"] == "6-24 months"
    assert get_tenure_tier(24)["name"] == "2-5 years"
    assert get_tenure_tier(59.9)["name"] == "2-5 years"
    assert get_tenure_tier(60)["name"] == "5+ years"
    assert get_tenure_tier(120)["name"] == "5+ years"


# ---------------------------------------------------------------------------
# Guardrail: approval within caps
# ---------------------------------------------------------------------------

def test_offer_within_tenure_cap_is_approved():
    # 0-6 month tier: max 5% discount, max 3 month bill credit
    offer = _base_offer(discount_percent=5, duration_months=3)
    result = evaluate_offer_policy(offer, _customer(tenure_months=2))
    assert result["status"] == "APPROVED"
    assert result["approved"] is True
    assert result["retryable"] is False
    assert result["violations"] == []


def test_offer_at_higher_tenure_tier_allows_higher_discount():
    # 5+ year tier: max 20% discount
    offer = _base_offer(discount_percent=18, duration_months=10)
    result = evaluate_offer_policy(offer, _customer(tenure_months=72))
    assert result["status"] == "APPROVED"


# ---------------------------------------------------------------------------
# Guardrail: rejected-and-retryable
# ---------------------------------------------------------------------------

def test_discount_above_tenure_cap_is_rejected_and_retryable():
    # 0-6 month tier caps discount at 5% -- 25% must be rejected
    offer = _base_offer(discount_percent=25, duration_months=2)
    result = evaluate_offer_policy(offer, _customer(tenure_months=1))
    assert result["status"] == "REJECTED"
    assert result["approved"] is False
    assert result["retryable"] is True
    assert "TENURE_DISCOUNT_CAP" in result["violations"]
    assert result["feedback"]  # non-empty guardrail feedback generated


def test_bill_credit_duration_above_cap_is_rejected():
    # 0-6 month tier caps bill credit duration at 3 months
    offer = _base_offer(offer_type="bill_credit", discount_percent=2, duration_months=9)
    result = evaluate_offer_policy(offer, _customer(tenure_months=1))
    assert result["status"] == "REJECTED"
    assert "BILL_CREDIT_DURATION_CAP" in result["violations"]


# ---------------------------------------------------------------------------
# Guardrail: escalation cases
# ---------------------------------------------------------------------------

def test_permanent_discount_escalates():
    offer = _base_offer(offer_type="permanent_discount", discount_percent=10, duration_months=0)
    result = evaluate_offer_policy(offer, _customer(tenure_months=30))
    assert result["status"] == "ESCALATE"
    assert result["retryable"] is False
    assert "PERMANENT_DISCOUNT" in result["violations"]


def test_zero_duration_discount_escalates_as_permanent():
    # discount_percent > 0 with duration_months == 0 implies an unbounded /
    # permanent discount even if offer_type wasn't explicitly labeled so.
    offer = _base_offer(offer_type="contract_discount", discount_percent=10, duration_months=0)
    result = evaluate_offer_policy(offer, _customer(tenure_months=30))
    assert result["status"] == "ESCALATE"
    assert "PERMANENT_DISCOUNT" in result["violations"]


def test_multi_line_account_wide_escalates():
    offer = _base_offer(offer_type="multi_line_discount", offer_text="A multi-line account-wide discount.")
    result = evaluate_offer_policy(offer, _customer(tenure_months=30))
    assert result["status"] == "ESCALATE"
    assert "MULTI_LINE_ACCOUNT_WIDE" in result["violations"]


def test_too_many_incentive_types_escalates():
    offer = _base_offer(incentive_types=["discount", "bill_credit", "service_addon"])
    result = evaluate_offer_policy(offer, _customer(tenure_months=30))
    assert result["status"] == "ESCALATE"
    assert "TOO_MANY_INCENTIVE_TYPES" in result["violations"]


def test_malformed_offer_missing_fields_is_rejected_and_retryable():
    result = evaluate_offer_policy({"offer_type": "bill_credit"}, _customer(tenure_months=10))
    assert result["status"] == "REJECTED"
    assert result["retryable"] is True
    assert "MALFORMED_OFFER" in result["violations"]


def test_non_dict_offer_is_rejected_safely():
    result = evaluate_offer_policy("not an offer", _customer(tenure_months=10))
    assert result["status"] == "REJECTED"
    assert "MALFORMED_OFFER" in result["violations"]


def test_no_offer_type_is_rejected_not_retryable():
    offer = _base_offer(offer_type="no_offer", reason="No policy-supported offer found.")
    result = evaluate_offer_policy(offer, _customer(tenure_months=10))
    assert result["status"] == "REJECTED"
    assert result["retryable"] is False


# ---------------------------------------------------------------------------
# Deterministic fallback offer generator
# ---------------------------------------------------------------------------

def test_fallback_offer_is_well_formed_and_policy_compliant():
    diagnosis = {
        "summary": "High churn risk due to month-to-month contract.",
        "risk_drivers": [{"feature": "Contract_Month-to-month", "shap_value": 0.8}],
        "churn_probability": 0.8,
    }
    offer = generate_fallback_offer(
        customer_data={"Tenure Months": 3},
        diagnosis=diagnosis,
        churn_probability=0.8,
        customer_segment="Standard",
    )
    required_fields = [
        "offer_type", "offer_text", "reason", "discount_percent",
        "duration_months", "minimum_term_months", "incentive_types",
        "policy_basis", "customer_message",
    ]
    for field in required_fields:
        assert field in offer

    # Should clear the guardrail for this tenure (0-6 months tier).
    result = evaluate_offer_policy(offer, {"Tenure Months": 3, "customer_segment": "Standard"})
    assert result["status"] == "APPROVED"


def test_fallback_offer_does_not_repeat_previous_offer_type():
    diagnosis = {
        "summary": "High churn risk.",
        "risk_drivers": [{"feature": "Contract_Month-to-month", "shap_value": 0.8}],
        "churn_probability": 0.8,
    }
    previous_offer = {"offer_type": "contract_discount"}
    offer = generate_fallback_offer(
        customer_data={"Tenure Months": 3},
        diagnosis=diagnosis,
        churn_probability=0.8,
        customer_segment="Standard",
        previous_offer=previous_offer,
    )
    assert offer["offer_type"] != "contract_discount"


# ---------------------------------------------------------------------------
# Backward-compatible eligibility helper (display/reporting only)
# ---------------------------------------------------------------------------

def test_evaluate_retention_rules_still_returns_eligibility_shape():
    result = evaluate_retention_rules(
        customer_data={"tenure": 36, "MonthlyCharges": 95, "Contract": "Month-to-month"},
        churn_probability=0.87,
        revenue_at_risk=24000,
        customer_segment="High Value",
    )
    assert result["eligible"] is True
    assert result["risk_level"] == "HIGH"

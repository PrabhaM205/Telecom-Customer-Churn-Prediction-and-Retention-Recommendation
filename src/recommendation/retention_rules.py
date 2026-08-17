"""
Retention Rules
---------------
Defines business rules for customer retention.

evaluate_retention_rules(): kept for backward compatibility -- a lightweight
"is this customer generally a retention candidate" check based on churn
probability / segment / revenue, used for reporting/UI display. This is
NOT the Guardrail Agent and is deliberately not treated as such (diagnosis
and eligibility are different concepts).

evaluate_offer_policy(): the real Guardrail Agent policy engine. Validates a
drafted structured offer (as produced by genai/offer_generator.py) against
numeric, tenure-tiered company policy caps loaded from config.yaml. Core
numeric checks are enforced here in Python -- never left to an LLM prompt.
"""

from src.config import get as config_get


# ---------------------------------------------------------------------------
# Guardrail policy tiers -- loaded from config.yaml (single source of truth)
# ---------------------------------------------------------------------------

_DEFAULT_TENURE_TIERS = [
    {"name": "0-6 months", "min_tenure_months": 0, "max_tenure_months": 6,
     "max_discount_percent": 5, "max_bill_credit_months": 3},
    {"name": "6-24 months", "min_tenure_months": 6, "max_tenure_months": 24,
     "max_discount_percent": 10, "max_bill_credit_months": 6},
    {"name": "2-5 years", "min_tenure_months": 24, "max_tenure_months": 60,
     "max_discount_percent": 15, "max_bill_credit_months": 9},
    {"name": "5+ years", "min_tenure_months": 60, "max_tenure_months": None,
     "max_discount_percent": 20, "max_bill_credit_months": 12},
]


def _get_tenure_tiers():
    return config_get("guardrail", "tenure_tiers", default=_DEFAULT_TENURE_TIERS)


def _get_max_incentive_types():
    return config_get("guardrail", "max_incentive_types", default=2)


def _get_restricted_segments():
    return config_get("guardrail", "restricted_customer_segments", default=[]) or []


def get_tenure_tier(tenure_months: float) -> dict:
    """Returns the policy tier dict that applies to this tenure (months)."""
    tenure_months = float(tenure_months)
    tiers = _get_tenure_tiers()
    for tier in tiers:
        lower = tier.get("min_tenure_months", 0) or 0
        upper = tier.get("max_tenure_months", None)
        if tenure_months >= lower and (upper is None or tenure_months < upper):
            return tier
    # Defensive fallback: most restrictive tier
    return tiers[0] if tiers else _DEFAULT_TENURE_TIERS[0]


def _extract_tenure_months(customer_data: dict) -> float:
    for key in ("Tenure Months", "tenure_months", "tenure", "Tenure"):
        if key in customer_data and customer_data[key] not in (None, ""):
            try:
                return float(customer_data[key])
            except (TypeError, ValueError):
                continue
    return 0.0


# ---------------------------------------------------------------------------
# Guardrail Agent -- real policy enforcement of a drafted structured offer
# ---------------------------------------------------------------------------

APPROVED = "APPROVED"
REJECTED = "REJECTED"
ESCALATE = "ESCALATE"

# Offer types that represent a PERMANENT plan-level discount (no expiry) --
# these always require human approval regardless of the discount size.
_PERMANENT_OFFER_TYPES = {"permanent_discount", "plan_level_discount", "permanent_plan_discount"}

# Keywords in offer_type/offer_text/customer_message that indicate a
# multi-line / account-wide discount -- these always escalate.
_MULTI_LINE_KEYWORDS = ("multi-line", "multi line", "multiline", "account-wide", "account wide", "all lines")


def evaluate_offer_policy(offer: dict, customer_data: dict, guardrail_config: dict = None) -> dict:
    """
    The Guardrail Agent. Validates a candidate structured offer against
    company retention policy. Returns:

        {"status": "APPROVED"|"REJECTED"|"ESCALATE",
         "approved": bool,
         "retryable": bool,
         "violations": [...],
         "feedback": "..."}

    REJECTED offers are retryable (the Offer-Strategist can produce a new
    compliant offer). ESCALATE means the offer structurally requires a human
    (permanent discounts, multi-line/account-wide, too many incentive types,
    restricted customers) and is never retried automatically.
    """
    violations = []
    feedback_parts = []

    if not isinstance(offer, dict):
        return {
            "status": REJECTED,
            "approved": False,
            "retryable": True,
            "violations": ["MALFORMED_OFFER"],
            "feedback": "Offer was not a valid structured object.",
        }

    required_fields = [
        "offer_type", "offer_text", "reason", "discount_percent",
        "duration_months", "minimum_term_months", "incentive_types",
        "policy_basis", "customer_message",
    ]
    missing = [f for f in required_fields if f not in offer]
    if missing:
        return {
            "status": REJECTED,
            "approved": False,
            "retryable": True,
            "violations": ["MALFORMED_OFFER"],
            "feedback": f"Offer JSON is missing required field(s): {', '.join(missing)}.",
        }

    offer_type = str(offer.get("offer_type", "")).strip().lower()

    # "no_offer" is not a policy violation -- there's simply nothing to approve.
    if offer_type == "no_offer":
        return {
            "status": REJECTED,
            "approved": False,
            "retryable": False,
            "violations": ["NO_POLICY_SUPPORTED_OFFER"],
            "feedback": offer.get("reason", "No policy-supported offer was identified."),
        }

    try:
        discount_percent = float(offer.get("discount_percent") or 0)
    except (TypeError, ValueError):
        discount_percent = 0.0
    try:
        duration_months = float(offer.get("duration_months") or 0)
    except (TypeError, ValueError):
        duration_months = 0.0
    try:
        minimum_term_months = float(offer.get("minimum_term_months") or 0)
    except (TypeError, ValueError):
        minimum_term_months = 0.0

    incentive_types = offer.get("incentive_types") or []
    if not isinstance(incentive_types, list):
        incentive_types = [incentive_types]

    customer_segment = str(customer_data.get("customer_segment", "")).strip()

    # -----------------------------------------------------------------
    # ESCALATION checks (never retried automatically)
    # -----------------------------------------------------------------

    # Permanent / plan-level discount
    if offer_type in _PERMANENT_OFFER_TYPES or (discount_percent > 0 and duration_months == 0):
        violations.append("PERMANENT_DISCOUNT")
        feedback_parts.append(
            "Permanent / plan-level discounts (no expiry) require human approval."
        )

    # Multi-line / account-wide restriction
    text_blob = " ".join([
        offer_type,
        str(offer.get("offer_text", "")),
        str(offer.get("customer_message", "")),
    ]).lower()
    if any(kw in text_blob for kw in _MULTI_LINE_KEYWORDS):
        violations.append("MULTI_LINE_ACCOUNT_WIDE")
        feedback_parts.append("Multi-line / account-wide discounts require human approval.")

    # Too many incentive types
    max_incentives = _get_max_incentive_types()
    if len(incentive_types) > max_incentives:
        violations.append("TOO_MANY_INCENTIVE_TYPES")
        feedback_parts.append(
            f"Offer bundles {len(incentive_types)} incentive types; "
            f"more than {max_incentives} requires human approval."
        )

    # Restricted / excluded customer segment
    if customer_segment and customer_segment in _get_restricted_segments():
        violations.append("RESTRICTED_CUSTOMER")
        feedback_parts.append(f"Customer segment '{customer_segment}' is restricted from automated offers.")

    if violations:
        return {
            "status": ESCALATE,
            "approved": False,
            "retryable": False,
            "violations": violations,
            "feedback": " ".join(feedback_parts),
        }

    # -----------------------------------------------------------------
    # REJECTED-AND-RETRYABLE checks (tenure-tiered numeric caps)
    # -----------------------------------------------------------------

    tenure_months = _extract_tenure_months(customer_data)
    tier = get_tenure_tier(tenure_months)

    if discount_percent > tier["max_discount_percent"]:
        violations.append("TENURE_DISCOUNT_CAP")
        feedback_parts.append(
            f"Discount of {discount_percent:.0f}% exceeds the {tier['max_discount_percent']}% "
            f"auto-approval cap for tenure tier '{tier['name']}'. Reduce the discount."
        )

    if offer_type in ("bill_credit", "bill credit") and duration_months > tier["max_bill_credit_months"]:
        violations.append("BILL_CREDIT_DURATION_CAP")
        feedback_parts.append(
            f"Bill credit duration of {duration_months:.0f} months exceeds the "
            f"{tier['max_bill_credit_months']}-month cap for tenure tier '{tier['name']}'. Shorten the duration."
        )

    if minimum_term_months and minimum_term_months > 24:
        violations.append("MINIMUM_TERM_TOO_LONG")
        feedback_parts.append(
            f"Minimum term of {minimum_term_months:.0f} months exceeds the 24-month policy ceiling."
        )

    if violations:
        return {
            "status": REJECTED,
            "approved": False,
            "retryable": True,
            "violations": violations,
            "feedback": " ".join(feedback_parts),
        }

    return {
        "status": APPROVED,
        "approved": True,
        "retryable": False,
        "violations": [],
        "feedback": (
            f"Offer approved: within {tier['name']} tenure tier caps "
            f"(discount <= {tier['max_discount_percent']}%, "
            f"bill credit <= {tier['max_bill_credit_months']} months)."
        ),
    }


def evaluate_retention_rules(
    customer_data,
    churn_probability,
    revenue_at_risk=0.0,
    customer_segment="Standard"
):
    """
    Evaluate whether a customer should receive a retention action.
    """

    # ---------------------------------------------------------
    # Basic values
    # ---------------------------------------------------------

    churn_probability = float(churn_probability)
    revenue_at_risk = float(revenue_at_risk)

    tenure = float(
        customer_data.get(
            "tenure",
            customer_data.get("Tenure", 0)
        )
    )

    monthly_charges = float(
        customer_data.get(
            "monthly_charges",
            customer_data.get("MonthlyCharges", 0)
        )
    )

    contract = str(
        customer_data.get(
            "contract",
            customer_data.get("Contract", "")
        )
    ).lower()

    payment_method = str(
        customer_data.get(
            "payment_method",
            customer_data.get("PaymentMethod", "")
        )
    ).lower()

    # ---------------------------------------------------------
    # Risk classification
    # ---------------------------------------------------------

    if churn_probability >= 0.75:
        risk_level = "HIGH"

    elif churn_probability >= 0.50:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # ---------------------------------------------------------
    # Retention eligibility
    # ---------------------------------------------------------

    eligible = False
    reasons = []
    recommended_actions = []

    # High churn customers
    if churn_probability >= 0.75:
        eligible = True

        reasons.append(
            "High predicted churn probability"
        )

        recommended_actions.append(
            "Personalized retention offer"
        )

    # Medium churn customers
    elif churn_probability >= 0.50:
        eligible = True

        reasons.append(
            "Moderate predicted churn probability"
        )

        recommended_actions.append(
            "Targeted retention communication"
        )

    # ---------------------------------------------------------
    # High-value customer
    # ---------------------------------------------------------

    if customer_segment.lower() in [
        "high value",
        "high_value",
        "premium",
        "vip"
    ]:
        eligible = True

        reasons.append(
            "Customer belongs to a high-value segment"
        )

        recommended_actions.append(
            "Priority retention treatment"
        )

    # ---------------------------------------------------------
    # Revenue at risk
    # ---------------------------------------------------------

    if revenue_at_risk >= 10000:

        eligible = True

        reasons.append(
            "High revenue is at risk"
        )

        recommended_actions.append(
            "Escalated retention action"
        )

    # ---------------------------------------------------------
    # Contract-related risk
    # ---------------------------------------------------------

    if "month-to-month" in contract:

        reasons.append(
            "Customer is on a month-to-month contract"
        )

        recommended_actions.append(
            "Offer long-term contract incentive"
        )

    # ---------------------------------------------------------
    # High monthly charges
    # ---------------------------------------------------------

    if monthly_charges >= 80:

        reasons.append(
            "Customer has relatively high monthly charges"
        )

        recommended_actions.append(
            "Consider eligible pricing or plan discount"
        )

    # ---------------------------------------------------------
    # Electronic check payment
    # ---------------------------------------------------------

    if "electronic check" in payment_method:

        reasons.append(
            "Customer uses electronic check payment"
        )

    # ---------------------------------------------------------
    # Tenure
    # ---------------------------------------------------------

    if tenure >= 24:

        reasons.append(
            "Customer has significant tenure"
        )

        recommended_actions.append(
            "Consider loyalty-based retention treatment"
        )

    # ---------------------------------------------------------
    # Remove duplicate actions
    # ---------------------------------------------------------

    recommended_actions = list(
        dict.fromkeys(recommended_actions)
    )

    reasons = list(
        dict.fromkeys(reasons)
    )

    # ---------------------------------------------------------
    # Default action
    # ---------------------------------------------------------

    if not recommended_actions:

        recommended_actions.append(
            "No immediate retention offer required"
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    return {
        "eligible": eligible,
        "risk_level": risk_level,
        "reasons": reasons,
        "recommended_actions": recommended_actions,
        "customer_segment": customer_segment,
        "churn_probability": churn_probability,
        "revenue_at_risk": revenue_at_risk
    }


if __name__ == "__main__":

    customer = {
        "tenure": 36,
        "MonthlyCharges": 95,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
    }

    result = evaluate_retention_rules(
        customer_data=customer,
        churn_probability=0.87,
        revenue_at_risk=24000,
        customer_segment="High Value"
    )

    print("\n" + "=" * 60)
    print("RETENTION RULE TEST")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    print("=" * 60)
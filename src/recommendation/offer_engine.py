from pathlib import Path
import sys
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "Telco_customer_churn.xlsx"
)


# ============================================================
# PYTHON PATH
# ============================================================

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# EXISTING PROJECT MODULES
# ============================================================

from src.risk.riske import score_single_customer

from genai.rag.retriever import PolicyRetriever

from genai.llm_client import generate_response


# ============================================================
# LOAD CUSTOMER DATA
# ============================================================

def load_customer_data():

    print("\nLoading customer dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_excel(DATASET_PATH)

    print(
        f"Dataset loaded successfully: "
        f"{len(df)} customers"
    )

    return df


# ============================================================
# FIND CUSTOMER
# ============================================================

def get_customer(df, customer_id):

    customer_id = str(customer_id).strip()

    df["CustomerID"] = (
        df["CustomerID"]
        .astype(str)
        .str.strip()
    )

    customer = df[
        df["CustomerID"] == customer_id
    ]

    if customer.empty:
        return None

    return customer.iloc[0]


# ============================================================
# CUSTOMER PROFILE
# ============================================================

def display_customer_profile(customer):

    print("\n")
    print("=" * 60)
    print("CUSTOMER PROFILE")
    print("=" * 60)

    print(
        f"Customer ID          : "
        f"{customer.get('CustomerID', 'N/A')}"
    )

    print(
        f"Tenure               : "
        f"{customer.get('Tenure Months', 'N/A')} months"
    )

    print(
        f"Monthly Charges      : "
        f"${float(customer.get('Monthly Charges', 0)):.2f}"
    )

    print(
        f"Contract             : "
        f"{customer.get('Contract', 'N/A')}"
    )

    print(
        f"Payment Method       : "
        f"{customer.get('Payment Method', 'N/A')}"
    )


# ============================================================
# RISK ANALYSIS
# ============================================================

def analyze_risk(customer):

    print("\n")
    print("=" * 60)
    print("CUSTOMER RISK ASSESSMENT")
    print("=" * 60)

    customer_data = customer.to_dict()

    risk_result = score_single_customer(
        customer_data
    )

    print(
        f"\nChurn Probability    : "
        f"{risk_result['churn_probability']:.2%}"
    )

    print(
        f"Risk Level           : "
        f"{risk_result['risk_tier'].upper()}"
    )

    print(
        f"Priority Score       : "
        f"{risk_result['priority_score']}"
    )

    print(
        f"Monthly Revenue Risk : "
        f"${risk_result['monthly_revenue_at_risk']:.2f}"
    )

    print(
        f"Annual Revenue Risk  : "
        f"${risk_result['annual_revenue_at_risk']:.2f}"
    )

    return customer_data, risk_result


# ============================================================
# BUSINESS CONTEXT
# ============================================================

def build_business_context(
    customer_data,
    risk_result
):

    """
    Uses the values already produced by the existing
    risk/business logic.

    No offer is hardcoded here.
    """

    return {
        "customer_id":
            customer_data.get("CustomerID"),

        "monthly_revenue_at_risk":
            risk_result.get(
                "monthly_revenue_at_risk"
            ),

        "annual_revenue_at_risk":
            risk_result.get(
                "annual_revenue_at_risk"
            ),

        "priority_score":
            risk_result.get(
                "priority_score"
            ),

        "risk_level":
            risk_result.get(
                "risk_tier"
            ),

        "churn_probability":
            risk_result.get(
                "churn_probability"
            )
    }


# ============================================================
# POLICY QUERY
# ============================================================

def build_policy_query(
    customer_data,
    risk_result,
    business_context
):

    return f"""
Find retention policies and eligible retention treatments
that may apply to this telecom customer.

Customer:

Customer ID:
{customer_data.get('CustomerID')}

Tenure:
{customer_data.get('Tenure Months')} months

Monthly Charges:
{customer_data.get('Monthly Charges')}

Contract:
{customer_data.get('Contract')}

Payment Method:
{customer_data.get('Payment Method')}


Risk:

Churn Probability:
{risk_result.get('churn_probability'):.2%}

Risk Level:
{risk_result.get('risk_tier')}

Priority Score:
{risk_result.get('priority_score')}


Business:

Monthly Revenue At Risk:
{business_context.get('monthly_revenue_at_risk')}

Annual Revenue At Risk:
{business_context.get('annual_revenue_at_risk')}


Retrieve only company-policy information relevant to:

- retention offers
- eligibility
- exclusions
- contract requirements
- discounts
- credits
- incentives
- approval requirements
- escalation requirements

Do not invent information.
"""


# ============================================================
# POLICY RETRIEVAL
# ============================================================

def retrieve_policy(
    customer_data,
    risk_result,
    business_context
):

    print("\n")
    print("=" * 60)
    print("COMPANY POLICY ANALYSIS")
    print("=" * 60)

    query = build_policy_query(
        customer_data,
        risk_result,
        business_context
    )

    print(
        "\nChecking company retention policy..."
    )

    retriever = PolicyRetriever()

    results = retriever.search(
        query,
        top_k=5
    )

    if not results:
        return None

    policy_parts = []

    for result in results:

        text = result.get(
            "text",
            ""
        )

        if text:
            policy_parts.append(text)

    policy_text = "\n\n".join(
        policy_parts
    )

    print(
        "\nRelevant policy information found."
    )

    return policy_text


# ============================================================
# GEMINI FLASH
# ============================================================

def generate_recommendation(
    customer_data,
    risk_result,
    business_context,
    policy_text
):

    print("\n")
    print("=" * 60)
    print("PERSONALIZED RETENTION RECOMMENDATION")
    print("=" * 60)

    prompt = f"""
You are a telecom customer retention recommendation
specialist.

Your job is to analyze the customer information and
recommend the most appropriate retention action.

IMPORTANT SOURCE RULE:

The COMPANY POLICY below is the ONLY source for deciding
what offers, discounts, credits, incentives, eligibility
rules and conditions are available.

Do not invent any offer.

Do not create any discount percentage.

Do not create any benefit.

Do not create eligibility rules.

Do not assume an offer exists unless it is supported
by the supplied company policy.


CUSTOMER INFORMATION
====================

Customer ID:
{customer_data.get('CustomerID')}

Tenure:
{customer_data.get('Tenure Months')} months

Monthly Charges:
{customer_data.get('Monthly Charges')}

Contract:
{customer_data.get('Contract')}

Payment Method:
{customer_data.get('Payment Method')}


RISK INFORMATION
================

Churn Probability:
{risk_result.get('churn_probability'):.2%}

Risk Level:
{risk_result.get('risk_tier')}

Priority Score:
{risk_result.get('priority_score')}

Monthly Revenue At Risk:
{risk_result.get('monthly_revenue_at_risk')}

Annual Revenue At Risk:
{risk_result.get('annual_revenue_at_risk')}


BUSINESS CONTEXT
================

{business_context}


COMPANY POLICY
==============

{policy_text}


DECISION PROCESS
================

1. Understand the customer's risk.

2. Understand the customer's profile.

3. Consider the business impact.

4. Examine the supplied company policy.

5. Identify policy-supported retention treatments.

6. Check the customer's profile against the policy
   eligibility conditions.

7. Select ONE suitable offer if one is supported.

8. Explain why that offer fits this customer.

9. Mention important policy conditions.

10. If the policy does not support an eligible offer,
    clearly say that no eligible retention offer was
    identified.


STRICT RULE
===========

The recommended offer MUST come from the supplied
company policy.

Never use your general knowledge to create an offer.


OUTPUT FORMAT
=============

RECOMMENDED OFFER:
<one policy-supported offer>

WHY THIS OFFER:
<clear explanation based on customer risk and profile>

POLICY BASIS:
<policy-supported reason>

RECOMMENDED ACTION:
<what the employee should do>


Do not mention:

RAG
FAISS
embeddings
vector database
retriever
XGBoost
machine learning
internal implementation
Python
software architecture

Do not use placeholders such as "...".

Do not leave any section empty.
"""

    response = generate_response(
        prompt
    )

    return response.strip()


# ============================================================
# FINAL OUTPUT
# ============================================================

def display_final_output(
    recommendation
):

    print("\n")
    print("=" * 60)
    print("FINAL RETENTION RECOMMENDATION")
    print("=" * 60)

    print("\n")
    print(recommendation)

    print("\n")
    print("=" * 60)
    print("RECOMMENDATION COMPLETED")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("TELECOM CUSTOMER RETENTION SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    df = load_customer_data()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    customer_id = input(
        "\nEnter Customer ID: "
    ).strip()

    if not customer_id:

        print(
            "\nCustomer ID cannot be empty."
        )

        return

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    customer = get_customer(
        df,
        customer_id
    )

    if customer is None:

        print("\n")
        print("=" * 60)
        print("CUSTOMER NOT FOUND")
        print("=" * 60)

        print(
            f"\nCustomer ID '{customer_id}' "
            "was not found in the dataset."
        )

        return

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    display_customer_profile(
        customer
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    customer_data, risk_result = (
        analyze_risk(
            customer
        )
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("RETENTION DECISION")
    print("=" * 60)

    risk_level = str(
        risk_result.get(
            "risk_tier",
            ""
        )
    ).strip().lower()

    print(
        f"\nCustomer Risk Level : "
        f"{risk_level.upper()}"
    )

    # --------------------------------------------------------
    # LOW / MEDIUM RISK
    # --------------------------------------------------------

    if risk_level not in [
        "high",
        "critical"
    ]:

        print(
            "\nNo high-priority retention intervention "
            "is required."
        )

        print("\n")
        print("=" * 60)
        print("RECOMMENDATION COMPLETED")
        print("=" * 60)

        return

    # --------------------------------------------------------
    # HIGH / CRITICAL RISK
    # --------------------------------------------------------

    print(
        "\nHIGH-RISK CUSTOMER DETECTED"
    )

    print(
        "Retention intervention is required."
    )

    # --------------------------------------------------------
    # STEP 7: BUSINESS CONTEXT
    # --------------------------------------------------------

    business_context = build_business_context(
        customer_data,
        risk_result
    )

    # --------------------------------------------------------
    # STEP 8: POLICY
    # --------------------------------------------------------

    policy_text = retrieve_policy(
        customer_data,
        risk_result,
        business_context
    )

    if not policy_text:

        print(
            "\nNo relevant company policy "
            "information was found."
        )

        print(
            "\nNo policy-supported offer can be "
            "recommended."
        )

        return

    # --------------------------------------------------------
    # STEP 9: GEMINI FLASH
    # --------------------------------------------------------

    recommendation = generate_recommendation(
        customer_data,
        risk_result,
        business_context,
        policy_text
    )

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    display_final_output(
        recommendation
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
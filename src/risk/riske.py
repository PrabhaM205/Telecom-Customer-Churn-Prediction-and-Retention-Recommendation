# src/risk/risk_engine.py
"""
Risk scoring engine built on top of the trained XGBoost churn model.

Turns a raw predicted churn probability into business-usable output:
- a risk TIER (Low / Medium / High / Critical) for triage
- revenue-at-risk figures (monthly + annualized), using Monthly Charges as
  the recurring revenue this customer represents
- an optional priority_score that blends probability and revenue, so
  retention teams work the highest-value-at-risk accounts first, not just
  the highest-probability ones

Does NOT retrain or refit anything -- loads the same fitted artifacts as
src/explainability/shap_explainer.py (artifacts/xgboost_churn_model.joblib,
artifacts/preprocessor.joblib) and reuses the same clean -> engineer ->
transform chain as training, so scores are guaranteed consistent with the
model's training-time behavior.
"""

import os

import joblib
import numpy as np
import pandas as pd

from src.data.loader import load_raw_data
from src.data.cleaner import clean_data
from src.data.feature_engineering import engineer_features
from src.preprocessing.pipeline import load_transform


# --------------------------------------------------
# Paths
# --------------------------------------------------

_THIS_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(_THIS_DIR, "..", "..", "artifacts", "xgboost_churn_model.joblib")
PREPROCESSOR_PATH = os.path.join(_THIS_DIR, "..", "..", "artifacts", "preprocessor.joblib")
PROCESSED_DIR = os.path.join(_THIS_DIR, "..", "..", "data", "processed")


# --------------------------------------------------
# Risk tier thresholds -- named constants, single source of truth
# (matches the style of src/preprocessing/pipeline.py's column groups)
# --------------------------------------------------

RISK_TIERS = [
    ("Low",      0.00, 0.30),
    ("Medium",   0.30, 0.60),
    ("High",     0.60, 0.80),
    ("Critical", 0.80, 1.01),  # 1.01 so probability == 1.0 is inclusive
]


def assign_risk_tier(probability: float) -> str:
    """Maps a churn probability in [0, 1] to a named risk tier."""
    for tier_name, lower, upper in RISK_TIERS:
        if lower <= probability < upper:
            return tier_name
    return "Unknown"  # defensive fallback; should never hit with valid probabilities


# --------------------------------------------------
# Artifact loading
# --------------------------------------------------

def load_artifacts(model_path: str = MODEL_PATH, preprocessor_path: str = PREPROCESSOR_PATH):
    """Loads the fitted XGBoost model and preprocessor."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            f"Run `python -m src.models.xgboost_model` first."
        )
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"No fitted preprocessor found at {preprocessor_path}. "
            f"Run `python -m src.preprocessing.pipeline` first."
        )
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


# --------------------------------------------------
# Scoring
# --------------------------------------------------

def predict_churn_probability(
    df_raw: pd.DataFrame,
    model=None,
    preprocessor_path: str = PREPROCESSOR_PATH,
) -> tuple:
    """
    Runs raw customer rows through clean -> engineer -> transform -> predict.
    Returns (probabilities, df_engineered) so callers can align scores back
    to CustomerID / Monthly Charges / etc.
    """
    if model is None:
        model, _ = load_artifacts(preprocessor_path=preprocessor_path)

    df = clean_data(df_raw)
    df = engineer_features(df)
    X_transformed = load_transform(df, artifact_path=preprocessor_path)
    probabilities = model.predict_proba(X_transformed)[:, 1]
    return probabilities, df


def compute_revenue_at_risk(probability: np.ndarray, monthly_charges: pd.Series) -> pd.DataFrame:
    """
    Expected revenue at risk = churn probability x recurring revenue.
    Monthly Charges is used as the recurring-revenue base (not ARPU, which
    is a historical average and can be distorted for very new customers --
    see feature_engineering.add_arpu). Annualized figure is monthly x 12,
    a standard simple run-rate projection (no discounting/survival curve).
    """
    monthly_charges = monthly_charges.reset_index(drop=True)
    probability = pd.Series(probability).reset_index(drop=True)
    return pd.DataFrame({
        "monthly_revenue_at_risk": (probability * monthly_charges).round(2),
        "annual_revenue_at_risk": (probability * monthly_charges * 12).round(2),
    })


def score_customers(
    df_raw: pd.DataFrame,
    model=None,
    preprocessor_path: str = PREPROCESSOR_PATH,
) -> pd.DataFrame:
    """
    Full scoring pipeline for a batch of raw customer rows. Returns one row
    per customer with: CustomerID, churn_probability, risk_tier,
    monthly/annual revenue at risk, and a priority_score for ranking
    retention outreach (probability-weighted revenue, 0-100 scaled).
    """
    probabilities, df_engineered = predict_churn_probability(
        df_raw, model=model, preprocessor_path=preprocessor_path
    )

    revenue_df = compute_revenue_at_risk(probabilities, df_engineered["Monthly Charges"])

    results = pd.DataFrame({
        "CustomerID": df_engineered["CustomerID"].reset_index(drop=True),
        "churn_probability": np.round(probabilities, 4),
        "risk_tier": [assign_risk_tier(p) for p in probabilities],
    })
    results = pd.concat([results, revenue_df], axis=1)

    # Priority score: rank customers by revenue-at-risk, scaled 0-100.
    # Two customers at the same churn probability are NOT equal priority --
    # the one paying more is the bigger loss if they leave.
    max_at_risk = results["annual_revenue_at_risk"].max()
    if max_at_risk > 0:
        results["priority_score"] = (results["annual_revenue_at_risk"] / max_at_risk * 100).round(1)
    else:
        results["priority_score"] = 0.0

    results = results.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return results


def score_single_customer(customer_row: dict, model=None, preprocessor_path: str = PREPROCESSOR_PATH) -> dict:
    """
    Scores one customer, given as a dict of raw column -> value (same schema
    as a row from load_raw_data()). Convenience wrapper for API/app use --
    see api/main.py or app/pages/02_Customer_Risk.py for the caller.
    """
    df_raw = pd.DataFrame([customer_row])
    results = score_customers(df_raw, model=model, preprocessor_path=preprocessor_path)
    return results.iloc[0].to_dict()


# --------------------------------------------------
# Portfolio-level summary
# --------------------------------------------------

def get_portfolio_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates scored customers by risk tier: customer counts and total
    revenue at risk per tier. This is the table a retention-ops dashboard
    would show at the top (see app/pages/01_Executive_Dashboard.py).
    """
    tier_order = [t[0] for t in RISK_TIERS]
    summary = (
        results.groupby("risk_tier")
        .agg(
            customer_count=("CustomerID", "count"),
            total_monthly_revenue_at_risk=("monthly_revenue_at_risk", "sum"),
            total_annual_revenue_at_risk=("annual_revenue_at_risk", "sum"),
            avg_churn_probability=("churn_probability", "mean"),
        )
        .reindex(tier_order)  # keep Low -> Critical order even if a tier has 0 customers
        .fillna(0)
        .reset_index()
    )
    summary["total_annual_revenue_at_risk"] = summary["total_annual_revenue_at_risk"].round(2)
    summary["total_monthly_revenue_at_risk"] = summary["total_monthly_revenue_at_risk"].round(2)
    summary["avg_churn_probability"] = summary["avg_churn_probability"].round(4)
    return summary


# --------------------------------------------------
# Single-customer lookup (interactive entry point)
# --------------------------------------------------

def lookup_customer_risk(customer_id: str, df_raw: pd.DataFrame, model=None) -> dict:
    """
    Looks up ONE customer by CustomerID and returns their risk tier,
    probability, and revenue-at-risk snapshot. Returns None (and prints a
    message) if the ID isn't found -- non-fatal so an interactive session
    can just try again.
    """
    match = df_raw[df_raw["CustomerID"] == customer_id]
    if match.empty:
        print(f"No customer found with ID '{customer_id}'. Check the ID and try again.")
        return None

    result = score_customers(match, model=model).iloc[0].to_dict()
    return result


def print_customer_risk_report(result: dict) -> None:
    """Pretty-prints one customer's risk snapshot to the console."""
    print("\n" + "=" * 60)
    print(f"CUSTOMER RISK SNAPSHOT -- {result['CustomerID']}")
    print("=" * 60)
    print(f"Churn probability:        {result['churn_probability']:.2%}")
    print(f"Risk tier:                {result['risk_tier']}")
    print(f"Monthly revenue at risk:  ${result['monthly_revenue_at_risk']:,.2f}")
    print(f"Annual revenue at risk:   ${result['annual_revenue_at_risk']:,.2f}")
    print(f"Priority score (0-100):   {result['priority_score']}")
    print(
        "\nFor the specific reasons behind this score and a suggested "
        "retention strategy, run: python -m src.explainability.shap_explainer"
    )
    print("=" * 60 + "\n")


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Loading artifacts...")
    model, _ = load_artifacts()

    print("Loading raw customer data...")
    df_raw = load_raw_data()

    # --- Interactive: look up ONE customer first, before the bulk report ---
    customer_id_input = input(
        "\nEnter a Customer ID to see their individual risk snapshot "
        "(press Enter to skip): "
    ).strip()
    if customer_id_input:
        single_result = lookup_customer_risk(customer_id_input, df_raw, model=model)
        if single_result:
            print_customer_risk_report(single_result)

    print("Scoring full customer base...")
    results = score_customers(df_raw, model=model)

    output_path = os.path.join(PROCESSED_DIR, "customer_risk_scores.csv")
    results.to_csv(output_path, index=False)
    print(f"\nScored {len(results)} customers. Saved to: {output_path}")

    print("\n========== Portfolio Risk Summary ==========")
    summary = get_portfolio_summary(results)
    print(summary.to_string(index=False))

    print("\nTop 10 customers by priority score (probability x revenue):")
    print(
        results[["CustomerID", "churn_probability", "risk_tier", "annual_revenue_at_risk", "priority_score"]]
        .head(10)
        .to_string(index=False)
    )
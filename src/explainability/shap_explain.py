# src/explainability/shap_explainer.py
"""
SHAP-based explainability for the trained XGBoost churn model.

Design mirrors src/models/xgboost_model.py and src/preprocessing/pipeline.py:
- Artifacts (model, preprocessor) are loaded from artifacts/, never refit here.
- SHAP is computed on the TRANSFORMED feature matrix (the same 48-column
  numeric/one-hot space the model was trained on) using the fitted
  preprocessor's get_feature_names_out() for readable labels.

SHAP values here are in margin (log-odds) space, which is the TreeExplainer
default for XGBClassifier -- NOT probability space. base_value + sum(shap
values) for a row, passed through a sigmoid, reproduces the model's
predicted probability (see verify_shap_consistency()). This matters when
interpreting magnitudes: a SHAP value of +0.4 does not mean "+40% probability",
it means "+0.4 log-odds toward churn."
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")  # headless-safe: never try to open a GUI window
import matplotlib.pyplot as plt

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

FIGURES_DIR = os.path.join(_THIS_DIR, "..", "..", "outputs", "figures")
REPORTS_DIR = os.path.join(_THIS_DIR, "..", "..", "outputs", "reports")


# --------------------------------------------------
# Artifact loading
# --------------------------------------------------

def load_artifacts(model_path: str = MODEL_PATH, preprocessor_path: str = PREPROCESSOR_PATH):
    """
    Loads the fitted XGBoost model and preprocessor. Raises a clear error
    if training hasn't been run yet (mirrors the error style in pipeline.py).
    """
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


def get_explainer(model) -> shap.TreeExplainer:
    """Builds a TreeExplainer for the XGBoost model. Cheap -- safe to call per run."""
    return shap.TreeExplainer(model)


# --------------------------------------------------
# Data prep -- raw customer rows -> transformed feature matrix
# --------------------------------------------------

def prepare_features(df_raw: pd.DataFrame, preprocessor_path: str = PREPROCESSOR_PATH):
    """
    Runs raw customer rows through the SAME clean -> engineer -> transform
    chain used at training time. Returns (X_transformed, feature_names, df_engineered).
    df_engineered is kept so callers can look up CustomerID / raw fields
    alongside SHAP output.
    """
    df = clean_data(df_raw)
    df = engineer_features(df)
    X_transformed = load_transform(df, artifact_path=preprocessor_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_names = list(preprocessor.get_feature_names_out())
    return X_transformed, feature_names, df


# --------------------------------------------------
# Global explainability
# --------------------------------------------------

def compute_shap_values(explainer: shap.TreeExplainer, X_transformed: np.ndarray) -> shap.Explanation:
    """Computes a shap.Explanation for a batch of transformed rows."""
    return explainer(X_transformed)


def global_feature_importance(
    shap_explanation: shap.Explanation,
    feature_names: list,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Mean |SHAP value| per feature across all rows -- the standard global
    importance ranking. Higher = more influence on the model's churn
    prediction across the customer base (direction-agnostic).
    """
    values = shap_explanation.values
    mean_abs = np.abs(values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance.head(top_n)


def plot_global_summary(
    shap_explanation: shap.Explanation,
    X_transformed: np.ndarray,
    feature_names: list,
    save_path: str = None,
    max_display: int = 20,
) -> str:
    """Saves a SHAP beeswarm summary plot. Returns the file path written."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = save_path or os.path.join(FIGURES_DIR, "shap_summary.png")

    plt.figure()
    shap.summary_plot(
        shap_explanation.values,
        features=X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# --------------------------------------------------
# Local (per-customer) explainability
# --------------------------------------------------

def explain_customer(
    shap_explanation: shap.Explanation,
    row_index: int,
    feature_names: list,
    customer_id: str = None,
    top_n: int = 10,
) -> dict:
    """
    Local explanation for one row already present in a computed
    shap_explanation batch. Returns the top positive (churn-driving) and
    top negative (retention-driving) features for that customer, plus the
    model's predicted probability reconstructed from base_value + shap sum.
    """
    row_values = shap_explanation.values[row_index]
    base_value = shap_explanation.base_values[row_index]
    if np.ndim(base_value) > 0:  # some SHAP/XGBoost versions return an array
        base_value = float(np.ravel(base_value)[0])

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": row_values,
    })

    top_positive = (
        contributions[contributions["shap_value"] > 0]
        .sort_values("shap_value", ascending=False)
        .head(top_n)
    )
    top_negative = (
        contributions[contributions["shap_value"] < 0]
        .sort_values("shap_value", ascending=True)
        .head(top_n)
    )

    predicted_logit = base_value + row_values.sum()
    predicted_probability = 1 / (1 + np.exp(-predicted_logit))

    return {
        "customer_id": customer_id,
        "predicted_churn_probability": round(float(predicted_probability), 4),
        "base_value_logit": round(float(base_value), 4),
        "top_risk_drivers": [
            {"feature": r.feature, "shap_value": round(float(r.shap_value), 4)}
            for r in top_positive.itertuples()
        ],
        "top_retention_drivers": [
            {"feature": r.feature, "shap_value": round(float(r.shap_value), 4)}
            for r in top_negative.itertuples()
        ],
    }


def plot_customer_waterfall(
    shap_explanation: shap.Explanation,
    row_index: int,
    customer_id: str,
    save_path: str = None,
    max_display: int = 12,
) -> str:
    """Saves a SHAP waterfall plot for a single customer. Returns the file path."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    safe_id = str(customer_id).replace("/", "_") if customer_id is not None else str(row_index)
    save_path = save_path or os.path.join(FIGURES_DIR, f"shap_waterfall_{safe_id}.png")

    plt.figure()
    shap.plots.waterfall(shap_explanation[row_index], max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def verify_shap_consistency(model, shap_explanation: shap.Explanation, X_transformed: np.ndarray, row_index: int = 0):
    """
    Sanity check: base_value + sum(shap values) (sigmoid-transformed) should
    match model.predict_proba() for the same row, up to floating point noise.
    Useful the first time you wire this up, or after an XGBoost/SHAP version bump.
    """
    reconstructed_logit = shap_explanation.base_values[row_index] + shap_explanation.values[row_index].sum()
    reconstructed_proba = 1 / (1 + np.exp(-reconstructed_logit))
    model_proba = model.predict_proba(X_transformed[row_index : row_index + 1])[0, 1]
    return {
        "reconstructed_from_shap": round(float(reconstructed_proba), 6),
        "model_predict_proba": round(float(model_proba), 6),
        "difference": round(float(abs(reconstructed_proba - model_proba)), 6),
    }


# --------------------------------------------------
# Retention strategy rules (simple, explainable lookup -- not a model)
# --------------------------------------------------
# Maps a SHAP risk-driver feature name to a plain-English retention action.
# Kept here (rather than a ML model) on purpose: retention offers need to be
# auditable and easy for a non-technical ops team to review/edit -- see
# src/recommendation/retention_rules.py for a place to grow this later into
# a full rules engine with eligibility/budget logic.
RETENTION_STRATEGY_RULES = {
    "Contract_Month-to-month": "Offer a discounted 1- or 2-year contract to lock in loyalty pricing.",
    "Tenure Months": "New/short-tenure customer -- enroll in an early-tenure onboarding program with a 90-day check-in call.",
    "Online Security_No": "Offer a free trial of the Online Security add-on.",
    "Tech Support_No": "Offer a free trial of the Tech Support add-on.",
    "Internet Service_Fiber optic": "Fiber customers are price-sensitive -- consider a fiber-tier loyalty discount.",
    "Payment Method_Electronic check": "Encourage a switch to autopay/credit-card billing with a small one-time incentive.",
    "Monthly Charges": "Review their plan for right-sizing or a loyalty discount -- bill size may be a pain point.",
    "Total Charges": "Long-tenure, high lifetime value -- route to a dedicated retention specialist for personal outreach.",
    "Paperless Billing_No": "Offer a small statement credit to switch to paperless billing.",
    "Multiple Lines_No": "Cross-sell a multi-line discount bundle.",
    "Streaming TV_No": "Offer a free trial of the Streaming TV / entertainment bundle.",
    "Streaming Movies_No": "Offer a free trial of the Streaming Movies bundle.",
    "Partner_No": "Consider a household/family plan offer.",
    "Dependents_No": "Consider a household/family plan offer.",
    "High Risk Contract": "Customer matches the month-to-month + electronic-check high-risk pattern -- prioritize a contract + autopay bundle offer.",
    "CLTV": "High lifetime value -- prioritize for white-glove retention outreach.",
}

_DEFAULT_STRATEGY = "No strong rule match -- flag for a general personalized retention call."


def suggest_retention_strategy(top_risk_drivers: list, max_suggestions: int = 3) -> list:
    """
    Turns a customer's top SHAP risk drivers into concrete retention actions.
    Matches driver feature names against RETENTION_STRATEGY_RULES (exact
    match first, then substring match for one-hot variants), de-duplicates,
    and falls back to a generic suggestion if nothing matches.
    """
    suggestions = []
    for driver in top_risk_drivers:
        feature = driver["feature"]
        action = RETENTION_STRATEGY_RULES.get(feature)
        if action is None:
            for rule_key, rule_action in RETENTION_STRATEGY_RULES.items():
                if rule_key in feature or feature in rule_key:
                    action = rule_action
                    break
        if action and action not in suggestions:
            suggestions.append(action)
        if len(suggestions) >= max_suggestions:
            break

    if not suggestions:
        suggestions = [_DEFAULT_STRATEGY]
    return suggestions


# --------------------------------------------------
# Single-customer lookup (interactive entry point)
# --------------------------------------------------

def explain_single_customer(
    customer_id: str,
    model,
    explainer: shap.TreeExplainer,
    df_raw: pd.DataFrame,
    preprocessor_path: str = PREPROCESSOR_PATH,
    save_waterfall: bool = True,
) -> dict:
    """
    Looks up ONE customer by CustomerID, computes their SHAP explanation,
    and attaches a rule-based retention strategy. Returns None (and prints
    a message) if the ID isn't found -- deliberately non-fatal so an
    interactive session can just try again instead of crashing.
    """
    match = df_raw[df_raw["CustomerID"] == customer_id]
    if match.empty:
        print(f"No customer found with ID '{customer_id}'. Check the ID and try again.")
        return None

    X_transformed, feature_names, _ = prepare_features(match, preprocessor_path=preprocessor_path)
    shap_explanation = compute_shap_values(explainer, X_transformed)
    explanation = explain_customer(shap_explanation, 0, feature_names, customer_id=customer_id)
    explanation["retention_strategy"] = suggest_retention_strategy(explanation["top_risk_drivers"])

    if save_waterfall:
        plot_path = plot_customer_waterfall(shap_explanation, 0, customer_id)
        explanation["waterfall_plot_path"] = plot_path

    return explanation


def print_customer_report(explanation: dict) -> None:
    """Pretty-prints a single customer's churn reasons + retention strategy to the console."""
    print("\n" + "=" * 60)
    print(f"INDIVIDUAL CUSTOMER REPORT -- {explanation['customer_id']}")
    print("=" * 60)
    print(f"Predicted churn probability: {explanation['predicted_churn_probability']:.2%}")

    print("\nTop reasons pushing this customer toward churn:")
    for d in explanation["top_risk_drivers"][:5]:
        print(f"  - {d['feature']}  (impact: +{d['shap_value']})")

    print("\nFactors currently helping retain this customer:")
    for d in explanation["top_retention_drivers"][:5]:
        print(f"  - {d['feature']}  (impact: {d['shap_value']})")

    print("\nSuggested retention strategy:")
    for i, action in enumerate(explanation["retention_strategy"], 1):
        print(f"  {i}. {action}")
    print("=" * 60 + "\n")


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading artifacts...")
    model, preprocessor = load_artifacts()
    explainer = get_explainer(model)

    print("Loading raw customer data...")
    df_raw = load_raw_data()

    # --- Interactive: look up ONE customer first, before the bulk report ---
    customer_id_input = input(
        "\nEnter a Customer ID to see their individual churn reason + "
        "retention strategy (press Enter to skip): "
    ).strip()
    if customer_id_input:
        single_explanation = explain_single_customer(customer_id_input, model, explainer, df_raw)
        if single_explanation:
            print_customer_report(single_explanation)
            print(f"Waterfall plot saved to: {single_explanation['waterfall_plot_path']}\n")

    print("Preparing full customer feature matrix...")
    X_transformed, feature_names, df_engineered = prepare_features(df_raw)
    print(f"Prepared feature matrix: {X_transformed.shape}")

    print("Computing SHAP values (this can take a minute on the full dataset)...")
    shap_explanation = compute_shap_values(explainer, X_transformed)

    # Sanity check on row 0
    check = verify_shap_consistency(model, shap_explanation, X_transformed, row_index=0)
    print(f"\nConsistency check (row 0): {check}")

    # --- Global importance ---
    importance = global_feature_importance(shap_explanation, feature_names, top_n=20)
    importance_path = os.path.join(REPORTS_DIR, "shap_global_importance.csv")
    importance.to_csv(importance_path, index=False)
    print(f"\nTop 20 global feature importances saved to: {importance_path}")
    print(importance.to_string(index=False))

    # --- Global summary plot ---
    summary_plot_path = plot_global_summary(shap_explanation, X_transformed, feature_names)
    print(f"\nSHAP summary plot saved to: {summary_plot_path}")

    # --- Local explanations for the 5 highest-risk customers ---
    probabilities = model.predict_proba(X_transformed)[:, 1]
    top_risk_idx = np.argsort(probabilities)[::-1][:5]

    local_explanations = []
    for idx in top_risk_idx:
        customer_id = df_engineered.iloc[idx].get("CustomerID", f"row_{idx}")
        explanation = explain_customer(shap_explanation, idx, feature_names, customer_id=customer_id)
        local_explanations.append(explanation)
        plot_customer_waterfall(shap_explanation, idx, customer_id)

    local_path = os.path.join(REPORTS_DIR, "shap_top5_risk_explanations.json")
    with open(local_path, "w") as f:
        json.dump(local_explanations, f, indent=2)
    print(f"\nTop 5 highest-risk customer explanations saved to: {local_path}")
    for exp in local_explanations:
        print(f"\nCustomer {exp['customer_id']} -- churn probability {exp['predicted_churn_probability']:.2%}")
        print("  Top risk drivers:", [d["feature"] for d in exp["top_risk_drivers"][:3]])
        print("  Top retention drivers:", [d["feature"] for d in exp["top_retention_drivers"][:3]])
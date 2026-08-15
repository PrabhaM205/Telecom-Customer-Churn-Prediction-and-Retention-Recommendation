import os
import joblib
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "xgboost_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "preprocessor.pkl"
)


# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================

def load_artifacts():

    print("Loading XGBoost model...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    preprocessor = None

    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(
            PREPROCESSOR_PATH
        )

    print("XGBoost model loaded.")

    if preprocessor is not None:
        print("Preprocessor loaded.")

    return model, preprocessor


# ============================================================
# SINGLE CUSTOMER PREDICTION
# ============================================================

def score_single_customer(
    customer_data,
    model=None,
    preprocessor=None
):

    # --------------------------------------------------------
    # Load artifacts if not provided
    # --------------------------------------------------------

    if model is None:

        model, loaded_preprocessor = load_artifacts()

        if preprocessor is None:
            preprocessor = loaded_preprocessor


    # --------------------------------------------------------
    # Convert customer dictionary to DataFrame
    # --------------------------------------------------------

    customer_df = pd.DataFrame(
        [customer_data]
    )


    # --------------------------------------------------------
    # Remove target / ID columns if present
    # --------------------------------------------------------

    columns_to_remove = [
        "CustomerID",
        "Churn Value",
        "Churn Label",
        "Churn Reason"
    ]

    X = customer_df.drop(
        columns=[
            col
            for col in columns_to_remove
            if col in customer_df.columns
        ],
        errors="ignore"
    )


    # --------------------------------------------------------
    # Apply preprocessing
    # --------------------------------------------------------

    if preprocessor is not None:

        X_processed = preprocessor.transform(X)

    else:

        X_processed = X


    # --------------------------------------------------------
    # XGBoost prediction
    # --------------------------------------------------------

    churn_probability = float(
        model.predict_proba(
            X_processed
        )[0][1]
    )


    # --------------------------------------------------------
    # Risk Tier
    # --------------------------------------------------------

    if churn_probability >= 0.70:

        risk_tier = "High"

    elif churn_probability >= 0.40:

        risk_tier = "Medium"

    else:

        risk_tier = "Low"


    # --------------------------------------------------------
    # Monthly Revenue At Risk
    # --------------------------------------------------------

    monthly_charges = customer_data.get(
        "Monthly Charges",
        0
    )

    try:

        monthly_charges = float(
            monthly_charges
        )

    except:

        monthly_charges = 0.0


    monthly_revenue_at_risk = (
        churn_probability *
        monthly_charges
    )


    # --------------------------------------------------------
    # Annual Revenue At Risk
    # --------------------------------------------------------

    annual_revenue_at_risk = (
        monthly_revenue_at_risk *
        12
    )


    # --------------------------------------------------------
    # Priority Score
    # --------------------------------------------------------

    priority_score = (
        churn_probability * 100
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "customer_id":
            customer_data.get(
                "CustomerID"
            ),

        "churn_probability":
            churn_probability,

        "risk_tier":
            risk_tier,

        "monthly_revenue_at_risk":
            round(
                monthly_revenue_at_risk,
                2
            ),

        "annual_revenue_at_risk":
            round(
                annual_revenue_at_risk,
                2
            ),

        "priority_score":
            round(
                priority_score,
                2
            )
    }


    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "XGBOOST PREDICTOR TEST"
    )

    print(
        "=" * 60
    )

    model, preprocessor = load_artifacts()

    print(
        "\nModel loaded successfully."
    )
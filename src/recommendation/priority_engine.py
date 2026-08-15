# src/recommendation/priority_engine.py

import os
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "telco_features.csv"
)

RISK_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "customer_risk_scores.csv"
)


# ============================================================
# CURRENCY CONFIGURATION
# ============================================================

CURRENCY = "USD"


# ============================================================
# RWCPS
# ============================================================

def compute_rwcps(
    churn_probability: float,
    revenue_at_risk_usd: float
) -> float:

    """
    Revenue-Weighted Churn Priority Score.

    Formula:

        RWCPS =
        Churn Probability × Revenue at Risk

    All monetary values are already in USD.
    """

    if not 0 <= churn_probability <= 1:

        raise ValueError(
            "Churn probability must be between 0 and 1."
        )

    if revenue_at_risk_usd < 0:

        raise ValueError(
            "Revenue at Risk cannot be negative."
        )

    return round(
        churn_probability
        * revenue_at_risk_usd,
        2
    )


# ============================================================
# PRIORITY LEVEL
# ============================================================

def get_priority_level(
    rwcps: float,
    max_rwcps: float
) -> str:

    """
    Classify customers according to
    their financial risk.

    Higher RWCPS = higher business priority.
    """

    if max_rwcps == 0:
        return "LOW"

    percentage = (
        rwcps / max_rwcps
    )

    if percentage >= 0.75:

        return "CRITICAL"

    elif percentage >= 0.50:

        return "HIGH"

    elif percentage >= 0.25:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# LOAD CUSTOMER DATA
# ============================================================

def load_customer_data():

    df = pd.read_csv(
        DATA_PATH
    )

    required_columns = [

        "CustomerID",

        "Monthly Charges",

        "CLTV"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Missing column in cleaned dataset: {column}"
            )

    return df


# ============================================================
# LOAD ML RISK OUTPUT
# ============================================================

def load_risk_data():

    """
    Load churn probability generated
    by the ML risk engine.
    """

    if not os.path.exists(
        RISK_PATH
    ):

        raise FileNotFoundError(
            "customer_risk_scores.csv not found.\n"
            "Ask Person 3 to run risk_engine.py first."
        )

    risk_df = pd.read_csv(
        RISK_PATH
    )

    required_columns = [

        "CustomerID",

        "churn_probability"
    ]

    for column in required_columns:

        if column not in risk_df.columns:

            raise ValueError(
                f"Missing column in risk output: {column}"
            )

    return risk_df


# ============================================================
# BUILD PRIORITY TABLE
# ============================================================

def build_priority_table():

    # --------------------------------------------------------
    # Load cleaned customer data
    # --------------------------------------------------------

    customer_df = load_customer_data()

    # --------------------------------------------------------
    # Load ML churn probability
    # --------------------------------------------------------

    risk_df = load_risk_data()

    # --------------------------------------------------------
    # Select required customer columns
    # --------------------------------------------------------

    customer_df = customer_df[
        [
            "CustomerID",
            "Monthly Charges",
            "CLTV"
        ]
    ]

    # --------------------------------------------------------
    # Select risk columns
    # --------------------------------------------------------

    risk_df = risk_df[
        [
            "CustomerID",
            "churn_probability"
        ]
    ]

    # --------------------------------------------------------
    # JOIN CUSTOMER + RISK DATA
    # --------------------------------------------------------

    df = customer_df.merge(

        risk_df,

        on="CustomerID",

        how="inner"
    )

    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # The cleaned dataset is ALREADY in USD.
    #
    # Therefore:
    #
    # DO NOT divide Monthly Charges by 83.
    # DO NOT divide CLTV by 83.
    #
    # ========================================================

    df["monthly_charges_usd"] = (
        df["Monthly Charges"]
        .astype(float)
        .round(2)
    )

    df["cltv_usd"] = (
        df["CLTV"]
        .astype(float)
        .round(2)
    )

    # --------------------------------------------------------
    # REVENUE AT RISK
    # --------------------------------------------------------
    #
    # Current business rule:
    #
    # Revenue at Risk = CLTV
    #
    # Since CLTV is already USD:
    #

    df["revenue_at_risk_usd"] = (
        df["cltv_usd"]
    )

    # --------------------------------------------------------
    # RWCPS
    # --------------------------------------------------------

    df["rwcps"] = df.apply(

        lambda row:

        compute_rwcps(

            row["churn_probability"],

            row["revenue_at_risk_usd"]

        ),

        axis=1
    )

    # --------------------------------------------------------
    # SORT BY BUSINESS PRIORITY
    # --------------------------------------------------------

    df = df.sort_values(

        by="rwcps",

        ascending=False

    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # PRIORITY RANK
    # --------------------------------------------------------

    df["priority_rank"] = (
        df.index + 1
    )

    # --------------------------------------------------------
    # MAX RWCPS
    # --------------------------------------------------------

    max_rwcps = df[
        "rwcps"
    ].max()

    # --------------------------------------------------------
    # PRIORITY LEVEL
    # --------------------------------------------------------

    df["priority_level"] = (

        df["rwcps"].apply(

            lambda x:

            get_priority_level(
                x,
                max_rwcps
            )
        )
    )

    return df


# ============================================================
# GET ONE CUSTOMER
# ============================================================

def get_customer_priority(
    customer_id: str
):

    df = build_priority_table()

    customer = df[
        df["CustomerID"]
        .astype(str)
        .str.strip()
        ==
        customer_id.strip()
    ]

    if customer.empty:

        raise ValueError(
            f"Customer '{customer_id}' not found."
        )

    return customer.iloc[0]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "CUSTOMER PRIORITY ENGINE"
    )

    print(
        "========================================"
    )

    print(
        "\nCurrency : USD"
    )

    print(
        "Dataset monetary values are already in USD."
    )

    print(
        "\nLoading cleaned customer data..."
    )

    print(
        "Loading ML risk output..."
    )

    try:

        df = build_priority_table()

        print(
            f"\nTotal customers evaluated : "
            f"{len(df)}"
        )

        # ----------------------------------------------------
        # TOP 10 CUSTOMERS
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "TOP 10 CUSTOMERS TO TARGET FIRST"
        )

        print(
            "========================================"
        )

        top10 = df.head(10)

        display_columns = [

            "priority_rank",

            "CustomerID",

            "churn_probability",

            "monthly_charges_usd",

            "cltv_usd",

            "revenue_at_risk_usd",

            "rwcps",

            "priority_level"
        ]

        display_df = top10[
            display_columns
        ].copy()

        display_df = display_df.rename(

            columns={

                "priority_rank":
                    "Priority Rank",

                "CustomerID":
                    "Customer ID",

                "churn_probability":
                    "Churn Probability",

                "monthly_charges_usd":
                    "Monthly Charges ($)",

                "cltv_usd":
                    "CLTV ($)",

                "revenue_at_risk_usd":
                    "Revenue at Risk ($)",

                "rwcps":
                    "RWCPS",

                "priority_level":
                    "Priority Level"
            }
        )

        # ----------------------------------------------------
        # Churn probability as percentage
        # ----------------------------------------------------

        display_df[
            "Churn Probability"
        ] = (

            display_df[
                "Churn Probability"
            ] * 100

        ).round(2)

        # ----------------------------------------------------
        # Monetary values
        # ----------------------------------------------------

        for column in [

            "Monthly Charges ($)",

            "CLTV ($)",

            "Revenue at Risk ($)",

            "RWCPS"

        ]:

            display_df[column] = (

                display_df[column]
                .round(2)

            )

        print(
            display_df.to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # PRIORITY DISTRIBUTION
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "PRIORITY DISTRIBUTION"
        )

        print(
            "========================================"
        )

        print(
            df[
                "priority_level"
            ].value_counts()
        )

        # ----------------------------------------------------
        # INDIVIDUAL CUSTOMER
        # ----------------------------------------------------

        print(
            "\n========================================"
        )

        print(
            "CHECK INDIVIDUAL CUSTOMER"
        )

        print(
            "========================================"
        )

        customer_id = input(

            "\nEnter Customer ID "
            "(or press Enter to skip): "

        ).strip()

        if customer_id:

            customer = (
                get_customer_priority(
                    customer_id
                )
            )

            print(
                "\n========================================"
            )

            print(
                "CUSTOMER PRIORITY DETAILS"
            )

            print(
                "========================================"
            )

            print(
                f"Customer ID       : "
                f"{customer['CustomerID']}"
            )

            print(
                f"Churn Probability : "
                f"{customer['churn_probability'] * 100:.2f}%"
            )

            print(
                f"Monthly Charges   : "
                f"${customer['monthly_charges_usd']:.2f}"
            )

            print(
                f"CLTV              : "
                f"${customer['cltv_usd']:.2f}"
            )

            print(
                f"Revenue at Risk   : "
                f"${customer['revenue_at_risk_usd']:.2f}"
            )

            print(
                f"RWCPS             : "
                f"${customer['rwcps']:.2f}"
            )

            print(
                f"Priority Rank     : "
                f"{int(customer['priority_rank'])}"
            )

            print(
                f"Priority Level    : "
                f"{customer['priority_level']}"
            )

            print(
                "========================================"
            )

    except FileNotFoundError as e:

        print(
            f"\nERROR: {e}"
        )

    except ValueError as e:

        print(
            f"\nERROR: {e}"
        )

    except Exception as e:

        print(
            f"\nERROR: {e}"
        )
# src/business/revenue_at_risk.py

import os
import pandas as pd


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "data",
    "processed",
    "telco_features.csv"
)


# ============================================================
# CURRENCY
# ============================================================

CURRENCY = "USD"


# ============================================================
# REVENUE AT RISK
# ============================================================

def compute_revenue_at_risk(
    monthly_charges: float,
    cltv: float
) -> float:
    """
    Calculate revenue exposure if the customer churns.

    Current business rule:

        Revenue at Risk = CLTV

    The cleaned dataset already contains
    monetary values in USD.
    """

    if monthly_charges < 0:

        raise ValueError(
            "Monthly Charges cannot be negative."
        )

    if cltv < 0:

        raise ValueError(
            "CLTV cannot be negative."
        )

    # CLTV is already in USD
    revenue_at_risk = float(cltv)

    return round(
        revenue_at_risk,
        2
    )


# ============================================================
# GET CUSTOMER
# ============================================================

def get_customer(
    customer_id: str
) -> dict:
    """
    Find one customer from the cleaned dataset.
    """

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found at:\n{DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Check CustomerID
    # --------------------------------------------------------

    if "CustomerID" not in df.columns:

        raise KeyError(
            "CustomerID column not found "
            "in telco_features.csv"
        )

    # --------------------------------------------------------
    # Find customer
    # --------------------------------------------------------

    customer = df[
        df["CustomerID"]
        .astype(str)
        .str.strip()
        ==
        customer_id.strip()
    ]

    if customer.empty:

        raise ValueError(
            f"Customer ID '{customer_id}' "
            f"not found."
        )

    return customer.iloc[0].to_dict()


# ============================================================
# CALCULATE CUSTOMER REVENUE
# ============================================================

def calculate_customer_revenue(
    customer_id: str
) -> dict:
    """
    Get customer data and calculate
    Revenue at Risk in USD.
    """

    customer = get_customer(
        customer_id
    )

    # --------------------------------------------------------
    # Values already in USD
    # --------------------------------------------------------

    monthly_charges = float(
        customer[
            "Monthly Charges"
        ]
    )

    cltv = float(
        customer[
            "CLTV"
        ]
    )

    # --------------------------------------------------------
    # Revenue at Risk
    # --------------------------------------------------------

    revenue_at_risk = (
        compute_revenue_at_risk(
            monthly_charges,
            cltv
        )
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "customer_id":
            customer_id,

        "monthly_charges":
            round(
                monthly_charges,
                2
            ),

        "cltv":
            round(
                cltv,
                2
            ),

        "revenue_at_risk":
            revenue_at_risk
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "REVENUE AT RISK ENGINE"
    )

    print(
        "========================================"
    )

    print(
        "Currency : USD"
    )

    print(
        "Dataset monetary values are already in USD."
    )

    customer_id = input(
        "\nEnter Customer ID: "
    ).strip()

    if not customer_id:

        print(
            "\nCustomer ID cannot be empty."
        )

        raise SystemExit


    try:

        result = (
            calculate_customer_revenue(
                customer_id
            )
        )

        print(
            "\n========================================"
        )

        print(
            "REVENUE AT RISK"
        )

        print(
            "========================================"
        )

        print(
            f"Customer ID     : "
            f"{result['customer_id']}"
        )

        print(
            f"Monthly Charges : "
            f"${result['monthly_charges']:.2f}"
        )

        print(
            f"CLTV            : "
            f"${result['cltv']:.2f}"
        )

        print(
            f"Revenue at Risk : "
            f"${result['revenue_at_risk']:.2f}"
        )

        print(
            "========================================"
        )


    except ValueError as e:

        print(
            f"\nERROR: {e}"
        )


    except KeyError as e:

        print(
            f"\nERROR: Required column "
            f"{e} was not found."
        )


    except FileNotFoundError as e:

        print(
            f"\nERROR: {e}"
        )


    except Exception as e:

        print(
            f"\nERROR: {e}"
        )
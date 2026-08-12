"""
scripts/eda.py

Day 1 exploratory data analysis for the Telco Customer Churn dataset.
Run this first, before any cleaning or feature engineering, to understand
what you're working with. Produces printed insights + saved chart images
under scripts/eda_output/.

Usage:
    python scripts/eda.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.loader import load_raw_data, basic_schema_check

OUTPUT_DIR = Path(__file__).resolve().parent / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid")


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_eda() -> pd.DataFrame:
    df = load_raw_data()
    basic_schema_check(df)

    # ---------- 1. Shape & schema ----------
    section("1. SHAPE & SCHEMA")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.dtypes)

    # ---------- 2. Missing values ----------
    section("2. MISSING VALUES")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    print(nulls if len(nulls) else "No nulls detected by pandas (but check blanks below).")

    # Total Charges is stored as text and has blank strings, not NaN —
    # pandas won't catch this in isnull(), so check explicitly.
    bad_total_charges = pd.to_numeric(df["Total Charges"], errors="coerce").isna().sum()
    print(f"\n'Total Charges' non-numeric / blank entries: {bad_total_charges}")
    blank_rows = df.loc[
        pd.to_numeric(df["Total Charges"], errors="coerce").isna(),
        ["CustomerID", "Tenure Months", "Total Charges"],
    ]
    print(blank_rows.head(11))
    print("--> All of these have Tenure Months = 0: brand-new customers with no")
    print("    billing history yet. This is NOT random missingness — it's structural.")

    # ---------- 3. Target distribution ----------
    section("3. TARGET DISTRIBUTION (Churn)")
    churn_counts = df["Churn Value"].value_counts()
    churn_rate = df["Churn Value"].mean()
    print(churn_counts)
    print(f"\nOverall churn rate: {churn_rate:.1%}")
    print("--> Confirms class imbalance (~26.5% positive class) — use PR-AUC,")
    print("    class weighting / SMOTE-Tomek, don't rely on plain accuracy.")

    # ---------- 4. Leakage risk columns ----------
    section("4. LEAKAGE-RISK COLUMNS — DO NOT USE AS MODEL FEATURES")
    print(df.groupby("Churn Value")["Churn Score"].describe()[["mean", "std", "min", "max"]])
    print(
        "\n'Churn Score' is IBM's own pre-computed churn likelihood score. "
        "Mean is 82.5 for churners vs 50.1 for non-churners — this is a proxy "
        "for the label itself. Using it as a training feature would be leakage: "
        "the model would learn to copy a score instead of learning real drivers."
    )
    print(f"\n'Churn Reason' is populated for {df['Churn Reason'].notna().sum()} rows — "
          f"exactly the {df['Churn Value'].sum()} churned customers, 0 for retained ones. "
          "This field only exists AFTER a customer has churned, so it is 100% leakage "
          "if used as a predictive feature. It IS very valuable later — for the GenAI "
          "layer, as real-world examples of churn drivers to ground offer generation.")
    print("\n'CLTV' is milder — distributions overlap between churners and non-churners "
          "(see below) — acceptable to keep as a business feature (revenue_at_risk.py), "
          "not a leakage concern in the same way.")
    print(df.groupby("Churn Value")["CLTV"].describe()[["mean", "std", "min", "max"]])

    # ---------- 5. Redundant / non-predictive columns ----------
    section("5. REDUNDANT / NON-PREDICTIVE COLUMNS")
    print(f"'Count' — constant value: {df['Count'].unique()}")
    print(f"'Country' — constant value: {df['Country'].unique()}")
    print(f"'State' — constant value: {df['State'].unique()}")
    print(f"'City' — {df['City'].nunique()} unique values (too high-cardinality "
          "for a 7K-row baseline model; drop for now, revisit if you add "
          "geo-based features later).")
    print("'Churn Label' duplicates 'Churn Value' (Yes/No vs 1/0) — keep only "
          "'Churn Value' as the numeric target.")
    print("'Lat Long' duplicates 'Latitude' + 'Longitude' — redundant, drop one.")
    print("'CustomerID' — unique identifier, not a predictive feature "
          "(keep for joins/traceability, exclude from model input).")

    # ---------- 6. Churn rate by key categorical drivers ----------
    section("6. CHURN RATE BY KEY SEGMENTS")
    for col in ["Contract", "Internet Service", "Payment Method", "Tech Support", "Online Security"]:
        rates = df.groupby(col)["Churn Value"].mean().sort_values(ascending=False)
        print(f"\n{col}:")
        print((rates * 100).round(1).astype(str) + "%")

    # ---------- 7. Numeric feature summary ----------
    section("7. NUMERIC FEATURE SUMMARY")
    numeric_cols = ["Tenure Months", "Monthly Charges"]
    print(df[numeric_cols].describe())

    # ---------- 8. Save charts ----------
    section("8. SAVING CHARTS")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    sns.countplot(data=df, x="Churn Label", ax=axes[0, 0])
    axes[0, 0].set_title("Churn Distribution")

    sns.histplot(data=df, x="Tenure Months", hue="Churn Label", bins=30,
                 multiple="stack", ax=axes[0, 1])
    axes[0, 1].set_title("Tenure vs Churn")

    contract_rate = (df.groupby("Contract")["Churn Value"].mean() * 100).sort_values()
    contract_rate.plot(kind="barh", ax=axes[1, 0], color="#2E74B5")
    axes[1, 0].set_title("Churn Rate % by Contract Type")
    axes[1, 0].set_xlabel("Churn Rate (%)")

    sns.boxplot(data=df, x="Churn Label", y="Monthly Charges", ax=axes[1, 1])
    axes[1, 1].set_title("Monthly Charges vs Churn")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "eda_summary.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")

    return df


if __name__ == "__main__":
    run_eda()
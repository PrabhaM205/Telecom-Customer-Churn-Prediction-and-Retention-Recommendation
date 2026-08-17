"""
api/main.py

FastAPI backend for the churn/retention frontend.

Features:
- Customer risk listing
- Customer ID search
- Risk tier filtering
- Contract type filtering
- Contract type dropdown options
- Pagination
- Customer profile
- Dashboard summary
- Offers trend
- Recommendations
- Accept / Dismiss recommendations
- CSV export
- Model metrics
- SHAP global importance
- AI Retention Assistant
- RAG + LLM retention offer generation
- Continuous Gmail complaint polling

Run:

    uvicorn api.main:app --reload --port 8000
"""

import json
import math
import re

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.complaints.complaint_handler import (
    COMPLAINTS_PATH,
    _load_json_store,
    start_email_polling_background,
    stop_email_polling,
)

from genai.offer_generator import (
    generate_retention_offer,
)

from api.schemas import (
    ChatRequest,
    ChatRole,
    CustomerProfile,
    CustomerRiskScore,
    DashboardSummary,
    ModelMetrics,
    OfferTrendPoint,
    OffersTrend,
    PaginatedResponse,
    Recommendation,
    RecommendationActionRequest,
    RecommendationStatus,
    RiskTier,
    RiskTierBreakdown,
    ShapFeatureContribution,
    ShapGlobalImportanceItem,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

RISK_SCORES_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_risk_scores.csv"
)

FEATURES_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_features.csv"
)

XGBOOST_METRICS_JSON = (
    PROJECT_ROOT
    / "artifacts"
    / "xgboost_metrics.json"
)

SHAP_GLOBAL_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "shap_global_importance.csv"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Churn Retention API",
    version="0.1.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# STARTUP / SHUTDOWN
# ============================================================

@app.on_event("startup")
def startup_event():

    print("\n" + "=" * 60)

    print(
        "FASTAPI BACKEND STARTED"
    )

    print(
        "Starting complaint email monitoring..."
    )

    print("=" * 60 + "\n")

    start_email_polling_background()


@app.on_event("shutdown")
def shutdown_event():

    print(
        "\n[main] "
        "FastAPI shutting down..."
    )

    stop_email_polling()

    print(
        "[main] "
        "Complaint polling stopped."
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# HELPERS
# ============================================================

def _clean_nan(value):

    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    return value


def _risk_tier_from_string(
    raw: str,
    fallback_probability: float = 0.0,
) -> RiskTier:

    if raw:

        normalized = (
            str(raw)
            .strip()
            .lower()
        )

        if normalized in (
            "high",
            "medium",
            "low",
        ):

            return RiskTier(
                normalized
            )

    if fallback_probability >= 0.66:

        return RiskTier.HIGH

    if fallback_probability >= 0.33:

        return RiskTier.MEDIUM

    return RiskTier.LOW


def _load_merged_customer_df() -> pd.DataFrame:

    if not RISK_SCORES_CSV.exists():

        raise HTTPException(
            status_code=503,
            detail=(
                "customer_risk_scores.csv "
                "not found at "
                f"{RISK_SCORES_CSV}. "
                "Run your training/scoring "
                "pipeline first."
            ),
        )

    risk_df = pd.read_csv(
        RISK_SCORES_CSV
    )

    if FEATURES_CSV.exists():

        features_df = pd.read_csv(
            FEATURES_CSV
        )

        merged = risk_df.merge(
            features_df,
            on="CustomerID",
            how="left",
            suffixes=(
                "",
                "_feat",
            ),
        )

    else:

        merged = risk_df

    return merged


def _row_to_customer_risk_score(
    row: dict,
) -> CustomerRiskScore:

    customer_id = str(
        row.get("CustomerID")
        or ""
    )

    churn_prob = float(
        _clean_nan(
            row.get(
                "churn_probability"
            )
        )
        or 0.0
    )

    return CustomerRiskScore(

        customer_id=customer_id,

        churn_probability=churn_prob,

        risk_tier=_risk_tier_from_string(
            row.get("risk_tier"),
            churn_prob,
        ),

        monthly_charges=_clean_nan(
            row.get(
                "Monthly Charges"
            )
        ),

        tenure_months=_clean_nan(
            row.get(
                "Tenure Months"
            )
        ),

        contract_type=_clean_nan(
            row.get(
                "Contract"
            )
        ),

        monthly_revenue_at_risk=_clean_nan(
            row.get(
                "monthly_revenue_at_risk"
            )
        ),

        annual_revenue_at_risk=_clean_nan(
            row.get(
                "annual_revenue_at_risk"
            )
        ),

        priority_score=_clean_nan(
            row.get(
                "priority_score"
            )
        ),
    )


# ============================================================
# OFFER PARSER
# ============================================================

def _parse_offer_recommendation(
    text: str,
) -> dict:

    pattern = re.compile(
        r"Recommended Offer:\s*"
        r"(?P<recommended_offer>.*?)"
        r"(?:\n\s*Why this offer:\s*"
        r"(?P<why_this_offer>.*?))?"
        r"(?:\n\s*Policy Basis:\s*"
        r"(?P<policy_basis>.*?))?"
        r"(?:\n\s*Customer Message:\s*"
        r"(?P<customer_message>.*))?$",
        re.IGNORECASE | re.DOTALL,
    )

    match = pattern.search(
        text or ""
    )

    sections = (
        match.groupdict()
        if match
        else {}
    )

    title = (
        sections.get(
            "recommended_offer"
        )
        or ""
    ).strip()

    title = (
        title.splitlines()[0].strip()
        if title
        else "AI-Generated Retention Offer"
    )

    description_parts = []

    if sections.get(
        "why_this_offer"
    ):

        description_parts.append(
            sections[
                "why_this_offer"
            ].strip()
        )

    if sections.get(
        "policy_basis"
    ):

        description_parts.append(
            "Policy: "
            + sections[
                "policy_basis"
            ].strip()
        )

    description = (
        " ".join(
            description_parts
        )
        if description_parts
        else (
            text or ""
        ).strip()[:500]
    )

    return {
        "title": title,
        "description": description,
    }


# ============================================================
# CUSTOMER FILTER
# ============================================================

def _filter_customer_dataframe(
    df: pd.DataFrame,
    risk_tier: Optional[RiskTier] = None,
    contract_type: Optional[str] = None,
    search: Optional[str] = None,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    if search:

        search_value = (
            search
            .strip()
            .lower()
        )

        if search_value:

            df = df[
                df["CustomerID"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_value,
                    na=False,
                    regex=False,
                )
            ]

    # --------------------------------------------------------
    # Contract
    # --------------------------------------------------------

    if contract_type:

        contract_value = (
            contract_type
            .strip()
            .lower()
        )

        if (
            contract_value
            and "Contract" in df.columns
        ):

            df = df[
                df["Contract"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq(
                    contract_value
                )
            ]

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if risk_tier is not None:

        calculated_tiers = []

        for row in df.to_dict(
            orient="records"
        ):

            churn_prob = float(
                _clean_nan(
                    row.get(
                        "churn_probability"
                    )
                )
                or 0.0
            )

            tier = (
                _risk_tier_from_string(
                    row.get(
                        "risk_tier"
                    ),
                    churn_prob,
                )
            )

            calculated_tiers.append(
                tier
            )

        if len(calculated_tiers) == len(df):

            mask = [
                tier == risk_tier
                for tier in calculated_tiers
            ]

            df = df.loc[
                mask
            ]

    return df


# ============================================================
# RECOMMENDATION STORE
# ============================================================

_RECS_STORE: dict = {}


# ============================================================
# CUSTOMERS
# ============================================================

@app.get(
    "/customers",
    response_model=PaginatedResponse[
        CustomerRiskScore
    ],
)
def list_customers(

    page: int = Query(
        1,
        ge=1,
    ),

    page_size: int = Query(
        25,
        ge=1,
        le=200,
    ),

    risk_tier: Optional[RiskTier] = None,

    contract_type: Optional[str] = None,

    search: Optional[str] = None,
):

    df = _load_merged_customer_df()

    df = _filter_customer_dataframe(
        df=df,
        risk_tier=risk_tier,
        contract_type=contract_type,
        search=search,
    )

    scores = [
        _row_to_customer_risk_score(
            row
        )
        for row in df.to_dict(
            orient="records"
        )
    ]

    total = len(scores)

    start = (
        page - 1
    ) * page_size

    end = (
        start
        + page_size
    )

    return PaginatedResponse(

        items=scores[
            start:end
        ],

        total=total,

        page=page,

        page_size=page_size,
    )


# ============================================================
# EXPORT
# ============================================================

@app.get(
    "/customers/export"
)
def export_customers_csv(

    risk_tier: Optional[RiskTier] = None,

    contract_type: Optional[str] = None,

    search: Optional[str] = None,
):

    df = _load_merged_customer_df()

    df = _filter_customer_dataframe(
        df=df,
        risk_tier=risk_tier,
        contract_type=contract_type,
        search=search,
    )

    if df.empty:

        csv_data = (
            "CustomerID,"
            "churn_probability,"
            "risk_tier,"
            "Monthly Charges,"
            "Tenure Months,"
            "Contract,"
            "monthly_revenue_at_risk,"
            "annual_revenue_at_risk,"
            "priority_score\n"
        )

    else:

        export_rows = []

        for row in df.to_dict(
            orient="records"
        ):

            churn_probability = float(
                _clean_nan(
                    row.get(
                        "churn_probability"
                    )
                )
                or 0.0
            )

            risk_tier_value = (
                _risk_tier_from_string(
                    row.get(
                        "risk_tier"
                    ),
                    churn_probability,
                ).value
            )

            export_rows.append({

                "CustomerID":
                    row.get(
                        "CustomerID"
                    ),

                "churn_probability":
                    churn_probability,

                "risk_tier":
                    risk_tier_value,

                "Monthly Charges":
                    _clean_nan(
                        row.get(
                            "Monthly Charges"
                        )
                    ),

                "Tenure Months":
                    _clean_nan(
                        row.get(
                            "Tenure Months"
                        )
                    ),

                "Contract":
                    _clean_nan(
                        row.get(
                            "Contract"
                        )
                    ),

                "monthly_revenue_at_risk":
                    _clean_nan(
                        row.get(
                            "monthly_revenue_at_risk"
                        )
                    ),

                "annual_revenue_at_risk":
                    _clean_nan(
                        row.get(
                            "annual_revenue_at_risk"
                        )
                    ),

                "priority_score":
                    _clean_nan(
                        row.get(
                            "priority_score"
                        )
                    ),
            })

        export_df = pd.DataFrame(
            export_rows
        )

        csv_data = export_df.to_csv(
            index=False
        )

    def csv_generator():

        yield csv_data

    timestamp = (
        datetime.now()
        .strftime("%Y-%m-%d")
    )

    return StreamingResponse(

        csv_generator(),

        media_type="text/csv",

        headers={
            "Content-Disposition":
                (
                    "attachment; "
                    f'filename="filtered-customers-{timestamp}.csv"'
                )
        },
    )


# ============================================================
# CONTRACT TYPES
# ============================================================

@app.get(
    "/customers/meta/contract-types",
    response_model=List[str],
)
def get_contract_types():

    df = _load_merged_customer_df()

    if "Contract" not in df.columns:

        return []

    contract_types = (
        df["Contract"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    return sorted([
        contract
        for contract
        in contract_types.unique()
        if contract
    ])


# ============================================================
# CUSTOMER PROFILE
# ============================================================

@app.get(
    "/customers/{customer_id}",
    response_model=CustomerProfile,
)
def get_customer_profile(
    customer_id: str,
):

    df = _load_merged_customer_df()

    match = df[
        df["CustomerID"]
        .astype(str)
        == customer_id
    ]

    if match.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Customer "
                f"{customer_id} "
                "not found"
            ),
        )

    row = (
        match
        .iloc[0]
        .to_dict()
    )

    base = (
        _row_to_customer_risk_score(
            row
        )
    )

    top_shap_features = []

    def _bool(value):

        value = _clean_nan(
            value
        )

        if value is None:
            return None

        return (
            str(value)
            .strip()
            .lower()
            == "yes"
        )

    return CustomerProfile(

        **base.model_dump(),

        gender=_clean_nan(
            row.get(
                "Gender"
            )
        ),

        partner=_bool(
            row.get(
                "Partner"
            )
        ),

        dependents=_bool(
            row.get(
                "Dependents"
            )
        ),

        internet_service=_clean_nan(
            row.get(
                "Internet Service"
            )
        ),

        payment_method=_clean_nan(
            row.get(
                "Payment Method"
            )
        ),

        top_shap_features=
            top_shap_features,
    )


# ============================================================
# DASHBOARD SUMMARY
# ============================================================

@app.get(
    "/dashboard/summary",
    response_model=DashboardSummary,
)
def get_dashboard_summary():

    df = _load_merged_customer_df()

    scores = [
        _row_to_customer_risk_score(
            row
        )
        for row in df.to_dict(
            orient="records"
        )
    ]

    total = len(scores)

    avg_prob = (
        sum(
            score.churn_probability
            for score in scores
        )
        / total
        if total
        else 0.0
    )

    total_monthly = sum(
        score.monthly_revenue_at_risk
        or 0.0
        for score in scores
    )

    total_annual = sum(
        score.annual_revenue_at_risk
        or 0.0
        for score in scores
    )

    tier_counts = {
        RiskTier.HIGH: 0,
        RiskTier.MEDIUM: 0,
        RiskTier.LOW: 0,
    }

    for score in scores:

        tier_counts[
            score.risk_tier
        ] += 1

    return DashboardSummary(

        total_customers=total,

        avg_churn_probability=round(
            avg_prob,
            4,
        ),

        total_monthly_revenue_at_risk=
            round(
                total_monthly,
                2,
            ),

        total_annual_revenue_at_risk=
            round(
                total_annual,
                2,
            ),

        risk_breakdown=[
            RiskTierBreakdown(
                tier=tier,
                count=count,
            )

            for tier, count
            in tier_counts.items()
        ],
    )


# ============================================================
# OFFERS TREND
# ============================================================

@app.get(
    "/dashboard/offers-trend",
    response_model=OffersTrend,
)
def get_offers_trend():

    buckets = defaultdict(
        lambda: {
            "generated": 0,
            "accepted": 0,
            "dismissed": 0,
        }
    )

    for rec in _RECS_STORE.values():

        day = (
            rec.created_at
            .date()
            .isoformat()
        )

        buckets[day][
            "generated"
        ] += 1

        if (
            rec.status
            == RecommendationStatus.ACCEPTED
        ):

            buckets[day][
                "accepted"
            ] += 1

        elif (
            rec.status
            == RecommendationStatus.DISMISSED
        ):

            buckets[day][
                "dismissed"
            ] += 1

    points = [
        OfferTrendPoint(
            date=day,
            **counts,
        )

        for day, counts
        in sorted(
            buckets.items()
        )
    ]

    return OffersTrend(
        points=points
    )


# ============================================================
# RECOMMENDATIONS
# ============================================================

@app.get(
    "/recommendations",
    response_model=PaginatedResponse[
        Recommendation
    ],
)
def list_recommendations(

    page: int = Query(
        1,
        ge=1,
    ),

    page_size: int = Query(
        25,
        ge=1,
        le=200,
    ),

    customer_id: Optional[str] = None,
):

    items = list(
        _RECS_STORE.values()
    )

    if customer_id:

        items = [
            recommendation
            for recommendation
            in items
            if recommendation.customer_id
            == customer_id
        ]

    total = len(items)

    start = (
        page - 1
    ) * page_size

    end = (
        start
        + page_size
    )

    return PaginatedResponse(

        items=items[
            start:end
        ],

        total=total,

        page=page,

        page_size=page_size,
    )


@app.post(
    "/recommendations/{recommendation_id}/action",
    response_model=Recommendation,
)
def act_on_recommendation(

    recommendation_id: str,

    action:
        RecommendationActionRequest,
):

    rec = _RECS_STORE.get(
        recommendation_id
    )

    if rec is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Recommendation "
                f"{recommendation_id} "
                "not found"
            ),
        )

    rec.status = action.status

    _RECS_STORE[
        recommendation_id
    ] = rec

    return rec


# ============================================================
# MODEL METRICS
# ============================================================

@app.get(
    "/models/metrics",
    response_model=ModelMetrics,
)
def get_model_metrics():

    if not XGBOOST_METRICS_JSON.exists():

        raise HTTPException(
            status_code=503,
            detail=(
                f"{XGBOOST_METRICS_JSON} "
                "not found"
            ),
        )

    try:

        with open(
            XGBOOST_METRICS_JSON,
            "r",
            encoding="utf-8",
        ) as f:

            raw = json.load(f)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read model "
                f"metrics: {str(exc)}"
            ),
        )

    return ModelMetrics(

        model_name=(
            raw.get("model")
            or "XGBoost"
        ),

        accuracy=raw.get(
            "accuracy"
        ),

        precision=raw.get(
            "precision"
        ),

        recall=raw.get(
            "recall"
        ),

        f1_score=(
            raw.get("f1_score")
            if raw.get("f1_score")
            is not None
            else raw.get("f1")
        ),

        auc=(
            raw.get("roc_auc")
            if raw.get("roc_auc")
            is not None
            else raw.get("auc")
        ),

        threshold=raw.get(
            "threshold"
        ),

        n_train_samples=
            raw.get(
                "n_train_samples"
            ),

        n_test_samples=
            raw.get(
                "n_test_samples"
            ),

        n_features=
            raw.get(
                "n_features"
            ),

        trained_at=
            raw.get(
                "trained_at"
            ),
    )


# ============================================================
# SHAP
# ============================================================

@app.get(
    "/models/shap/global",
    response_model=List[
        ShapGlobalImportanceItem
    ],
)
def get_shap_global_importance():

    if not SHAP_GLOBAL_CSV.exists():

        raise HTTPException(
            status_code=503,
            detail=(
                f"{SHAP_GLOBAL_CSV} "
                "not found"
            ),
        )

    try:

        df = pd.read_csv(
            SHAP_GLOBAL_CSV
        )

        if df.empty:
            return []

        if "feature" not in df.columns:

            raise HTTPException(
                status_code=500,
                detail=(
                    "SHAP CSV must contain "
                    "'feature'."
                ),
            )

        shap_column = None

        possible_columns = [
            "mean_abs_shap",
            "importance",
            "mean_abs_shap_value",
        ]

        for column in possible_columns:

            if column in df.columns:

                shap_column = column

                break

        if shap_column is None:

            raise HTTPException(
                status_code=500,
                detail=(
                    "SHAP CSV must contain "
                    "a SHAP importance column."
                ),
            )

        df["feature"] = (
            df["feature"]
            .astype(str)
            .str.strip()
        )

        df[shap_column] = pd.to_numeric(
            df[shap_column],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "feature",
                shap_column,
            ]
        )

        df = df.sort_values(
            by=shap_column,
            ascending=False,
        )

        result = []

        for row in df.to_dict(
            orient="records"
        ):

            result.append(
                ShapGlobalImportanceItem(

                    feature=str(
                        row["feature"]
                    ),

                    mean_abs_shap=float(
                        row[shap_column]
                    ),
                )
            )

        return result

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load SHAP "
                f"global importance: "
                f"{str(exc)}"
            ),
        )


# ============================================================
# SSE
# ============================================================

def _sse(event: dict) -> str:

    return (
        "data: "
        f"{json.dumps(event)}"
        "\n\n"
    )


# ============================================================
# AI RETENTION ASSISTANT
# ============================================================

@app.post(
    "/assistant/chat"
)
def chat_with_assistant(
    request: ChatRequest,
):

    def event_generator():

        # ----------------------------------------------------
        # NO CUSTOMER ID
        # ----------------------------------------------------

        if not request.customer_id:

            last_user_msg = next(
                (
                    message.content
                    for message
                    in reversed(
                        request.messages
                    )
                    if message.role
                    == ChatRole.USER
                ),
                "",
            )

            yield _sse({

                "role":
                    "assistant",

                "node":
                    "orchestrator",

                "content":
                    (
                        f"You said: "
                        f"{last_user_msg!r}. "
                        "Please mention a "
                        "customer ID to "
                        "generate a "
                        "retention offer."
                    ),
            })

            yield (
                "data: [DONE]\n\n"
            )

            return

        # ----------------------------------------------------
        # LOAD CUSTOMER
        # ----------------------------------------------------

        df = (
            _load_merged_customer_df()
        )

        match = df[
            df["CustomerID"]
            .astype(str)
            == request.customer_id
        ]

        if match.empty:

            yield _sse({

                "role":
                    "assistant",

                "node":
                    "orchestrator",

                "content":
                    (
                        f"I couldn't find "
                        f"customer "
                        f"{request.customer_id!r} "
                        "in the system."
                    ),
            })

            yield (
                "data: [DONE]\n\n"
            )

            return

        # ----------------------------------------------------
        # CUSTOMER SCORE
        # ----------------------------------------------------

        row = (
            match
            .iloc[0]
            .to_dict()
        )

        score = (
            _row_to_customer_risk_score(
                row
            )
        )

        yield _sse({

            "role":
                "assistant",

            "node":
                "diagnosis",

            "content":
                (
                    "Analyzing churn risk "
                    f"for "
                    f"{score.customer_id}…"
                ),
        })

        # ----------------------------------------------------
        # GENERATE OFFER
        # ----------------------------------------------------

        try:

            offer_result = (
                generate_retention_offer(

                    customer_data=row,

                    churn_probability=(
                        score.churn_probability
                    ),

                    customer_segment=
                        "Standard",
                )
            )

        except Exception as exc:

            yield _sse({

                "role":
                    "assistant",

                "node":
                    "offer_strategist",

                "content":
                    (
                        "I was unable to "
                        "generate the retention "
                        "offer right now. "
                        f"Error: {str(exc)}"
                    ),
            })

            yield (
                "data: [DONE]\n\n"
            )

            return

        # ----------------------------------------------------
        # PARSE OFFER
        # ----------------------------------------------------

        offer = (
            _parse_offer_recommendation(
                offer_result[
                    "recommendation"
                ]
            )
        )

        # ----------------------------------------------------
        # SAVE RECOMMENDATION
        # ----------------------------------------------------

        rec_id = (
            f"rec-"
            f"{score.customer_id}-"
            f"{len(_RECS_STORE) + 1}"
        )

        recommendation = Recommendation(

            id=rec_id,

            customer_id=(
                score.customer_id
            ),

            offer_type=(
                "ai_generated"
            ),

            title=offer[
                "title"
            ],

            description=offer[
                "description"
            ],

            priority_score=min(
                (
                    score.priority_score
                    or 0.0
                ) / 100,
                1.0,
            ),

            expected_revenue_impact=(
                score.annual_revenue_at_risk
            ),

            status=(
                RecommendationStatus.PENDING
            ),

            created_at=datetime.utcnow(),
        )

        _RECS_STORE[
            rec_id
        ] = recommendation

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        reply_text = (
            "I've generated a "
            "retention offer for "
            f"{score.customer_id} "
            "(churn risk "
            f"{score.churn_probability:.0%}).\n\n"
            f"{offer_result['recommendation']}"
        )

        yield _sse({

            "role":
                "assistant",

            "node":
                "offer_strategist",

            "content":
                reply_text,
        })

        yield (
            "data: [DONE]\n\n"
        )

    return StreamingResponse(

        event_generator(),

        media_type=
            "text/event-stream",
    )


# ============================================================
# COMPLAINTS
# ============================================================

@app.get(
    "/complaints"
)
def get_complaints():

    complaints = (
        _load_json_store(
            COMPLAINTS_PATH
        )
    )

    return list(
        complaints.values()
    )


# ============================================================
# CUSTOMER-SPECIFIC COMPLAINTS
# ============================================================

@app.get(
    "/complaints/{customer_id}"
)
def get_customer_complaints(
    customer_id: str,
):

    complaints = (
        _load_json_store(
            COMPLAINTS_PATH
        )
    )

    customer_complaints = [

        complaint

        for complaint
        in complaints.values()

        if str(
            complaint.get(
                "customer_id"
            )
        ).strip().upper()
        ==
        customer_id.strip().upper()
    ]

    return {

        "customer_id":
            customer_id,

        "total":
            len(
                customer_complaints
            ),

        "complaints":
            customer_complaints,
    }
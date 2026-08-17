"""
api/schemas.py

Pydantic response/request models. These are the CONTRACT between the FastAPI
backend and the React frontend — keep frontend/src/api/types.ts in sync with
whatever you change here.
"""

from datetime import datetime
from enum import Enum
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared / generic
# ---------------------------------------------------------------------------

class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int


# ---------------------------------------------------------------------------
# Customers / risk
# ---------------------------------------------------------------------------

class CustomerRiskScore(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    risk_tier: RiskTier
    monthly_charges: Optional[float] = None
    tenure_months: Optional[int] = None
    contract_type: Optional[str] = None
    monthly_revenue_at_risk: Optional[float] = None
    annual_revenue_at_risk: Optional[float] = None
    priority_score: Optional[float] = None


class ShapFeatureContribution(BaseModel):
    feature: str
    value: float          # the feature's actual value for this customer
    shap_value: float      # contribution to the prediction (+/-)


class CustomerProfile(CustomerRiskScore):
    # Extended detail for the single-customer view
    gender: Optional[str] = None
    partner: Optional[bool] = None
    dependents: Optional[bool] = None
    internet_service: Optional[str] = None
    payment_method: Optional[str] = None
    top_shap_features: List[ShapFeatureContribution] = []


# ---------------------------------------------------------------------------
# Recommendations / offers
# ---------------------------------------------------------------------------

class RecommendationStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"


class Recommendation(BaseModel):
    id: str
    customer_id: str
    offer_type: str
    title: str
    description: str
    priority_score: float = Field(..., ge=0, le=1)
    expected_revenue_impact: Optional[float] = None
    status: RecommendationStatus = RecommendationStatus.PENDING
    created_at: datetime


class RecommendationActionRequest(BaseModel):
    status: RecommendationStatus


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------

class ModelMetrics(BaseModel):
    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    threshold: Optional[float] = None
    n_train_samples: Optional[int] = None
    n_test_samples: Optional[int] = None
    n_features: Optional[int] = None
    trained_at: Optional[str] = None


class ShapGlobalImportanceItem(BaseModel):
    feature: str
    mean_abs_shap: float


# ---------------------------------------------------------------------------
# Executive Dashboard
# ---------------------------------------------------------------------------

class RiskTierBreakdown(BaseModel):
    tier: RiskTier
    count: int


class DashboardSummary(BaseModel):
    total_customers: int
    avg_churn_probability: float
    total_monthly_revenue_at_risk: float
    total_annual_revenue_at_risk: float
    risk_breakdown: List[RiskTierBreakdown]


class OfferTrendPoint(BaseModel):
    date: str  # ISO date, e.g. "2026-08-17"
    generated: int
    accepted: int
    dismissed: int


class OffersTrend(BaseModel):
    points: List[OfferTrendPoint]


# ---------------------------------------------------------------------------
# AI Retention Assistant (agent)
# ---------------------------------------------------------------------------

class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    customer_id: str  # required — the retention agent runs per-customer, not free-text chat


class ChatResponse(BaseModel):
    message: ChatMessage
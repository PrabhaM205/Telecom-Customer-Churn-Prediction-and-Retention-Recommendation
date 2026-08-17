// Wire-format types mirroring api/schemas.py — keep in sync with the backend.
export type RecommendationStatus = "pending" | "accepted" | "dismissed";
export type RiskTier = "low" | "medium" | "high";

export interface CustomerRiskScore {
  customer_id: string;
  churn_probability: number;
  risk_tier: RiskTier;
  monthly_charges: number | null;
  tenure_months: number | null;
  contract_type: string | null;
  monthly_revenue_at_risk: number | null;
  annual_revenue_at_risk: number | null;
  priority_score: number | null;
}

export interface ShapFeatureContribution {
  feature: string;
  value: number;
  shap_value: number;
}

export interface CustomerProfile extends CustomerRiskScore {
  gender: string | null;
  partner: boolean | null;
  dependents: boolean | null;
  internet_service: string | null;
  payment_method: string | null;
  top_shap_features: ShapFeatureContribution[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface RiskTierBreakdown {
  tier: RiskTier;
  count: number;
}

export interface DashboardSummary {
  total_customers: number;
  avg_churn_probability: number;
  total_monthly_revenue_at_risk: number;
  total_annual_revenue_at_risk: number;
  risk_breakdown: RiskTierBreakdown[];
}

export interface ModelMetrics {
  model_name: string;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  auc: number | null;
  threshold: number | null;
  n_train_samples: number | null;
  n_test_samples: number | null;
  n_features: number | null;
  trained_at: string | null;
}

export interface ShapGlobalImportanceItem {
  feature: string;
  mean_abs_shap: number;
}

export interface Recommendation {
  id: string;
  customer_id: string;
  offer_type: string;
  title: string;
  description: string;
  priority_score: number;
  expected_revenue_impact: number | null;
  status: RecommendationStatus;
  created_at: string;
}

export interface RecommendationActionRequest {
  status: RecommendationStatus;
}

// ---------------------------------------------------------------------------
// AI Retention Assistant (agent)
// ---------------------------------------------------------------------------

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  role: ChatRole;
  content: string;
  // Present on the assistant message that actually generated an offer —
  // lets ChatMessage.tsx render the same RecommendationCard used on the
  // Retention Recommendations page, inline in the conversation.
  recommendation?: Recommendation;
}

export interface ChatRequest {
  messages: ChatMessage[];
  customer_id: string; // required — the retention agent runs per-customer, not free-text chat
}

// One parsed SSE payload from POST /assistant/chat. Distinct from ChatMessage
// because it carries which graph node produced it (diagnosis / offer_strategist
// / guardrail / orchestrator) — useful for the UI to show progress/step context,
// not just the message text.
export interface AgentStreamEvent {
  role: ChatRole;
  content: string;
  node: "diagnosis" | "offer_strategist" | "guardrail" | "orchestrator";
  // Only present on the offer_strategist event — the offer that was just
  // generated and saved to the recommendations store.
  recommendation?: Recommendation;
}
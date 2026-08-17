export const RISK_TIERS = {
  LOW: { label: "Low", color: "risk-low", threshold: 0.33 },
  MEDIUM: { label: "Medium", color: "risk-medium", threshold: 0.66 },
  HIGH: { label: "High", color: "risk-high", threshold: 1.0 },
} as const;

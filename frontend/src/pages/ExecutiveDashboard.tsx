// frontend/src/pages/ExecutiveDashboard.tsx

import { useDashboardSummary } from "@/hooks/useDashboardSummary";
import { useOffersTrend } from "@/hooks/useOffersTrend";

import KpiGrid from "@/components/kpi/KpiGrid";
import KpiCard from "@/components/kpi/KpiCard";

import RiskDistributionChart from "@/components/charts/RiskDistributionChart";
import OffersTrendChart from "@/components/charts/OffersTrendChart";

import { formatCurrency } from "@/lib/utils";

export default function ExecutiveDashboard() {
  const {
    data,
    isLoading,
    isError,
    error,
  } = useDashboardSummary();

  const {
    data: offersTrend,
    isLoading: isOffersTrendLoading,
    isError: isOffersTrendError,
  } = useOffersTrend();

  return (
    <div className="space-y-6">

      {/* ============================================================
          DASHBOARD SUMMARY
          ============================================================ */}

      {isLoading && (
        <div className="rounded-lg border border-border bg-panel py-12 text-center text-sm text-ink-muted">
          Loading summary…
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-accent-rose/30 bg-accent-rose/10 p-4 text-sm text-accent-rose">
          Failed to load dashboard summary:{" "}
          {(error as Error)?.message ??
            "unknown error"}
        </div>
      )}

      {data && (
        <>
          {/* ========================================================
              KPI CARDS
              ======================================================== */}

          <KpiGrid>

            <KpiCard
              label="Total Customers"
              value={data.total_customers.toLocaleString()}
              tone="cyan"
            />

            <KpiCard
              label="Avg. Churn Probability"
              value={`${(
                data.avg_churn_probability * 100
              ).toFixed(1)}%`}
              tone="amber"
            />

            <KpiCard
              label="Monthly Revenue at Risk"
              value={formatCurrency(
                data.total_monthly_revenue_at_risk
              )}
              tone="rose"
            />

            <KpiCard
              label="Annual Revenue at Risk"
              value={formatCurrency(
                data.total_annual_revenue_at_risk
              )}
              tone="rose"
            />

          </KpiGrid>

          {/* ========================================================
              RISK DISTRIBUTION
              ======================================================== */}

          <div className="rounded-lg border border-border bg-panel p-5">
            <RiskDistributionChart
              data={data.risk_breakdown}
            />
          </div>

          {/* ========================================================
              OFFERS TREND
              ======================================================== */}

          <div className="rounded-lg border border-border bg-panel p-5">

            {isOffersTrendLoading && (
              <div className="flex h-[320px] items-center justify-center text-sm text-ink-muted">
                Loading offers trend…
              </div>
            )}

            {isOffersTrendError && (
              <div className="flex h-[320px] items-center justify-center text-sm text-accent-rose">
                Failed to load offers trend.
              </div>
            )}

            {offersTrend && (
              <OffersTrendChart
                data={offersTrend.points}
              />
            )}

          </div>

        </>
      )}

    </div>
  );
}
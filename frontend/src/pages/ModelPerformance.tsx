import KpiCard from "@/components/kpi/KpiCard";
import KpiGrid from "@/components/kpi/KpiGrid";
import ShapSummaryChart from "@/components/charts/ShapSummaryChart";
import { useModelMetrics, useShapGlobalImportance } from "@/hooks/useModelMetrics";

function formatPercent(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | null): string {
  return value == null ? "—" : value.toLocaleString();
}

export default function ModelPerformance() {
  const { data: metrics, isLoading: metricsLoading, isError: metricsError } = useModelMetrics();
  const { data: shapData, isLoading: shapLoading, isError: shapError } = useShapGlobalImportance();

  return (
    <div className="space-y-6">
      {metrics && (
        <div className="eyebrow text-[10px]">
          {metrics.model_name}
          {metrics.threshold != null && ` · threshold ${metrics.threshold}`}
          {metrics.trained_at && ` · trained ${metrics.trained_at}`}
        </div>
      )}

      {metricsError && (
        <div className="rounded-lg border border-accent-rose/30 bg-accent-rose/10 p-4 text-sm text-accent-rose">
          Failed to load model metrics.
        </div>
      )}

      {!metricsError && (
        <KpiGrid>
          <KpiCard label="Accuracy" value={metricsLoading ? "…" : formatPercent(metrics?.accuracy ?? null)} tone="cyan" />
          <KpiCard label="Precision" value={metricsLoading ? "…" : formatPercent(metrics?.precision ?? null)} tone="cyan" />
          <KpiCard label="Recall" value={metricsLoading ? "…" : formatPercent(metrics?.recall ?? null)} tone="cyan" />
          <KpiCard label="F1 Score" value={metricsLoading ? "…" : formatPercent(metrics?.f1_score ?? null)} tone="cyan" />
          <KpiCard label="ROC AUC" value={metricsLoading ? "…" : formatPercent(metrics?.auc ?? null)} tone="violet" />
          <KpiCard
            label="Train / Test Samples"
            value={
              metricsLoading
                ? "…"
                : `${formatNumber(metrics?.n_train_samples ?? null)} / ${formatNumber(metrics?.n_test_samples ?? null)}`
            }
          />
          <KpiCard label="Features Used" value={metricsLoading ? "…" : formatNumber(metrics?.n_features ?? null)} />
        </KpiGrid>
      )}

      {shapError && (
        <div className="rounded-lg border border-accent-rose/30 bg-accent-rose/10 p-4 text-sm text-accent-rose">
          Failed to load SHAP feature importance.
        </div>
      )}
      {!shapError && !shapLoading && shapData && (
        <div className="rounded-lg border border-border bg-panel p-5">
          <ShapSummaryChart data={shapData} />
        </div>
      )}
    </div>
  );
}

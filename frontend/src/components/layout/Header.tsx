import { useLocation } from "react-router-dom";

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  "/": {
    title: "Executive Dashboard",
    subtitle: "Churn risk and revenue exposure, at a glance.",
  },
  "/customer-risk": {
    title: "Customer Risk",
    subtitle: "Every customer, scored and ranked by churn probability.",
  },
  "/recommendations": {
    title: "Retention Recommendations",
    subtitle: "AI-suggested actions, ranked by expected impact.",
  },
  "/model-performance": {
    title: "Model Performance",
    subtitle: "How the churn model is doing, and why it makes the calls it does.",
  },
  "/assistant": {
    title: "AI Retention Assistant",
    subtitle: "Ask questions about customers, risk, and retention strategy.",
  },
};

export default function Header() {
  const location = useLocation();
  const meta = PAGE_META[location.pathname] ?? { title: "", subtitle: "" };

  return (
    <header className="flex shrink-0 items-center justify-between border-b border-border bg-panel px-6 py-4">
      <div>
        <h1 className="text-lg font-semibold text-ink">{meta.title}</h1>
        {meta.subtitle && <p className="mt-0.5 text-sm text-ink-muted">{meta.subtitle}</p>}
      </div>
      <div className="hidden shrink-0 items-center gap-2 rounded-full border border-border-subtle bg-panel-raised/60 px-3 py-1.5 md:flex">
        <span className="h-1.5 w-1.5 rounded-full bg-accent-amber" />
        <span className="text-[11px] text-ink-muted">Predictions are advisory — verify before acting</span>
      </div>
    </header>
  );
}

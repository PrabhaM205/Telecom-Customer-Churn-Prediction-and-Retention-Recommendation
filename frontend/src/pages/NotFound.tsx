import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="flex h-full flex-col items-center justify-center rounded-lg border border-border bg-panel py-24 text-center">
      <div className="eyebrow text-[10px]">Error 404</div>
      <h1 className="mt-2 text-2xl font-semibold text-ink">Route not found</h1>
      <p className="mt-2 max-w-sm text-sm text-ink-muted">
        That page doesn't exist in the console. Check the URL, or head back to the dashboard.
      </p>
      <Link
        to="/"
        className="mt-6 rounded-md border border-border bg-panel-raised px-4 py-2 text-sm font-medium text-ink hover:border-accent-cyan/40 hover:text-accent-cyan"
      >
        Back to dashboard
      </Link>
    </div>
  );
}

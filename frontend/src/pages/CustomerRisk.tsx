import { useEffect, useState } from "react";
import { useCustomerRisk } from "@/hooks/useCustomerRisk";
import { useFilterStore } from "@/store/useFilterStore";
import CustomerFilters from "@/components/customer/CustomerFilters";
import { formatCurrency } from "@/lib/utils";
import Badge from "@/components/ui/badge";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import type { RiskTier } from "@/api/types";

const RISK_BADGE_VARIANT: Record<RiskTier, "danger" | "warning" | "success"> = {
  high: "danger",
  medium: "warning",
  low: "success",
};

export default function CustomerRisk() {
  const [page, setPage] = useState(1);
  const pageSize = 25;

  const { riskTier, contractType, search } = useFilterStore();

  // Any filter change should reset back to page 1 — otherwise you can land
  // on a page number that no longer exists for the new filtered result set.
  useEffect(() => {
    setPage(1);
  }, [riskTier, contractType, search]);

  const { data, isLoading, isError, error } = useCustomerRisk({
    page,
    pageSize,
    riskTier,
    contractType,
    search,
  });

  const totalPages = data ? Math.max(1, Math.ceil(data.total / pageSize)) : 1;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="eyebrow text-[10px]">
          {data ? `${data.total.toLocaleString()} customers` : "Loading customers…"}
        </div>
        <CustomerFilters />
      </div>

      {isLoading && (
        <div className="rounded-lg border border-border bg-panel py-12 text-center text-sm text-ink-muted">
          Loading risk scores…
        </div>
      )}

      {isError && (
        <div className="rounded-lg border border-accent-rose/30 bg-accent-rose/10 p-4 text-sm text-accent-rose">
          Failed to load customers: {(error as Error)?.message ?? "unknown error"}
          <div className="mt-1 text-xs text-accent-rose/70">
            Is the API running at the address in VITE_API_BASE_URL?
          </div>
        </div>
      )}

      {data && (
        <>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Customer ID</TableHead>
                <TableHead>Risk Tier</TableHead>
                <TableHead>Churn Probability</TableHead>
                <TableHead>Contract</TableHead>
                <TableHead>Tenure (mo)</TableHead>
                <TableHead>Monthly Charges</TableHead>
                <TableHead>Annual Revenue at Risk</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.items.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="py-8 text-center text-ink-faint">
                    No customers match the current filters.
                  </TableCell>
                </TableRow>
              ) : (
                data.items.map((c) => (
                  <TableRow key={c.customer_id}>
                    <TableCell className="font-mono text-xs text-ink-muted">{c.customer_id}</TableCell>
                    <TableCell>
                      <Badge variant={RISK_BADGE_VARIANT[c.risk_tier]}>{c.risk_tier}</Badge>
                    </TableCell>
                    <TableCell>{(c.churn_probability * 100).toFixed(1)}%</TableCell>
                    <TableCell>{c.contract_type ?? "—"}</TableCell>
                    <TableCell>{c.tenure_months ?? "—"}</TableCell>
                    <TableCell>{c.monthly_charges != null ? formatCurrency(c.monthly_charges) : "—"}</TableCell>
                    <TableCell>
                      {c.annual_revenue_at_risk != null ? formatCurrency(c.annual_revenue_at_risk) : "—"}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>

          <div className="flex items-center justify-between text-sm">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="rounded-md border border-border px-3 py-1.5 text-ink-muted hover:bg-panel-raised disabled:opacity-40"
            >
              Previous
            </button>
            <span className="text-ink-muted">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="rounded-md border border-border px-3 py-1.5 text-ink-muted hover:bg-panel-raised disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}

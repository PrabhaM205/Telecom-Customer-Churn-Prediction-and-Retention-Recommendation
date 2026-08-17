import { useEffect, useState } from "react";
import { useCustomerRisk } from "@/hooks/useCustomerRisk";
import { useFilterStore } from "@/store/useFilterStore";
import CustomerFilters from "@/components/customer/CustomerFilters";
import { formatCurrency } from "@/lib/utils";
import Badge from "@/components/ui/badge";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import type { RiskTier } from "@/api/types";

const RISK_BADGE_VARIANT: Record<
  RiskTier,
  "danger" | "warning" | "success"
> = {
  high: "danger",
  medium: "warning",
  low: "success",
};

// ============================================================
// Complaint type
// ============================================================

interface Complaint {
  complaint_id: string;
  source: string;
  customer_id: string;
  contact_number: string | null;
  sender_email: string | null;
  subject: string;
  complaint_text: string;
  matched_keywords: string[];
  received_at: string;
  status: string;
}

export default function CustomerRisk() {
  const [page, setPage] = useState(1);

  const pageSize = 25;

  const { riskTier, contractType, search } = useFilterStore();

  // ============================================================
  // Complaints state
  // ============================================================

  const [complaints, setComplaints] = useState<Complaint[]>([]);
  const [complaintsLoading, setComplaintsLoading] = useState(true);
  const [complaintsError, setComplaintsError] = useState<string | null>(
    null
  );

  // ============================================================
  // Reset page when filters change
  // ============================================================

  useEffect(() => {
    setPage(1);
  }, [riskTier, contractType, search]);

  // ============================================================
  // Customer risk API
  // ============================================================

  const {
    data,
    isLoading,
    isError,
    error,
  } = useCustomerRisk({
    page,
    pageSize,
    riskTier,
    contractType,
    search,
  });

  // ============================================================
  // Fetch complaints
  // ============================================================

  useEffect(() => {
    const fetchComplaints = async () => {
      try {
        setComplaintsLoading(true);
        setComplaintsError(null);

        const apiBaseUrl =
          import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

        const response = await fetch(
          `${apiBaseUrl}/complaints`
        );

        if (!response.ok) {
          throw new Error(
            `Failed to fetch complaints (${response.status})`
          );
        }

        const result: Complaint[] = await response.json();

        setComplaints(result);
      } catch (err) {
        setComplaintsError(
          err instanceof Error
            ? err.message
            : "Failed to load complaints"
        );
      } finally {
        setComplaintsLoading(false);
      }
    };

    fetchComplaints();
  }, []);

  // ============================================================
  // Get complaint count for a customer
  // ============================================================

  const getComplaintCount = (customerId: string) => {
    return complaints.filter(
      (complaint) =>
        complaint.customer_id === customerId
    ).length;
  };

  // ============================================================
  // Get latest complaint
  // ============================================================

  const getLatestComplaint = (customerId: string) => {
    const customerComplaints = complaints.filter(
      (complaint) =>
        complaint.customer_id === customerId
    );

    if (customerComplaints.length === 0) {
      return null;
    }

    return customerComplaints.sort(
      (a, b) =>
        new Date(b.received_at).getTime() -
        new Date(a.received_at).getTime()
    )[0];
  };

  // ============================================================
  // Pagination
  // ============================================================

  const totalPages = data
    ? Math.max(
        1,
        Math.ceil(data.total / pageSize)
      )
    : 1;

  // ============================================================
  // Render
  // ============================================================

  return (
    <div className="space-y-4">

      {/* ======================================================
          Header
      ====================================================== */}

      <div className="flex flex-wrap items-center justify-between gap-3">

        <div className="eyebrow text-[10px]">
          {data
            ? `${data.total.toLocaleString()} customers`
            : "Loading customers…"}
        </div>

        <CustomerFilters />

      </div>

      {/* ======================================================
          Complaint loading/error information
      ====================================================== */}

      {complaintsLoading && (
        <div className="rounded-lg border border-border bg-panel px-4 py-3 text-xs text-ink-muted">
          Loading customer complaints…
        </div>
      )}

      {complaintsError && (
        <div className="rounded-lg border border-accent-rose/30 bg-accent-rose/10 px-4 py-3 text-xs text-accent-rose">
          Failed to load complaints: {complaintsError}
        </div>
      )}

      {/* ======================================================
          Customer loading
      ====================================================== */}

      {isLoading && (
        <div className="rounded-lg border border-border bg-panel py-12 text-center text-sm text-ink-muted">
          Loading risk scores…
        </div>
      )}

      {/* ======================================================
          Customer error
      ====================================================== */}

      {isError && (
        <div className="rounded-lg border border-accent-rose/30 bg-accent-rose/10 p-4 text-sm text-accent-rose">

          Failed to load customers:{" "}
          {(error as Error)?.message ??
            "unknown error"}

          <div className="mt-1 text-xs text-accent-rose/70">
            Is the API running at the address in
            VITE_API_BASE_URL?
          </div>

        </div>
      )}

      {/* ======================================================
          Customer table
      ====================================================== */}

      {data && (
        <>

          <Table>

            <TableHeader>

              <TableRow>

                <TableHead>
                  Customer ID
                </TableHead>

                <TableHead>
                  Risk Tier
                </TableHead>

                <TableHead>
                  Churn Probability
                </TableHead>

                <TableHead>
                  Contract
                </TableHead>

                <TableHead>
                  Tenure (mo)
                </TableHead>

                <TableHead>
                  Monthly Charges
                </TableHead>

                <TableHead>
                  Annual Revenue at Risk
                </TableHead>

                <TableHead>
                  Complaints
                </TableHead>

              </TableRow>

            </TableHeader>

            <TableBody>

              {data.items.length === 0 ? (

                <TableRow>

                  <TableCell
                    colSpan={8}
                    className="py-8 text-center text-ink-faint"
                  >
                    No customers match the current
                    filters.
                  </TableCell>

                </TableRow>

              ) : (

                data.items.map((c) => {

                  const complaintCount =
                    getComplaintCount(
                      c.customer_id
                    );

                  const latestComplaint =
                    getLatestComplaint(
                      c.customer_id
                    );

                  return (

                    <TableRow
                      key={c.customer_id}
                    >

                      {/* Customer ID */}

                      <TableCell className="font-mono text-xs text-ink-muted">
                        {c.customer_id}
                      </TableCell>

                      {/* Risk */}

                      <TableCell>

                        <Badge
                          variant={
                            RISK_BADGE_VARIANT[
                              c.risk_tier
                            ]
                          }
                        >
                          {c.risk_tier}
                        </Badge>

                      </TableCell>

                      {/* Churn probability */}

                      <TableCell>
                        {(
                          c.churn_probability *
                          100
                        ).toFixed(1)}
                        %
                      </TableCell>

                      {/* Contract */}

                      <TableCell>
                        {c.contract_type ?? "—"}
                      </TableCell>

                      {/* Tenure */}

                      <TableCell>
                        {c.tenure_months ?? "—"}
                      </TableCell>

                      {/* Monthly charges */}

                      <TableCell>
                        {c.monthly_charges != null
                          ? formatCurrency(
                              c.monthly_charges
                            )
                          : "—"}
                      </TableCell>

                      {/* Annual revenue at risk */}

                      <TableCell>
                        {c.annual_revenue_at_risk !=
                        null
                          ? formatCurrency(
                              c.annual_revenue_at_risk
                            )
                          : "—"}
                      </TableCell>

                      {/* Complaints */}

                      <TableCell>

                        {complaintCount === 0 ? (

                          <span className="text-ink-faint">
                            —
                          </span>

                        ) : (

                          <div className="flex flex-col gap-1">

                            <Badge variant="danger">
                              {complaintCount}{" "}
                              {complaintCount === 1
                                ? "complaint"
                                : "complaints"}
                            </Badge>

                            {latestComplaint && (
                              <span
                                className="max-w-[220px] truncate text-xs text-ink-muted"
                                title={
                                  latestComplaint.complaint_text
                                }
                              >
                                {
                                  latestComplaint.complaint_text
                                }
                              </span>
                            )}

                          </div>

                        )}

                      </TableCell>

                    </TableRow>

                  );

                })

              )}

            </TableBody>

          </Table>

          {/* ==================================================
              Pagination
          ================================================== */}

          <div className="flex items-center justify-between text-sm">

            <button
              onClick={() =>
                setPage((p) =>
                  Math.max(1, p - 1)
                )
              }
              disabled={page <= 1}
              className="rounded-md border border-border px-3 py-1.5 text-ink-muted hover:bg-panel-raised disabled:opacity-40"
            >
              Previous
            </button>

            <span className="text-ink-muted">
              Page {page} of {totalPages}
            </span>

            <button
              onClick={() =>
                setPage((p) =>
                  Math.min(
                    totalPages,
                    p + 1
                  )
                )
              }
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
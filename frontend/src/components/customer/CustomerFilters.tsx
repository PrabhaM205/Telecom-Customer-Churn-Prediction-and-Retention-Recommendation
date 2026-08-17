// frontend/src/components/customer/CustomerFilters.tsx

import { useState } from "react";
import { Download } from "lucide-react";

import { useFilterStore } from "@/store/useFilterStore";
import { useContractTypes } from "@/hooks/useContractTypes";
import type { RiskTier } from "@/api/types";

import { toast } from "sonner";

const RISK_TIERS: RiskTier[] = [
  "high",
  "medium",
  "low",
];

const API_BASE_URL = "http://localhost:8000";

export default function CustomerFilters() {
  const {
    riskTier,
    contractType,
    search,
    setRiskTier,
    setContractType,
    setSearch,
    reset,
  } = useFilterStore();

  const {
    data: contractTypes,
  } = useContractTypes();

  const [isDownloading, setIsDownloading] =
    useState(false);

  const hasActiveFilters =
    Boolean(riskTier) ||
    Boolean(contractType) ||
    Boolean(search.trim());

  // ============================================================
  // DOWNLOAD FILTERED DATA
  // ============================================================

  const handleDownloadCSV = async () => {
    if (!hasActiveFilters) {
      toast.info("Apply a filter first", {
        description:
          "Select a risk tier, contract type, or search for a customer.",
      });

      return;
    }

    try {
      setIsDownloading(true);

      const params = new URLSearchParams();

      if (search.trim()) {
        params.set(
          "search",
          search.trim()
        );
      }

      if (riskTier) {
        params.set(
          "risk_tier",
          riskTier
        );
      }

      if (contractType) {
        params.set(
          "contract_type",
          contractType
        );
      }

      const response = await fetch(
        `${API_BASE_URL}/customers/export?${params.toString()}`
      );

      if (!response.ok) {
        throw new Error(
          "Failed to download filtered data"
        );
      }

      const blob = await response.blob();

      const url =
        window.URL.createObjectURL(blob);

      const link =
        document.createElement("a");

      link.href = url;

      const timestamp =
        new Date()
          .toISOString()
          .slice(0, 10);

      link.download =
        `filtered-customers-${timestamp}.csv`;

      document.body.appendChild(link);

      link.click();

      link.remove();

      window.URL.revokeObjectURL(url);

      toast.success(
        "CSV downloaded successfully",
        {
          description:
            "The filtered customer data has been downloaded.",
        }
      );
    } catch (error) {
      console.error(
        "CSV download failed:",
        error
      );

      toast.error(
        "Failed to download CSV",
        {
          description:
            "Unable to download the filtered customer data.",
        }
      );
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-gray-200 bg-white p-4">

      {/* ========================================================
          SEARCH
          ======================================================== */}

      <input
        type="text"
        value={search}
        onChange={(e) =>
          setSearch(e.target.value)
        }
        placeholder="Search customer ID…"
        className="w-48 rounded-md border border-gray-200 px-3 py-1.5 text-sm text-black placeholder:text-black focus:outline-none focus:ring-2 focus:ring-gray-300"
      />

      {/* ========================================================
          RISK TIER
          ======================================================== */}

      <select
        value={riskTier ?? ""}
        onChange={(e) =>
          setRiskTier(
            (e.target.value || undefined) as
              | RiskTier
              | undefined
          )
        }
        className="rounded-md border border-gray-200 px-3 py-1.5 text-sm capitalize text-black focus:outline-none focus:ring-2 focus:ring-gray-300"
      >
        <option
          value=""
          className="text-black"
        >
          All risk tiers
        </option>

        {RISK_TIERS.map((tier) => (
          <option
            key={tier}
            value={tier}
            className="capitalize text-black"
          >
            {tier}
          </option>
        ))}
      </select>

      {/* ========================================================
          CONTRACT TYPE
          ======================================================== */}

      <select
        value={contractType ?? ""}
        onChange={(e) =>
          setContractType(
            e.target.value || undefined
          )
        }
        className="rounded-md border border-gray-200 px-3 py-1.5 text-sm text-black focus:outline-none focus:ring-2 focus:ring-gray-300"
      >
        <option
          value=""
          className="text-black"
        >
          All contract types
        </option>

        {(contractTypes ?? []).map(
          (type) => (
            <option
              key={type}
              value={type}
              className="text-black"
            >
              {type}
            </option>
          )
        )}
      </select>

      {/* ========================================================
          DOWNLOAD CSV
          ======================================================== */}

      {hasActiveFilters && (
        <button
          type="button"
          onClick={handleDownloadCSV}
          disabled={isDownloading}
          className="inline-flex items-center gap-2 rounded-md border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-white transition hover:bg-gray-50 hover:text-gray-900 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Download
            size={15}
            className={
              isDownloading
                ? "animate-pulse"
                : ""
            }
          />

          {isDownloading
            ? "Downloading..."
            : "Download CSV"}
        </button>
      )}

      {/* ========================================================
          CLEAR FILTERS
          ======================================================== */}

      {hasActiveFilters && (
        <button
          type="button"
          onClick={reset}
          className="text-sm text-gray-500 underline-offset-2 hover:text-gray-900 hover:underline"
        >
          Clear filters
        </button>
      )}
    </div>
  );
}
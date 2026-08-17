import { useQuery } from "@tanstack/react-query";
import { getDashboardSummary } from "@/api/dashboard";

export function useDashboardSummary() {
  return useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: getDashboardSummary,
    refetchInterval: 5 * 60 * 1000, // refresh every 5 min, matches "exec dashboard" cadence
  });
}
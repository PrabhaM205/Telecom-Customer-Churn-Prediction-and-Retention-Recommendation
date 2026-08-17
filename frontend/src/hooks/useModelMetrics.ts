import { useQuery } from "@tanstack/react-query";
import { getModelMetrics, getShapGlobalImportance } from "@/api/models";

export function useModelMetrics() {
  return useQuery({
    queryKey: ["models", "metrics"],
    queryFn: getModelMetrics,
  });
}

export function useShapGlobalImportance() {
  return useQuery({
    queryKey: ["models", "shap", "global"],
    queryFn: getShapGlobalImportance,
  });
}
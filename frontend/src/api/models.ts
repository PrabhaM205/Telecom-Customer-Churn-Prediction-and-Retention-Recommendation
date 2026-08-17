import { apiClient } from "./client";
import type { ModelMetrics, ShapGlobalImportanceItem } from "./types";

export async function getModelMetrics(): Promise<ModelMetrics> {
  const { data } = await apiClient.get<ModelMetrics>("/models/metrics");
  return data;
}

export async function getShapGlobalImportance(): Promise<ShapGlobalImportanceItem[]> {
  const { data } = await apiClient.get<ShapGlobalImportanceItem[]>("/models/shap/global");
  return data;
}
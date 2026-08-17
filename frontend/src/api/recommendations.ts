import { apiClient } from "./client";
import type { PaginatedResponse, Recommendation, RecommendationStatus } from "./types";

export interface ListRecommendationsParams {
  page?: number;
  pageSize?: number;
  customerId?: string;
}

export async function listRecommendations(
  params: ListRecommendationsParams = {}
): Promise<PaginatedResponse<Recommendation>> {
  const { page = 1, pageSize = 25, customerId } = params;
  const { data } = await apiClient.get<PaginatedResponse<Recommendation>>("/recommendations", {
    params: {
      page,
      page_size: pageSize,
      customer_id: customerId,
    },
  });
  return data;
}

export async function actionOnRecommendation(
  recommendationId: string,
  status: RecommendationStatus
): Promise<Recommendation> {
  const { data } = await apiClient.post<Recommendation>(
    `/recommendations/${recommendationId}/action`,
    { status }
  );
  return data;
}
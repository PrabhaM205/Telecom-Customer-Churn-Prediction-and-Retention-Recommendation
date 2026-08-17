import { apiClient } from "./client";
import type {
  CustomerProfile,
  CustomerRiskScore,
  PaginatedResponse,
  RiskTier,
} from "./types";

export interface ListCustomersParams {
  page?: number;
  pageSize?: number;
  riskTier?: RiskTier;
  contractType?: string;
  search?: string;
}

export async function listCustomers(
  params: ListCustomersParams = {}
): Promise<PaginatedResponse<CustomerRiskScore>> {
  const { page = 1, pageSize = 25, riskTier, contractType, search } = params;
  const { data } = await apiClient.get<PaginatedResponse<CustomerRiskScore>>("/customers", {
    params: {
      page,
      page_size: pageSize,
      risk_tier: riskTier,
      contract_type: contractType,
      search: search || undefined,
    },
  });
  return data;
}

export async function getCustomerProfile(customerId: string): Promise<CustomerProfile> {
  const { data } = await apiClient.get<CustomerProfile>(`/customers/${customerId}`);
  return data;
}

export async function getContractTypes(): Promise<string[]> {
  const { data } = await apiClient.get<string[]>("/customers/meta/contract-types");
  return data;
}
import { useQuery } from "@tanstack/react-query";
import { listCustomers, type ListCustomersParams } from "@/api/customers";

export function useCustomerRisk(params: ListCustomersParams = {}) {
  return useQuery({
    queryKey: ["customers", params],
    queryFn: () => listCustomers(params),
    placeholderData: (previousData) => previousData,
  });
}
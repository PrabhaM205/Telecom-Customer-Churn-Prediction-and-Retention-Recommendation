import { useQuery } from "@tanstack/react-query";
import { getContractTypes } from "@/api/customers";

export function useContractTypes() {
  return useQuery({
    queryKey: ["customers", "contract-types"],
    queryFn: getContractTypes,
    staleTime: Infinity, // this list essentially never changes at runtime
  });
}
import { create } from "zustand";
import type { RiskTier } from "@/api/types";

interface FilterStoreState {
  riskTier: RiskTier | undefined;
  contractType: string | undefined;
  search: string;
  setRiskTier: (tier: RiskTier | undefined) => void;
  setContractType: (type: string | undefined) => void;
  setSearch: (search: string) => void;
  reset: () => void;
}

export const useFilterStore = create<FilterStoreState>((set) => ({
  riskTier: undefined,
  contractType: undefined,
  search: "",
  setRiskTier: (tier) => set({ riskTier: tier }),
  setContractType: (type) => set({ contractType: type }),
  setSearch: (search) => set({ search }),
  reset: () => set({ riskTier: undefined, contractType: undefined, search: "" }),
}));
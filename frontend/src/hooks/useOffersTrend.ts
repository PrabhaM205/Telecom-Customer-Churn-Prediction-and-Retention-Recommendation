// frontend/src/hooks/useOffersTrend.ts

import { useQuery } from "@tanstack/react-query";
import type { OffersTrend } from "@/api/types";

const API_BASE_URL = "http://localhost:8000";

async function fetchOffersTrend(): Promise<OffersTrend> {
  const response = await fetch(
    `${API_BASE_URL}/dashboard/offers-trend`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch offers trend");
  }

  return response.json();
}

export function useOffersTrend() {
  return useQuery({
    queryKey: ["offers-trend"],
    queryFn: fetchOffersTrend,
  });
}
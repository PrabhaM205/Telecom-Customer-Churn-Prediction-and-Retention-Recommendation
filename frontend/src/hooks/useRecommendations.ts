import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  actionOnRecommendation,
  listRecommendations,
  type ListRecommendationsParams,
} from "@/api/recommendations";
import type { PaginatedResponse, Recommendation, RecommendationStatus } from "@/api/types";

export function useRecommendations(params: ListRecommendationsParams = {}) {
  return useQuery({
    queryKey: ["recommendations", params],
    queryFn: () => listRecommendations(params),
    placeholderData: (previousData) => previousData,
  });
}

export function useRecommendationAction() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: RecommendationStatus }) =>
      actionOnRecommendation(id, status),

    onMutate: async ({ id, status }) => {
      await queryClient.cancelQueries({ queryKey: ["recommendations"] });
      const previous = queryClient.getQueriesData<PaginatedResponse<Recommendation>>({
        queryKey: ["recommendations"],
      });

      queryClient.setQueriesData<PaginatedResponse<Recommendation>>(
        { queryKey: ["recommendations"] },
        (old) =>
          old && {
            ...old,
            items: old.items.map((r) => (r.id === id ? { ...r, status } : r)),
          }
      );

      return { previous };
    },

    onError: (_err, _vars, context) => {
      context?.previous.forEach(([queryKey, data]) => {
        queryClient.setQueryData(queryKey, data);
      });
    },

    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["recommendations"] });
    },
  });
}
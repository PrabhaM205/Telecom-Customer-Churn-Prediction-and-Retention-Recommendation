import { useState } from "react";
import RecommendationList from "@/components/recommendation/RecommendationList";
import { useRecommendationAction, useRecommendations } from "@/hooks/useRecommendations";
import type { RecommendationStatus } from "@/api/types";

export default function RetentionRecommendations() {
  const [page, setPage] = useState(1);
  const { data, isLoading, isError } = useRecommendations({ page });
  const { mutate: takeAction, isPending, variables } = useRecommendationAction();

  const handleAction = (id: string, status: RecommendationStatus) => {
    takeAction({ id, status });
  };

  if (isError) {
    return (
      <div className="rounded-lg border border-accent-rose/30 bg-accent-rose/10 p-4 text-sm text-accent-rose">
        Failed to load recommendations.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {isLoading ? (
        <div className="rounded-lg border border-border bg-panel py-12 text-center text-sm text-ink-muted">
          Loading…
        </div>
      ) : (
        <>
          <RecommendationList
            recommendations={data?.items ?? []}
            onAction={handleAction}
            updatingId={isPending ? variables?.id : undefined}
          />

          {data && data.total > data.page_size && (
            <div className="flex items-center justify-between text-sm">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="rounded-md border border-border px-3 py-1.5 text-ink-muted hover:bg-panel-raised disabled:opacity-40"
              >
                Previous
              </button>
              <span className="text-ink-muted">
                Page {data.page} of {Math.ceil(data.total / data.page_size)}
              </span>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={page * data.page_size >= data.total}
                className="rounded-md border border-border px-3 py-1.5 text-ink-muted hover:bg-panel-raised disabled:opacity-40"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

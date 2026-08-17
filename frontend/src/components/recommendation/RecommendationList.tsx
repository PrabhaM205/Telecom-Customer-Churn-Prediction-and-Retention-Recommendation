import RecommendationCard from "./RecommendationCard";
import type { Recommendation, RecommendationStatus } from "@/api/types";

interface RecommendationListProps {
  recommendations: Recommendation[];
  onAction: (id: string, status: RecommendationStatus) => void;
  updatingId?: string;
}

export default function RecommendationList({
  recommendations,
  onAction,
  updatingId,
}: RecommendationListProps) {
  if (recommendations.length === 0) {
    return (
      <div className="text-sm text-gray-400">
        No recommendations found.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {recommendations.map((rec) => (
        <RecommendationCard
          key={rec.id}
          recommendation={rec}
          onAccept={(id) => onAction(id, "accepted")}
          onDismiss={(id) => onAction(id, "dismissed")}
          isUpdating={updatingId === rec.id}
        />
      ))}
    </div>
  );
}
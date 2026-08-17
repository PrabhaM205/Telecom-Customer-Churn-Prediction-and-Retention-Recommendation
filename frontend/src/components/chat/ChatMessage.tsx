// frontend/src/components/chat/ChatMessage.tsx

import RecommendationCard from "@/components/recommendation/RecommendationCard";
import { useRecommendationAction } from "@/hooks/useRecommendations";
import type { ChatMessage as ChatMessageType, RecommendationStatus } from "@/api/types";

interface ChatMessageProps {
  message: ChatMessageType;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const { mutate: takeAction, isPending, variables } = useRecommendationAction();

  const handleAction = (id: string, status: RecommendationStatus) => {
    takeAction({ id, status });
  };

  return (
    <div className={`flex flex-col gap-2 ${isUser ? "items-end" : "items-start"}`}>
      <div
        className={`max-w-[80%] rounded-lg px-4 py-2 text-sm ${
          isUser ? "bg-gray-900 text-white" : "bg-gray-100 text-gray-900"
        }`}
      >
        {message.content}
      </div>

      {!isUser && message.recommendation && (
        <div className="w-full max-w-md">
          <RecommendationCard
            recommendation={message.recommendation}
            onAccept={(id) => handleAction(id, "accepted")}
            onDismiss={(id) => handleAction(id, "dismissed")}
            isUpdating={isPending && variables?.id === message.recommendation.id}
          />
        </div>
      )}
    </div>
  );
}
import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { API_BASE_URL } from "@/api/client";
import { useChatStore } from "@/store/useChatStore";
import type { AgentStreamEvent } from "@/api/types";

export function useAgentStream() {
  const addMessage = useChatStore((state) => state.addMessage);
  const clearMessages = useChatStore((state) => state.clearMessages);
  const setStreaming = useChatStore((state) => state.setStreaming);
  const setCustomerId = useChatStore((state) => state.setCustomerId);
  const queryClient = useQueryClient();

  const sendMessage = useCallback(
    async (customerId: string) => {
      clearMessages();
      setCustomerId(customerId);
      addMessage({ role: "user", content: `Run retention analysis for customer ${customerId}` });
      setStreaming(true);

      try {
        const response = await fetch(`${API_BASE_URL}/assistant/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: [], customer_id: customerId }),
        });

        if (!response.ok || !response.body) {
          throw new Error(`Request failed with status ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let offerWasSaved = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const chunks = buffer.split("\n\n");
          buffer = chunks.pop() ?? "";

          for (const chunk of chunks) {
            const line = chunk.trim();
            if (!line.startsWith("data:")) continue;

            const payload = line.slice("data:".length).trim();
            if (payload === "[DONE]") continue;

            const event: AgentStreamEvent = JSON.parse(payload);

            // Pass the recommendation through into the stored message so
            // ChatMessage.tsx can render the offer card inline. Previously
            // this only forwarded { role, content }, silently dropping it.
            addMessage({
              role: event.role,
              content: event.content,
              recommendation: event.recommendation,
            });

            // The offer_strategist event carries the generated offer — use
            // its presence (not a specific node name) as the signal that a
            // recommendation now exists server-side and the Retention
            // Recommendations page's list needs to be refetched.
            if (event.recommendation) {
              offerWasSaved = true;
            }
          }
        }

        if (offerWasSaved) {
          queryClient.invalidateQueries({ queryKey: ["recommendations"] });
        }
      } catch (error) {
        addMessage({
          role: "assistant",
          content: "Something went wrong while running the retention agent. Please try again.",
        });
      } finally {
        setStreaming(false);
      }
    },
    [addMessage, clearMessages, setCustomerId, setStreaming, queryClient]
  );

  return { sendMessage };
}
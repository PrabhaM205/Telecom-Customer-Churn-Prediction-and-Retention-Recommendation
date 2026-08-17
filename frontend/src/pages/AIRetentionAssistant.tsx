import ChatWindow from "@/components/chat/ChatWindow";
import ChatInput from "@/components/chat/ChatInput";
import { useAgentStream } from "@/hooks/useAgentStream";
import { useChatStore } from "@/store/useChatStore";

export default function AIRetentionAssistant() {
  const { sendMessage } = useAgentStream();
  const isStreaming = useChatStore((state) => state.isStreaming);

  return (
    <div className="flex h-[calc(100vh-10rem)] flex-col rounded-lg border border-border bg-panel">
      <ChatWindow />
      <ChatInput onSubmit={sendMessage} disabled={isStreaming} />
    </div>
  );
}

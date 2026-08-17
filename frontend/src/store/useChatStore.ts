import { create } from "zustand";
import type { ChatMessage } from "@/api/types";

interface ChatStoreState {
  customerId: string | null;
  messages: ChatMessage[];
  isStreaming: boolean;
  setCustomerId: (id: string) => void;
  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
  setStreaming: (streaming: boolean) => void;
}

export const useChatStore = create<ChatStoreState>((set) => ({
  customerId: null,
  messages: [],
  isStreaming: false,
  setCustomerId: (id) => set({ customerId: id }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  clearMessages: () => set({ messages: [] }),
  setStreaming: (streaming) => set({ isStreaming: streaming }),
}));
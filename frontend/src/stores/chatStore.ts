import { create } from 'zustand'

export interface ChatMessage {
  id?: number
  role: 'user' | 'assistant'
  content: string
}

interface ChatStore {
  messages: ChatMessage[]
  isOpen: boolean
  isStreaming: boolean
  addMessage: (msg: ChatMessage) => void
  appendToLast: (chunk: string) => void
  setMessages: (msgs: ChatMessage[]) => void
  toggleOpen: () => void
  setStreaming: (v: boolean) => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isOpen: false,
  isStreaming: false,
  addMessage: (msg) => set((s) => ({ messages: [...s.messages, msg] })),
  appendToLast: (chunk) =>
    set((s) => {
      const msgs = [...s.messages]
      if (msgs.length > 0 && msgs[msgs.length - 1].role === 'assistant') {
        msgs[msgs.length - 1] = { ...msgs[msgs.length - 1], content: msgs[msgs.length - 1].content + chunk }
      }
      return { messages: msgs }
    }),
  setMessages: (msgs) => set({ messages: msgs }),
  toggleOpen: () => set((s) => ({ isOpen: !s.isOpen })),
  setStreaming: (v) => set({ isStreaming: v }),
}))

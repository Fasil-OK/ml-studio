import { useState, useRef, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { X, Send, Square, MessageSquare } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useChatStore, type ChatMessage } from '../../stores/chatStore'
import { useChatSocket } from '../../hooks/useChatSocket'

export default function ChatPanel() {
  const { id } = useParams()
  const { messages, isOpen, isStreaming, toggleOpen } = useChatStore()
  const { sendMessage, stopGeneration } = useChatSocket(isOpen ? (id || null) : null)
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (!input.trim() || isStreaming) return
    sendMessage(input.trim(), { page: window.location.pathname })
    setInput('')
  }

  if (!isOpen) return null

  return (
    <div className="w-[28rem] bg-white border-l border-gray-200 flex flex-col h-full">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4 text-blue-600" />
          <h3 className="text-sm font-semibold text-gray-900">AI Assistant</h3>
        </div>
        <button onClick={toggleOpen} className="p-1 rounded hover:bg-gray-100">
          <X className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && (
          <div className="text-center py-8">
            <MessageSquare className="w-8 h-8 text-gray-200 mx-auto mb-2" />
            <p className="text-xs text-gray-400">Ask me anything about your model</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'user' ? (
              <div className="max-w-[85%] rounded-xl px-3 py-2 text-sm bg-blue-600 text-white">
                {msg.content}
              </div>
            ) : (
              <div className="max-w-[95%] rounded-xl px-3 py-2 text-sm bg-gray-100 text-gray-800">
                {msg.content ? (
                  <div className="prose prose-sm max-w-none prose-headings:text-gray-900 prose-headings:text-sm prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1 prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-code:text-xs prose-code:bg-gray-200 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-800 prose-pre:text-gray-100 prose-pre:text-xs prose-pre:rounded-lg prose-pre:my-2 prose-table:text-xs prose-th:px-2 prose-th:py-1 prose-td:px-2 prose-td:py-1 prose-strong:text-gray-900">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  isStreaming && i === messages.length - 1 ? '...' : ''
                )}
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggestion chips */}
      {messages.length === 0 && (
        <div className="px-4 pb-2 flex flex-wrap gap-1.5">
          {['Analyze my dataset', 'Suggest hyperparameters', 'How to improve accuracy?'].map((s) => (
            <button
              key={s}
              onClick={() => { sendMessage(s) }}
              className="text-xs px-2.5 py-1 bg-blue-50 text-blue-600 rounded-full hover:bg-blue-100 transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      <div className="p-3 border-t border-gray-200">
        {isStreaming && (
          <div className="flex justify-center mb-2">
            <button
              onClick={stopGeneration}
              className="flex items-center gap-1.5 px-3 py-1 text-xs text-gray-600 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
            >
              <Square className="w-3 h-3 fill-current" />
              Stop generating
            </button>
          </div>
        )}
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask about your model..."
            className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isStreaming}
          />
          <button
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
            className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}

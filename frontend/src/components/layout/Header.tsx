import { MessageSquare } from 'lucide-react'
import { useProjectStore } from '../../stores/projectStore'
import { useChatStore } from '../../stores/chatStore'

export default function Header() {
  const project = useProjectStore((s) => s.currentProject)
  const toggleChat = useChatStore((s) => s.toggleOpen)

  return (
    <header className="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-6">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold text-gray-900">
          {project?.name || 'Project'}
        </h1>
        {project && (
          <span className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded-full font-medium">
            {project.task_type}
          </span>
        )}
        {project && (
          <span className="px-2 py-0.5 bg-gray-100 text-gray-600 text-xs rounded-full">
            {project.status}
          </span>
        )}
      </div>
      <button
        onClick={toggleChat}
        className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-900 transition-colors"
        title="Toggle AI Chat"
      >
        <MessageSquare className="w-5 h-5" />
      </button>
    </header>
  )
}

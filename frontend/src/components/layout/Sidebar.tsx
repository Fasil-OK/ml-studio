import { useParams, useLocation, Link } from 'react-router-dom'
import { Home, Upload, Brain, Play, BarChart3, Eye, FlaskConical, Settings, ArrowLeft } from 'lucide-react'

const steps = [
  { path: '', label: 'Overview', icon: Home },
  { path: '/data', label: 'Dataset', icon: Upload },
  { path: '/model', label: 'Model', icon: Brain },
  { path: '/train', label: 'Training', icon: Play },
  { path: '/evaluate', label: 'Evaluate', icon: BarChart3 },
  { path: '/explain', label: 'Explain', icon: Eye },
  { path: '/test', label: 'Test', icon: FlaskConical },
  { path: '/settings', label: 'Settings', icon: Settings },
]

export default function Sidebar() {
  const { id } = useParams()
  const location = useLocation()

  return (
    <aside className="w-56 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <Link to="/" className="flex items-center gap-2 text-gray-600 hover:text-gray-900 text-sm">
          <ArrowLeft className="w-4 h-4" />
          All Projects
        </Link>
      </div>
      <nav className="flex-1 py-4">
        {steps.map((step) => {
          const href = `/project/${id}${step.path}`
          const isActive = location.pathname === href
          const Icon = step.icon
          return (
            <Link
              key={step.path}
              to={href}
              className={`flex items-center gap-3 px-4 py-2.5 text-sm transition-colors ${
                isActive
                  ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-700'
                  : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              }`}
            >
              <Icon className="w-4 h-4" />
              {step.label}
            </Link>
          )
        })}
      </nav>
      <div className="p-4 text-xs text-gray-400 border-t border-gray-200">
        AutoML Platform
      </div>
    </aside>
  )
}

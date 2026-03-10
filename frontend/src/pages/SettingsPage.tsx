import { useState, useEffect } from 'react'
import { Settings, Save, CheckCircle } from 'lucide-react'
import api from '../api/client'
import toast from 'react-hot-toast'

export default function SettingsPage() {
  const [baseUrl, setBaseUrl] = useState('https://api.openai.com/v1')
  const [apiKey, setApiKey] = useState('')
  const [model, setModel] = useState('gpt-4o')
  const [hasKey, setHasKey] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    api.get('/settings').then(({ data }) => {
      setBaseUrl(data.llm_base_url)
      setModel(data.llm_model)
      setHasKey(data.has_key)
    })
  }, [])

  const handleSave = async () => {
    setSaving(true)
    try {
      await api.put('/settings', {
        llm_base_url: baseUrl,
        llm_api_key: apiKey,
        llm_model: model,
      })
      toast.success('Settings saved!')
      setHasKey(true)
      setApiKey('')
    } catch {
      toast.error('Failed to save settings')
    }
    setSaving(false)
  }

  return (
    <div>
      <h2 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
        <Settings className="w-5 h-5" /> Settings
      </h2>

      <div className="max-w-lg">
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-900 mb-4">LLM Configuration</h3>
          <p className="text-xs text-gray-500 mb-5">
            Configure any OpenAI-compatible API. Works with OpenAI, Ollama, vLLM, LMStudio, Azure, etc.
          </p>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Base URL</label>
              <input
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="https://api.openai.com/v1"
              />
              <p className="text-xs text-gray-400 mt-1">
                Ollama: http://localhost:11434/v1 | LMStudio: http://localhost:1234/v1
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                API Key
                {hasKey && (
                  <span className="ml-2 text-green-600 text-xs font-normal inline-flex items-center gap-1">
                    <CheckCircle className="w-3 h-3" /> configured
                  </span>
                )}
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"
                placeholder={hasKey ? '(leave empty to keep current)' : 'sk-...'}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
              <input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="gpt-4o"
              />
              <p className="text-xs text-gray-400 mt-1">
                e.g., gpt-4o, gpt-4o-mini, llama3, mistral, claude-3-opus
              </p>
            </div>
          </div>

          <button
            onClick={handleSave}
            disabled={saving}
            className="mt-6 flex items-center gap-2 bg-blue-600 text-white px-4 py-2.5 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            <Save className="w-4 h-4" /> {saving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>
    </div>
  )
}

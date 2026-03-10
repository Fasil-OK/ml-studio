import { useEffect } from 'react'
import { Outlet, useParams } from 'react-router-dom'
import Sidebar from './Sidebar'
import Header from './Header'
import ChatPanel from '../chat/ChatPanel'
import { useProjectStore } from '../../stores/projectStore'
import { getProject } from '../../api/projects'

export default function AppShell() {
  const { id } = useParams()
  const { setCurrentProject } = useProjectStore()

  useEffect(() => {
    if (id) {
      getProject(id).then(setCurrentProject)
    }
    return () => setCurrentProject(null)
  }, [id])

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
      <ChatPanel />
    </div>
  )
}

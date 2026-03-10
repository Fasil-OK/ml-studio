import api from './client'

export interface Project {
  id: string
  name: string
  task_type: string
  description: string | null
  status: string
  created_at: string
  updated_at: string
}

export const createProject = (data: { name: string; task_type: string; description?: string }) =>
  api.post<Project>('/projects', data).then(r => r.data)

export const listProjects = () =>
  api.get<Project[]>('/projects').then(r => r.data)

export const getProject = (id: string) =>
  api.get<Project>(`/projects/${id}`).then(r => r.data)

export const deleteProject = (id: string) =>
  api.delete(`/projects/${id}`).then(r => r.data)

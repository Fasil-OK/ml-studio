import api from './client'

export interface DatasetInfo {
  id: string
  project_id: string
  name: string
  total_images: number | null
  num_classes: number | null
  class_names: string[] | null
  class_counts: Record<string, number> | null
  image_stats: Record<string, any> | null
  annotation_format: string | null
  quality_issues: Array<Record<string, any>> | null
  split_info: Record<string, number> | null
  created_at: string
}

export const uploadDataset = (projectId: string, file: File) => {
  const form = new FormData()
  form.append('file', file)
  return api.post<DatasetInfo>(`/projects/${projectId}/dataset`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  }).then(r => r.data)
}

export const getDataset = (projectId: string) =>
  api.get<DatasetInfo | null>(`/projects/${projectId}/dataset`).then(r => r.data)

export const getSamples = (projectId: string, className?: string, limit = 20) =>
  api.get(`/projects/${projectId}/dataset/samples`, {
    params: { class_name: className, limit },
  }).then(r => r.data)

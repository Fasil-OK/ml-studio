import api from './client'

export interface PredictionResult {
  predictions: Array<{ class: string; confidence: number }>
  processing_time_ms: number
  task_type: string
}

export const predict = (experimentId: string, image: File) => {
  const form = new FormData()
  form.append('image', image)
  return api.post<PredictionResult>(`/experiments/${experimentId}/predict`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

export const generateExplanation = (experimentId: string, image: File, method: string) => {
  const form = new FormData()
  form.append('image', image)
  form.append('method', method)
  return api.post(`/experiments/${experimentId}/explain`, form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data)
}

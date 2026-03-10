import api from './client'

export interface EvaluationResult {
  id: string
  experiment_id: string
  metrics: Record<string, number>
  confusion_matrix: number[][] | null
  per_class_metrics: Array<Record<string, any>> | null
  best_checkpoint: string | null
  insights: string[] | null
  created_at: string
}

export const runEvaluation = (experimentId: string) =>
  api.post<EvaluationResult>(`/experiments/${experimentId}/evaluate`).then(r => r.data)

export const getEvaluation = (experimentId: string) =>
  api.get<EvaluationResult | null>(`/experiments/${experimentId}/evaluation`).then(r => r.data)

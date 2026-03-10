import api from './client'

export interface TrainingMetric {
  epoch: number
  train_loss: number | null
  train_accuracy: number | null
  val_loss: number | null
  val_accuracy: number | null
  learning_rate: number | null
  epoch_duration: number | null
  gpu_memory_used: number | null
  extra_metrics: Record<string, any> | null
  timestamp: string
}

export interface Experiment {
  id: string
  project_id: string
  dataset_id: string
  architecture: string
  pretrained: boolean
  hyperparameters: Record<string, any>
  resource_config: Record<string, any> | null
  status: string
  created_at: string
}

export const createExperiment = (projectId: string, data: any) =>
  api.post<Experiment>(`/projects/${projectId}/experiments`, data).then(r => r.data)

export const listExperiments = (projectId: string) =>
  api.get<Experiment[]>(`/projects/${projectId}/experiments`).then(r => r.data)

export const startTraining = (experimentId: string) =>
  api.post(`/experiments/${experimentId}/train`).then(r => r.data)

export const stopTraining = (experimentId: string) =>
  api.post(`/experiments/${experimentId}/train/stop`).then(r => r.data)

export const getTrainingStatus = (experimentId: string) =>
  api.get(`/experiments/${experimentId}/train`).then(r => r.data)

export const getModels = (taskType: string) =>
  api.get('/models', { params: { task_type: taskType } }).then(r => r.data)

export const getResources = () =>
  api.get('/resources').then(r => r.data)

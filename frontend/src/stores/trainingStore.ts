import { create } from 'zustand'
import type { TrainingMetric } from '../api/training'

interface TrainingStore {
  metrics: TrainingMetric[]
  status: string
  currentEpoch: number | null
  totalEpochs: number | null
  addMetric: (metric: TrainingMetric) => void
  setMetrics: (metrics: TrainingMetric[]) => void
  setStatus: (status: string) => void
  setProgress: (current: number, total: number) => void
  reset: () => void
}

export const useTrainingStore = create<TrainingStore>((set) => ({
  metrics: [],
  status: 'idle',
  currentEpoch: null,
  totalEpochs: null,
  addMetric: (metric) => set((s) => ({ metrics: [...s.metrics, metric] })),
  setMetrics: (metrics) => set({ metrics }),
  setStatus: (status) => set({ status }),
  setProgress: (current, total) => set({ currentEpoch: current, totalEpochs: total }),
  reset: () => set({ metrics: [], status: 'idle', currentEpoch: null, totalEpochs: null }),
}))

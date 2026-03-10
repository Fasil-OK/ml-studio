import { useWebSocket } from './useWebSocket'
import { useTrainingStore } from '../stores/trainingStore'

export function useTrainingSocket(experimentId: string | null) {
  const { addMetric, setStatus, setProgress } = useTrainingStore()

  const { send } = useWebSocket(
    experimentId ? `/ws/training/${experimentId}` : null,
    (data) => {
      if (data.type === 'epoch_end') {
        addMetric({
          epoch: data.epoch,
          train_loss: data.train_loss,
          train_accuracy: data.train_accuracy,
          val_loss: data.val_loss,
          val_accuracy: data.val_accuracy,
          learning_rate: data.lr,
          epoch_duration: data.duration,
          gpu_memory_used: data.gpu_memory_mb,
          extra_metrics: data.extra_metrics,
          timestamp: new Date().toISOString(),
        })
        setProgress(data.epoch + 1, data.total_epochs)
      } else if (data.type === 'training_complete') {
        setStatus('completed')
      } else if (data.type === 'training_failed') {
        setStatus('failed')
      }
    },
  )

  return { send }
}

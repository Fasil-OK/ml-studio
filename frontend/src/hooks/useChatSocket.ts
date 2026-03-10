import { useWebSocket } from './useWebSocket'
import { useChatStore } from '../stores/chatStore'
import { useTrainingStore } from '../stores/trainingStore'

export function useChatSocket(projectId: string | null) {
  const { addMessage, appendToLast, setStreaming } = useChatStore()

  const { send } = useWebSocket(
    projectId ? `/ws/chat/${projectId}` : null,
    (data) => {
      if (data.type === 'chunk') {
        appendToLast(data.content)
      } else if (data.type === 'end') {
        setStreaming(false)
      }
    },
  )

  const sendMessage = (content: string, context?: Record<string, any>) => {
    // Gather live training context from store
    const { metrics, status, currentEpoch, totalEpochs } = useTrainingStore.getState()
    const recentMetrics = metrics.slice(-5).map((m) => ({
      epoch: m.epoch,
      train_loss: m.train_loss,
      val_loss: m.val_loss,
      val_accuracy: m.val_accuracy,
      learning_rate: m.learning_rate,
    }))

    addMessage({ role: 'user', content })
    addMessage({ role: 'assistant', content: '' })
    setStreaming(true)
    send({
      type: 'message',
      content,
      context: {
        ...context,
        trainingMetrics: recentMetrics,
        trainingStatus: status,
        currentEpoch,
        totalEpochs,
      },
    })
  }

  const stopGeneration = () => {
    send({ type: 'stop' })
    setStreaming(false)
  }

  return { sendMessage, stopGeneration }
}

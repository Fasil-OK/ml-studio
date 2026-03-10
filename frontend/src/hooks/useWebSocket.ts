import { useEffect, useRef, useCallback } from 'react'
import { createWebSocket } from '../api/ws'

export function useWebSocket(
  path: string | null,
  onMessage: (data: any) => void,
) {
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!path) return

    const ws = createWebSocket(path)
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        onMessage(data)
      } catch {}
    }

    ws.onclose = () => {
      // Reconnect after 3 seconds
      setTimeout(() => {
        if (wsRef.current === ws) {
          wsRef.current = createWebSocket(path)
          wsRef.current.onmessage = ws.onmessage
        }
      }, 3000)
    }

    return () => {
      ws.close()
      wsRef.current = null
    }
  }, [path])

  const send = useCallback((data: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { send }
}

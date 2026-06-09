import { useCallback, useEffect, useRef } from 'react'
import { useMissionStore } from '../stores/missionStore'
import { wsMessageSchema } from '../types/ws-messages'

const WS_BASE = import.meta.env.VITE_BACKEND_WS_URL ?? 'ws://localhost:8000/ws/mission'
const MAX_RECONNECT_MS = 30000

export function useWebSocket(missionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const attemptRef = useRef(0)
  const reconnectTimer = useRef<number | null>(null)

  const setWsStatus = useMissionStore((s) => s.setWsStatus)
  const setWsSend = useMissionStore((s) => s.setWsSend)
  const setHeatmapFull = useMissionStore((s) => s.setHeatmapFull)
  const applyHeatmapDelta = useMissionStore((s) => s.applyHeatmapDelta)
  const setEngineTick = useMissionStore((s) => s.setEngineTick)

  const handleMessage = useCallback(
    (raw: string) => {
      let parsed: unknown
      try {
        parsed = JSON.parse(raw)
      } catch {
        return
      }
      const result = wsMessageSchema.safeParse(parsed)
      if (!result.success) {
        if (import.meta.env.DEV) console.warn('Unknown WS message:', result.error)
        return
      }
      const msg = result.data
      if ('type' in msg && msg.type === 'heatmap_full') {
        setHeatmapFull(msg.metadata, msg.probabilities)
      } else if ('type' in msg && msg.type === 'heatmap_delta') {
        applyHeatmapDelta(msg.cells)
      } else if ('event' in msg && msg.event === 'engine_tick') {
        setEngineTick(msg.mpp_coords, msg.tick_count, msg.layers)
      }
    },
    [setHeatmapFull, applyHeatmapDelta, setEngineTick],
  )

  useEffect(() => {
    if (!missionId) {
      setWsSend(null)
      return
    }

    const connect = () => {
      setWsStatus('connecting')
      const url = `${WS_BASE}/${missionId}`
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        attemptRef.current = 0
        setWsStatus('open')
        setWsSend((payload: unknown) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(payload))
          }
        })
      }

      ws.onmessage = (ev) => handleMessage(ev.data as string)

      ws.onerror = () => setWsStatus('error')

      ws.onclose = () => {
        setWsStatus('closed')
        setWsSend(null)
        wsRef.current = null
        const delay = Math.min(1000 * 2 ** attemptRef.current, MAX_RECONNECT_MS)
        attemptRef.current += 1
        reconnectTimer.current = window.setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
      wsRef.current = null
      setWsSend(null)
    }
  }, [missionId, handleMessage, setWsStatus, setWsSend])

  return wsRef
}

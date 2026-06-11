import { useCallback, useEffect, useRef } from 'react'
import { useMissionStore } from '../stores/missionStore'
import {
  engineTickSchema,
  heatmapDeltaSchema,
  heatmapFullSchema,
} from '../types/ws-messages'

const WS_BASE = import.meta.env.VITE_BACKEND_WS_URL ?? 'ws://localhost:8000/ws/mission'
const MAX_RECONNECT_MS = 30000

export function useWebSocket(missionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const attemptRef = useRef(0)
  const reconnectTimer = useRef<number | null>(null)
  const intentionalCloseRef = useRef(false)
  const activeMissionRef = useRef<string | null>(null)

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
      if (!parsed || typeof parsed !== 'object') return

      const msg = parsed as Record<string, unknown>

      if (msg.type === 'heatmap_full') {
        const result = heatmapFullSchema.safeParse(msg)
        if (result.success) {
          setHeatmapFull(result.data.metadata, result.data.probabilities)
        } else if (import.meta.env.DEV) {
          console.warn('Invalid heatmap_full:', result.error)
        }
        return
      }

      if (msg.type === 'heatmap_delta') {
        const result = heatmapDeltaSchema.safeParse(msg)
        if (result.success) {
          applyHeatmapDelta(result.data.cells)
        }
        return
      }

      if (msg.event === 'engine_tick') {
        const result = engineTickSchema.safeParse(msg)
        if (result.success) {
          setEngineTick(result.data.mpp_coords, result.data.tick_count, result.data.layers)
        }
        return
      }

      if (msg.type === 'mission_closed') {
        return
      }
    },
    [setHeatmapFull, applyHeatmapDelta, setEngineTick],
  )

  useEffect(() => {
    activeMissionRef.current = missionId

    if (!missionId) {
      intentionalCloseRef.current = true
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
      wsRef.current?.close()
      wsRef.current = null
      setWsSend(null)
      setWsStatus('idle')
      return
    }

    intentionalCloseRef.current = false
    attemptRef.current = 0

    const connect = () => {
      if (activeMissionRef.current !== missionId) return

      setWsStatus('connecting')
      const url = `${WS_BASE}/${missionId}`
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        if (activeMissionRef.current !== missionId) {
          ws.close()
          return
        }
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
        setWsSend(null)
        wsRef.current = null
        if (intentionalCloseRef.current || activeMissionRef.current !== missionId) {
          setWsStatus('closed')
          return
        }
        setWsStatus('closed')
        const delay = Math.min(1000 * 2 ** attemptRef.current, MAX_RECONNECT_MS)
        attemptRef.current += 1
        reconnectTimer.current = window.setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      intentionalCloseRef.current = true
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
        reconnectTimer.current = null
      }
      wsRef.current?.close()
      wsRef.current = null
      setWsSend(null)
    }
  }, [missionId, handleMessage, setWsStatus, setWsSend])

  return wsRef
}

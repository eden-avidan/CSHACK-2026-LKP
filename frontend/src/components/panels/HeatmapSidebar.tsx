import { useCallback, useEffect, useRef, useState } from 'react'
import type { Map } from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import { useWebSocket } from '../../hooks/useWebSocket'
import { HAIFA_MAP_VIEW } from '../../types/geo'
import type { GridMetadata } from '../../types/geo'
import type { DroneRoute } from '../../types/geo'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

interface HeatmapSidebarProps {
  map: Map | null
}

export function HeatmapSidebar({ map }: HeatmapSidebarProps) {
  const mode = useMissionStore((s) => s.mode)
  const layers = useMissionStore((s) => s.layers)
  const draftLkp = useMissionStore((s) => s.draftLkp)
  const pinnedLkp = useMissionStore((s) => s.pinnedLkp)
  const lkpTimestamp = useMissionStore((s) => s.lkpTimestamp)
  const pace = useMissionStore((s) => s.pace)
  const missionId = useMissionStore((s) => s.missionId)
  const wsStatus = useMissionStore((s) => s.wsStatus)
  const setMode = useMissionStore((s) => s.setMode)
  const setPinnedLkp = useMissionStore((s) => s.setPinnedLkp)
  const setLkpTimestamp = useMissionStore((s) => s.setLkpTimestamp)
  const setPace = useMissionStore((s) => s.setPace)
  const setMission = useMissionStore((s) => s.setMission)
  const setHeatmapFull = useMissionStore((s) => s.setHeatmapFull)
  const resetMission = useMissionStore((s) => s.resetMission)
  const setDroneRoute = useMissionStore((s) => s.setDroneRoute)

  const [loading, setLoading] = useState(false)
  const [routeLoading, setRouteLoading] = useState(false)
  const [routeSummary, setRouteSummary] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [backendOk, setBackendOk] = useState<boolean | null>(null)

  const pacePatchReady = useRef(false)

  useWebSocket(missionId)

  useEffect(() => {
    let cancelled = false
    const check = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/health`)
        if (!cancelled) setBackendOk(res.ok)
      } catch {
        if (!cancelled) setBackendOk(false)
      }
    }
    check()
    const id = window.setInterval(check, 10000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  useEffect(() => {
    pacePatchReady.current = false
    setRouteSummary(null)
  }, [missionId])

  useEffect(() => {
    if (!missionId || mode !== 'live') return
    if (!pacePatchReady.current) {
      pacePatchReady.current = true
      return
    }
    const t = window.setTimeout(async () => {
      await fetch(`${BACKEND_URL}/missions/${missionId}/pace`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pace }),
      })
    }, 400)
    return () => clearTimeout(t)
  }, [missionId, mode, pace])

  const pinLkp = useCallback(() => {
    if (draftLkp) {
      setPinnedLkp(draftLkp)
      setError(null)
    } else {
      setError('Click the map to pick a position first')
    }
  }, [draftLkp, setPinnedLkp])

  const runHeatmap = useCallback(async () => {
    if (!pinnedLkp) {
      setError('Pin LKP on the map before running the heatmap')
      return
    }
    if (mode === 'offline' && !lkpTimestamp) {
      setError('Set the LKP date and time for offline mode')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const body: Record<string, unknown> = {
        mode,
        lkp: pinnedLkp,
        pace,
        layers,
      }
      if (mode === 'offline' && lkpTimestamp) {
        body.lkp_timestamp = new Date(lkpTimestamp).toISOString()
      }
      const res = await fetch(`${BACKEND_URL}/missions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = (await res.json()) as { mission_id: string }

      const heatmapRes = await fetch(`${BACKEND_URL}/missions/${data.mission_id}/heatmap`)
      if (heatmapRes.ok) {
        const heat = (await heatmapRes.json()) as {
          metadata: GridMetadata
          probabilities: number[]
        }
        setMission(data.mission_id, pinnedLkp, mode, pace)
        setHeatmapFull(heat.metadata, heat.probabilities)
      } else {
        setMission(data.mission_id, pinnedLkp, mode, pace)
      }
    } catch (e) {
      if (e instanceof TypeError) {
        setError(
          `Cannot reach backend at ${BACKEND_URL}. Start backend: cd backend && uvicorn app.main:app --reload --port 8000`,
        )
      } else {
        setError(e instanceof Error ? e.message : 'Failed to run heatmap')
      }
    } finally {
      setLoading(false)
    }
  }, [pinnedLkp, mode, lkpTimestamp, pace, layers, setMission, setHeatmapFull])

  const stopAndNew = useCallback(async () => {
    if (!missionId) return
    setError(null)
    try {
      await fetch(`${BACKEND_URL}/missions/${missionId}`, { method: 'DELETE' })
    } catch {
      // reset locally even if backend unreachable
    }
    resetMission()
    if (map) {
      map.flyTo({ center: HAIFA_MAP_VIEW.center, zoom: HAIFA_MAP_VIEW.zoom, duration: 800 })
    }
  }, [missionId, resetMission, map])

  const findDroneRoute = useCallback(async () => {
    if (!missionId) return
    setRouteLoading(true)
    setError(null)
    try {
      const res = await fetch(`${BACKEND_URL}/missions/${missionId}/drone-route`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error(await res.text())
      const data = (await res.json()) as {
        route: DroneRoute
        expected_coverage: number
        length_m: number
        route_points: number
      }
      setDroneRoute(data.route)
      setRouteSummary(
        `${data.route_points} points · ${(data.length_m / 1000).toFixed(1)} km · ${(data.expected_coverage * 100).toFixed(1)}% probability coverage`,
      )
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to find drone route')
    } finally {
      setRouteLoading(false)
    }
  }, [missionId, setDroneRoute])

  return (
    <div className="heatmap-sidebar">
      <h2>Heatmap</h2>
      <p className="hint">Click the map, then Pin LKP in the active section.</p>

      <p className={`backend-status ${backendOk === false ? 'offline' : backendOk ? 'online' : ''}`}>
        Backend: {backendOk === null ? 'checking…' : backendOk ? 'connected' : 'offline'}
      </p>

      <fieldset className="mode-fieldset">
        <legend>Mode</legend>
        <label className="mode-radio">
          <input
            type="radio"
            name="heatmap-mode"
            value="live"
            checked={mode === 'live'}
            onChange={() => setMode('live')}
            disabled={!!missionId}
          />
          <span>Live</span>
        </label>
        <label className="mode-radio">
          <input
            type="radio"
            name="heatmap-mode"
            value="offline"
            checked={mode === 'offline'}
            onChange={() => setMode('offline')}
            disabled={!!missionId}
          />
          <span>Offline</span>
        </label>
      </fieldset>

      {mode === 'live' && (
        <section className="mode-section" aria-label="Live mode controls">
          <button type="button" className="pin-btn" onClick={pinLkp} disabled={!!missionId}>
            Pin LKP
          </button>
          {pinnedLkp && !missionId && (
            <p className="coords">
              LKP: {pinnedLkp.lat.toFixed(5)}, {pinnedLkp.lon.toFixed(5)}
            </p>
          )}
          <label className="field pace-slider">
            <span>Pace — {pace.toFixed(1)}× reality</span>
            <input
              type="range"
              min={0.1}
              max={120}
              step={0.1}
              value={pace}
              onChange={(e) => setPace(Number(e.target.value))}
              disabled={!!missionId && mode !== 'live'}
              aria-label="Simulation pace multiplier"
            />
          </label>
          <p className="pace-hint">Heatmap refreshes every 1 s. Pace controls simulated time per tick.</p>
        </section>
      )}

      {mode === 'offline' && (
        <section className="mode-section" aria-label="Offline mode controls">
          <button type="button" className="pin-btn" onClick={pinLkp} disabled={!!missionId}>
            Pin LKP
          </button>
          {pinnedLkp && !missionId && (
            <p className="coords">
              LKP: {pinnedLkp.lat.toFixed(5)}, {pinnedLkp.lon.toFixed(5)}
            </p>
          )}
          <label className="field">
            <span>LKP date &amp; time</span>
            <input
              type="datetime-local"
              value={lkpTimestamp ?? ''}
              onChange={(e) => setLkpTimestamp(e.target.value || null)}
              disabled={!!missionId}
              aria-label="Last known position timestamp"
            />
          </label>
          <p className="pace-hint">
            Computes where the subject likely is now based on elapsed time since last seen.
          </p>
        </section>
      )}

      {missionId && (
        <p className="mission-id">
          Mission: <code>{missionId.slice(0, 8)}…</code>
          {wsStatus === 'open' ? ' · live' : ` · ${wsStatus}`}
        </p>
      )}

      {error && <p className="error">{error}</p>}

      {!missionId && (
        <button type="button" onClick={runHeatmap} disabled={loading || !pinnedLkp || backendOk === false}>
          {loading ? 'Loading terrain…' : 'Run Heatmap'}
        </button>
      )}

      {missionId && (
        <>
          <button type="button" onClick={findDroneRoute} disabled={routeLoading}>
            {routeLoading ? 'Planning Route…' : 'Find Drone Route'}
          </button>
          {routeSummary && <p className="route-summary">{routeSummary}</p>}
          <button type="button" className="secondary" onClick={stopAndNew}>
            Stop &amp; New
          </button>
        </>
      )}
    </div>
  )
}

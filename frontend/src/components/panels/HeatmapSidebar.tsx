import { useCallback, useEffect, useRef, useState } from 'react'
import type { Map } from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import type { TerrainData } from '../../stores/missionStore'
import { useWebSocket } from '../../hooks/useWebSocket'
import type { GridMetadata } from '../../types/geo'
import type { DroneRoute } from '../../types/geo'
import { BASE_STEP_SEC, formatDuration } from '../../utils/formatTime'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

interface HeatmapSidebarProps {
  map?: Map | null
}

export function HeatmapSidebar(_props: HeatmapSidebarProps) {
  const mode = useMissionStore((s) => s.mode)
  const layers = useMissionStore((s) => s.layers)
  const personality = useMissionStore((s) => s.personality)
  const draftLkp = useMissionStore((s) => s.draftLkp)
  const pinnedLkp = useMissionStore((s) => s.pinnedLkp)
  const lkpTimestamp = useMissionStore((s) => s.lkpTimestamp)
  const pace = useMissionStore((s) => s.pace)
  const stepSec = useMissionStore((s) => s.stepSec)
  const tickCount = useMissionStore((s) => s.tickCount)
  const missionId = useMissionStore((s) => s.missionId)
  const wsStatus = useMissionStore((s) => s.wsStatus)
  const simulationRunning = useMissionStore((s) => s.simulationRunning)
  const setMode = useMissionStore((s) => s.setMode)
  const setPinnedLkp = useMissionStore((s) => s.setPinnedLkp)
  const setLkpTimestamp = useMissionStore((s) => s.setLkpTimestamp)
  const setPace = useMissionStore((s) => s.setPace)
  const setStepSec = useMissionStore((s) => s.setStepSec)
  const setMission = useMissionStore((s) => s.setMission)
  const setHeatmapFull = useMissionStore((s) => s.setHeatmapFull)
  const setTerrainData = useMissionStore((s) => s.setTerrainData)
  const resetMission = useMissionStore((s) => s.resetMission)
  const setDroneRoute = useMissionStore((s) => s.setDroneRoute)
  const setSimulationRunning = useMissionStore((s) => s.setSimulationRunning)

  const [loading, setLoading] = useState(false)
  const [pauseLoading, setPauseLoading] = useState(false)
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
    if (!missionId) return
    if (!pacePatchReady.current) {
      pacePatchReady.current = true
      return
    }
    const t = window.setTimeout(async () => {
      const res = await fetch(`${BACKEND_URL}/missions/${missionId}/pace`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pace }),
      })
      if (res.ok) {
        const data = (await res.json()) as { step_sec?: number }
        if (typeof data.step_sec === 'number') setStepSec(data.step_sec)
      }
    }, 400)
    return () => clearTimeout(t)
  }, [missionId, pace, setStepSec])

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
      if (layers.personality) {
        body.personality = personality
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

      const [heatmapRes, nodeFieldsRes, missionRes] = await Promise.all([
        fetch(`${BACKEND_URL}/missions/${data.mission_id}/heatmap`),
        fetch(`${BACKEND_URL}/missions/${data.mission_id}/node-fields`),
        fetch(`${BACKEND_URL}/missions/${data.mission_id}`),
      ])

      let resolvedStepSec: number | undefined
      let resolvedSimulationRunning: boolean | undefined
      if (missionRes.ok) {
        const mission = (await missionRes.json()) as {
          step_sec?: number
          simulation_running?: boolean
        }
        if (typeof mission.step_sec === 'number') resolvedStepSec = mission.step_sec
        if (typeof mission.simulation_running === 'boolean') {
          resolvedSimulationRunning = mission.simulation_running
        }
      }

      if (nodeFieldsRes.ok) {
        const terrain = (await nodeFieldsRes.json()) as TerrainData
        setTerrainData(terrain)
      } else {
        setTerrainData(null)
      }

      if (heatmapRes.ok) {
        const heat = (await heatmapRes.json()) as {
          metadata: GridMetadata
          probabilities: number[]
        }
        setMission(data.mission_id, pinnedLkp, mode, pace, resolvedStepSec)
        setHeatmapFull(heat.metadata, heat.probabilities)
      } else {
        setMission(data.mission_id, pinnedLkp, mode, pace, resolvedStepSec)
      }
      if (resolvedSimulationRunning !== undefined) {
        setSimulationRunning(resolvedSimulationRunning)
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
  }, [pinnedLkp, mode, lkpTimestamp, pace, layers, personality, setMission, setHeatmapFull, setTerrainData, setSimulationRunning])

  const simulatedElapsedSec = tickCount * stepSec

  const togglePauseResume = useCallback(async () => {
    if (!missionId || mode !== 'live') return
    setPauseLoading(true)
    setError(null)
    const endpoint = simulationRunning ? 'pause' : 'resume'
    try {
      const res = await fetch(`${BACKEND_URL}/missions/${missionId}/${endpoint}`, {
        method: 'POST',
      })
      if (!res.ok) throw new Error(await res.text())
      const data = (await res.json()) as { simulation_running?: boolean }
      if (typeof data.simulation_running === 'boolean') {
        setSimulationRunning(data.simulation_running)
      } else {
        setSimulationRunning(!simulationRunning)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : `Failed to ${endpoint} mission`)
    } finally {
      setPauseLoading(false)
    }
  }, [missionId, mode, simulationRunning, setSimulationRunning])

  const newPin = useCallback(async () => {
    if (!missionId) return
    setError(null)
    try {
      await fetch(`${BACKEND_URL}/missions/${missionId}`, { method: 'DELETE' })
    } catch {
      // reset locally even if backend unreachable
    }
    resetMission()
  }, [missionId, resetMission])

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
            <span>
              Pace — {pace.toFixed(1)}× ({Math.round(stepSec)}s sim / tick)
            </span>
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
          <p className="pace-hint">
            Heatmap refreshes every 1 s wall clock. Base {BASE_STEP_SEC}s simulated time per tick;
            pace slider speeds that up (e.g. 6× → {BASE_STEP_SEC * 6}s/tick).
          </p>
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
          <label className="field pace-slider">
            <span>
              Pace - {pace.toFixed(1)}x ({Math.round(stepSec)}s sim / tick)
            </span>
            <input
              type="range"
              min={0.1}
              max={120}
              step={0.1}
              value={pace}
              onChange={(e) => setPace(Number(e.target.value))}
              disabled={!!missionId && mode !== 'offline'}
              aria-label="Offline simulation pace multiplier"
            />
          </label>
          <p className="pace-hint">
            Computes from the last known time, then keeps updating every second at this pace.
          </p>
        </section>
      )}

      {missionId && mode === 'live' && (
        <p className="live-timer" aria-live="polite">
          Simulated time: <strong>{formatDuration(simulatedElapsedSec)}</strong>
          <span className="live-timer-meta">
            {' '}
            ({tickCount} × {Math.round(stepSec)}s)
          </span>
        </p>
      )}

      {missionId && (
        <p className="mission-id">
          Mission: <code>{missionId.slice(0, 8)}…</code>
          {mode === 'live' && simulationRunning && wsStatus === 'open'
            ? ' · live'
            : mode === 'live' && !simulationRunning
              ? ' · paused'
              : wsStatus !== 'open'
                ? ` · ${wsStatus}`
                : ''}
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
          {mode === 'live' && (
            <button
              type="button"
              className={simulationRunning ? 'secondary' : undefined}
              onClick={togglePauseResume}
              disabled={pauseLoading}
            >
              {pauseLoading ? '…' : simulationRunning ? 'Stop' : 'Resume'}
            </button>
          )}
          <button type="button" className="secondary" onClick={newPin}>
            New Pin
          </button>
        </>
      )}
    </div>
  )
}

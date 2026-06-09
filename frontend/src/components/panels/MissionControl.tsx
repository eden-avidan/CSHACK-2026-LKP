import { useCallback, useEffect, useState } from 'react'
import type { Map } from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import { useWebSocket } from '../../hooks/useWebSocket'
import { HAIFA_MAP_VIEW } from '../../types/geo'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

interface MissionControlProps {
  map: Map | null
}

export function MissionControl({ map }: MissionControlProps) {
  const layers = useMissionStore((s) => s.layers)
  const pendingLkp = useMissionStore((s) => s.pendingLkp)
  const missionId = useMissionStore((s) => s.missionId)
  const lkp = useMissionStore((s) => s.lkp)
  const wsStatus = useMissionStore((s) => s.wsStatus)
  const gridVersion = useMissionStore((s) => s.gridVersion)
  const tickCount = useMissionStore((s) => s.tickCount)
  const simulationRunning = useMissionStore((s) => s.simulationRunning)
  const stepSec = useMissionStore((s) => s.stepSec)
  const updateIntervalSec = useMissionStore((s) => s.updateIntervalSec)
  const setMission = useMissionStore((s) => s.setMission)
  const setStepSec = useMissionStore((s) => s.setStepSec)
  const setUpdateIntervalSec = useMissionStore((s) => s.setUpdateIntervalSec)
  const setSimulationRunning = useMissionStore((s) => s.setSimulationRunning)
  const resetMission = useMissionStore((s) => s.resetMission)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [backendOk, setBackendOk] = useState<boolean | null>(null)

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

  const applyPace = useCallback(async () => {
    if (!missionId) return
    await fetch(`${BACKEND_URL}/missions/${missionId}/pace`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        step_sec: stepSec,
        update_interval_sec: updateIntervalSec,
      }),
    })
  }, [missionId, stepSec, updateIntervalSec])

  useEffect(() => {
    if (!missionId) return
    const t = window.setTimeout(() => {
      applyPace()
    }, 400)
    return () => clearTimeout(t)
  }, [missionId, stepSec, updateIntervalSec, applyPace])

  const createMission = useCallback(async () => {
    if (!pendingLkp) {
      setError('Click the map to set Last Known Position')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${BACKEND_URL}/missions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lkp: pendingLkp,
          step_sec: stepSec,
          update_interval_sec: updateIntervalSec,
          layers,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = (await res.json()) as { mission_id: string }
      setMission(data.mission_id, pendingLkp, stepSec, updateIntervalSec)
    } catch (e) {
      if (e instanceof TypeError) {
        setError(
          `Cannot reach backend at ${BACKEND_URL}. From project root: cd backend && source .venv/bin/activate && PYTHONPATH=. uvicorn app.main:app --reload --port 8000`,
        )
      } else {
        setError(e instanceof Error ? e.message : 'Failed to create mission')
      }
    } finally {
      setLoading(false)
    }
  }, [pendingLkp, stepSec, updateIntervalSec, layers, setMission])

  const pauseSimulation = useCallback(async () => {
    if (!missionId) return
    const res = await fetch(`${BACKEND_URL}/missions/${missionId}/pause`, { method: 'POST' })
    if (res.ok) setSimulationRunning(false)
  }, [missionId, setSimulationRunning])

  const resumeSimulation = useCallback(async () => {
    if (!missionId) return
    const res = await fetch(`${BACKEND_URL}/missions/${missionId}/resume`, { method: 'POST' })
    if (res.ok) setSimulationRunning(true)
  }, [missionId, setSimulationRunning])

  const stopAndNewMission = useCallback(async () => {
    if (!missionId) return
    setError(null)
    try {
      await fetch(`${BACKEND_URL}/missions/${missionId}`, { method: 'DELETE' })
    } catch {
      // Still reset locally if backend unreachable
    }
    resetMission()
    if (map) {
      map.flyTo({ center: HAIFA_MAP_VIEW.center, zoom: HAIFA_MAP_VIEW.zoom, duration: 800 })
    }
  }, [missionId, resetMission, map])

  const manualTick = useCallback(async () => {
    if (!missionId) return
    await fetch(`${BACKEND_URL}/missions/${missionId}/tick`, { method: 'POST' })
  }, [missionId])

  return (
    <div className="mission-control">
      <h2>Mission Control</h2>
      <p className="hint">Click map to set LKP, then create mission.</p>

      <p className={`backend-status ${backendOk === false ? 'offline' : backendOk ? 'online' : ''}`}>
        Backend: {backendOk === null ? 'checking…' : backendOk ? 'connected' : 'offline — start backend on port 8000'}
      </p>

      {pendingLkp && !missionId && (
        <p className="coords">
          LKP: {pendingLkp.lat.toFixed(5)}, {pendingLkp.lon.toFixed(5)}
        </p>
      )}

      <div className="pace-controls">
        <label className="field">
          <span>Step duration (min)</span>
          <input
            type="number"
            min={1}
            max={120}
            step={1}
            value={stepSec / 60}
            onChange={(e) => setStepSec(Math.max(1, Number(e.target.value) || 1) * 60)}
            aria-label="Simulated minutes advanced per engine tick"
          />
        </label>
        <label className="field">
          <span>Update every (sec)</span>
          <input
            type="number"
            min={1}
            max={3600}
            step={1}
            value={updateIntervalSec}
            onChange={(e) => setUpdateIntervalSec(Math.max(1, Number(e.target.value) || 60))}
            aria-label="Wall-clock seconds between heatmap updates"
          />
        </label>
      </div>
      <p className="pace-hint">
        Each tick advances the search by the step duration (default 1 minute). Update every controls how
        often the map refreshes in real time.
      </p>

      <div className="sim-status">
        <h3>Simulation Status</h3>
        <p className="ws-status">
          Engine ticks: <strong>{tickCount}</strong>
        </p>
        <p className="ws-status">
          WebSocket: {missionId ? wsStatus : 'idle'}
        </p>
        {missionId && (
          <p className="ws-status">Grid repaints: {gridVersion}</p>
        )}
      </div>

      {lkp && missionId && (
        <>
          <p className="mission-id">
            Mission: <code>{missionId.slice(0, 8)}…</code>
          </p>
          <p className="status">
            Status:{' '}
            {simulationRunning ? (
              <span className="badge searching">Searching</span>
            ) : (
              <span className="badge paused">Paused</span>
            )}
          </p>
          <p className="ws-status">
            Pace: {stepSec / 60} min/step · refresh every {updateIntervalSec}s
          </p>
        </>
      )}

      {error && <p className="error">{error}</p>}

      {!missionId && (
        <button type="button" onClick={createMission} disabled={loading || !pendingLkp || backendOk === false}>
          {loading ? 'Loading terrain & roads…' : 'Create Mission'}
        </button>
      )}

      {missionId && simulationRunning && (
        <button type="button" onClick={pauseSimulation}>
          Pause Simulation
        </button>
      )}

      {missionId && !simulationRunning && (
        <button type="button" onClick={resumeSimulation}>
          Resume Simulation
        </button>
      )}

      {missionId && (
        <button type="button" className="secondary" onClick={stopAndNewMission}>
          Stop & New Mission
        </button>
      )}

      {missionId && (
        <button type="button" className="secondary" onClick={manualTick}>
          Manual Tick (dev)
        </button>
      )}
    </div>
  )
}

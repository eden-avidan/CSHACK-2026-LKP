import { create } from 'zustand'
import type { DroneRoute, GridMetadata } from '../types/geo'
import type { LatLon } from '../types/geo'

export type WsStatus = 'idle' | 'connecting' | 'open' | 'closed' | 'error'
export type MissionMode = 'live' | 'offline'

export interface LayerState {
  topography: boolean
  roads: boolean
  subject_injured: boolean
  weather: boolean
}

export const DEFAULT_LAYERS: LayerState = {
  topography: true,
  roads: false,
  subject_injured: false,
  weather: false,
}

function defaultLkpTimestamp(): string {
  const d = new Date(Date.now() - 2 * 60 * 60 * 1000)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

interface MissionStore {
  missionId: string | null
  status: string
  mode: MissionMode
  lkp: LatLon | null
  mpp: LatLon | null
  mppTrail: LatLon[]
  tickCount: number
  engineTickVersion: number
  wsStatus: WsStatus
  simulationRunning: boolean
  layers: LayerState
  metadata: GridMetadata | null
  grid: Float32Array | null
  gridVersion: number
  droneRoute: DroneRoute | null
  pinnedLkp: LatLon | null
  draftLkp: LatLon | null
  lkpTimestamp: string | null
  pace: number
  wsSend: ((payload: unknown) => void) | null

  setDraftLkp: (lkp: LatLon) => void
  setPinnedLkp: (lkp: LatLon | null) => void
  setMode: (mode: MissionMode) => void
  setPace: (pace: number) => void
  setLkpTimestamp: (ts: string | null) => void
  setMission: (id: string, lkp: LatLon, mode: MissionMode, pace: number) => void
  setSimulationRunning: (running: boolean) => void
  resetMission: () => void
  setWsStatus: (status: WsStatus) => void
  setWsSend: (fn: ((payload: unknown) => void) | null) => void
  setLayers: (layers: Partial<LayerState>) => void
  setEngineTick: (mpp: LatLon, tickCount: number, layers?: Partial<LayerState>) => void
  setHeatmapFull: (metadata: GridMetadata, probabilities: number[]) => void
  applyHeatmapDelta: (cells: { row: number; col: number; probability: number }[]) => void
  setDroneRoute: (route: DroneRoute | null) => void
  setTickCount: (n: number) => void
}

export const useMissionStore = create<MissionStore>((set, get) => ({
  missionId: null,
  status: 'idle',
  mode: 'live',
  lkp: null,
  mpp: null,
  mppTrail: [],
  tickCount: 0,
  engineTickVersion: 0,
  wsStatus: 'idle',
  simulationRunning: true,
  layers: { ...DEFAULT_LAYERS },
  metadata: null,
  grid: null,
  gridVersion: 0,
  droneRoute: null,
  pinnedLkp: null,
  draftLkp: null,
  lkpTimestamp: defaultLkpTimestamp(),
  pace: 1,
  wsSend: null,

  setDraftLkp: (draftLkp) => set({ draftLkp }),

  setPinnedLkp: (pinnedLkp) => set({ pinnedLkp }),

  setMode: (mode) => set({ mode }),

  setPace: (pace) => set({ pace: Math.max(0.1, Math.min(120, pace)) }),

  setLkpTimestamp: (lkpTimestamp) => set({ lkpTimestamp }),

  setMission: (id, lkp, mode, pace) =>
    set((state) => ({
      missionId: id,
      lkp,
      mpp: lkp,
      mppTrail: [lkp],
      status: 'searching',
      mode,
      tickCount: 0,
      engineTickVersion: 0,
      simulationRunning: mode === 'live',
      layers: state.layers,
      pace,
      // Preserve grid/metadata when already loaded (REST prefetch before WS connect)
      metadata: state.metadata,
      grid: state.grid,
      gridVersion: state.gridVersion,
      droneRoute: null,
    })),

  setSimulationRunning: (simulationRunning) => set({ simulationRunning }),

  resetMission: () =>
    set({
      missionId: null,
      status: 'idle',
      mode: 'live',
      lkp: null,
      mpp: null,
      mppTrail: [],
      tickCount: 0,
      engineTickVersion: 0,
      wsStatus: 'idle',
      simulationRunning: true,
      layers: { ...DEFAULT_LAYERS },
      metadata: null,
      grid: null,
      gridVersion: 0,
      droneRoute: null,
      pinnedLkp: null,
      draftLkp: null,
      lkpTimestamp: defaultLkpTimestamp(),
      pace: 1,
      wsSend: null,
    }),

  setWsStatus: (wsStatus) => set({ wsStatus }),

  setWsSend: (wsSend) => set({ wsSend }),

  setLayers: (layers) => {
    const next = { ...get().layers, ...layers }
    if (!Object.values(next).some(Boolean)) {
      next.topography = true
    }
    set({ layers: next })
  },

  setEngineTick: (mpp, tickCount, layers) =>
    set((state) => {
      const trail =
        state.mppTrail.length > 0
          ? [...state.mppTrail]
          : state.lkp
            ? [state.lkp]
            : []
      const idx = tickCount + 1
      if (trail.length <= idx) {
        trail.push(mpp)
      } else {
        trail[idx] = mpp
      }
      return {
        mpp,
        mppTrail: trail,
        tickCount,
        engineTickVersion: state.engineTickVersion + 1,
        ...(layers ? { layers: { ...state.layers, ...layers } as LayerState } : {}),
      }
    }),

  setHeatmapFull: (metadata, probabilities) => {
    const grid = new Float32Array(probabilities)
    set({ metadata, grid, gridVersion: get().gridVersion + 1 })
  },

  applyHeatmapDelta: (cells) => {
    const { grid, metadata } = get()
    if (!grid || !metadata) return
    const next = new Float32Array(grid)
    for (const cell of cells) {
      const idx = cell.row * metadata.cols + cell.col
      if (idx >= 0 && idx < next.length) {
        next[idx] = cell.probability
      }
    }
    set({ grid: next, gridVersion: get().gridVersion + 1 })
  },

  setDroneRoute: (droneRoute) => set({ droneRoute }),

  setTickCount: (tickCount) => set({ tickCount }),
}))

import { create } from 'zustand'
import type { GridMetadata } from '../types/geo'
import type { LatLon } from '../types/geo'

export type WsStatus = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

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

interface MissionStore {
  missionId: string | null
  status: string
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
  pendingLkp: LatLon | null
  stepSec: number
  updateIntervalSec: number
  wsSend: ((payload: unknown) => void) | null

  setPendingLkp: (lkp: LatLon) => void
  setStepSec: (n: number) => void
  setUpdateIntervalSec: (n: number) => void
  setMission: (id: string, lkp: LatLon, stepSec: number, updateIntervalSec: number) => void
  setSimulationRunning: (running: boolean) => void
  resetMission: () => void
  setWsStatus: (status: WsStatus) => void
  setWsSend: (fn: ((payload: unknown) => void) | null) => void
  setLayers: (layers: Partial<LayerState>) => void
  setEngineTick: (mpp: LatLon, tickCount: number, layers?: Partial<LayerState>) => void
  setHeatmapFull: (metadata: GridMetadata, probabilities: number[]) => void
  applyHeatmapDelta: (cells: { row: number; col: number; probability: number }[]) => void
  setTickCount: (n: number) => void
}

export const useMissionStore = create<MissionStore>((set, get) => ({
  missionId: null,
  status: 'idle',
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
  pendingLkp: null,
  stepSec: 60,
  updateIntervalSec: 60,
  wsSend: null,

  setPendingLkp: (lkp) => set({ pendingLkp: lkp }),

  setStepSec: (stepSec) => set({ stepSec }),

  setUpdateIntervalSec: (updateIntervalSec) => set({ updateIntervalSec }),

  setMission: (id, lkp, stepSec, updateIntervalSec) =>
    set((state) => ({
      missionId: id,
      lkp,
      mpp: lkp,
      mppTrail: [lkp],
      status: 'searching',
      tickCount: 0,
      engineTickVersion: 0,
      simulationRunning: true,
      layers: state.layers,
      stepSec,
      updateIntervalSec,
      metadata: null,
      grid: null,
      gridVersion: 0,
    })),

  setSimulationRunning: (simulationRunning) => set({ simulationRunning }),

  resetMission: () =>
    set({
      missionId: null,
      status: 'idle',
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
      pendingLkp: null,
      wsSend: null,
    }),

  setWsStatus: (wsStatus) => set({ wsStatus }),

  setWsSend: (wsSend) => set({ wsSend }),

  setLayers: (layers) =>
    set({
      layers: { ...get().layers, ...layers },
    }),

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

  setTickCount: (tickCount) => set({ tickCount }),
}))

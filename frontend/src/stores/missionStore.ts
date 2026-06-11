import { create } from 'zustand'
import type { GridMetadata } from '../types/geo'
import type { LatLon } from '../types/geo'

export type WsStatus = 'idle' | 'connecting' | 'open' | 'closed' | 'error'
export type MissionMode = 'live' | 'offline'
export type TerrainFieldKind = 'scalar' | 'mask'

export interface TerrainFieldMeta {
  id: string
  label: string
  kind: TerrainFieldKind
  unit?: string
  description?: string
}

export interface TerrainData {
  metadata: GridMetadata
  rows: number
  cols: number
  fields: Record<string, number[]>
  field_stats?: Record<string, { min: number; max: number; nonzero_frac: number }>
  warnings?: string[]
  available: TerrainFieldMeta[]
}

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
  pinnedLkp: LatLon | null
  draftLkp: LatLon | null
  lkpTimestamp: string | null
  pace: number
  wsSend: ((payload: unknown) => void) | null
  terrainData: TerrainData | null
  terrainField: string | null
  terrainVersion: number
  terrainMaskMode: boolean
  terrainMaskThreshold: number

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
  setTickCount: (n: number) => void
  setTerrainData: (data: TerrainData | null) => void
  setTerrainField: (field: string | null) => void
  setTerrainMaskMode: (on: boolean) => void
  setTerrainMaskThreshold: (threshold: number) => void
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
  pinnedLkp: null,
  draftLkp: null,
  lkpTimestamp: defaultLkpTimestamp(),
  pace: 1,
  wsSend: null,
  terrainData: null,
  terrainField: null,
  terrainVersion: 0,
  terrainMaskMode: false,
  terrainMaskThreshold: 0.5,

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
      pinnedLkp: null,
      draftLkp: null,
      lkpTimestamp: defaultLkpTimestamp(),
      pace: 1,
      wsSend: null,
      terrainData: null,
      terrainField: null,
      terrainMaskMode: false,
      terrainMaskThreshold: 0.5,
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

  setTickCount: (tickCount) => set({ tickCount }),

  setTerrainData: (terrainData) =>
    set((state) => {
      if (!terrainData) {
        return {
          terrainData: null,
          terrainField: null,
          terrainVersion: state.terrainVersion + 1,
        }
      }
      const stillValid =
        state.terrainField !== null &&
        terrainData.available.some((field) => field.id === state.terrainField)
      return {
        terrainData,
        terrainField: stillValid ? state.terrainField : (terrainData.available[0]?.id ?? null),
        terrainVersion: state.terrainVersion + 1,
      }
    }),

  setTerrainField: (terrainField) =>
    set((state) => ({ terrainField, terrainVersion: state.terrainVersion + 1 })),

  setTerrainMaskMode: (terrainMaskMode) =>
    set((state) => ({ terrainMaskMode, terrainVersion: state.terrainVersion + 1 })),

  setTerrainMaskThreshold: (terrainMaskThreshold) =>
    set((state) => ({
      terrainMaskThreshold: Math.max(0, Math.min(1, terrainMaskThreshold)),
      terrainVersion: state.terrainVersion + 1,
    })),
}))

import { create } from 'zustand'
import type { DroneRoute, GridMetadata } from '../types/geo'
import type { LatLon } from '../types/geo'
import { BASE_STEP_SEC } from '../utils/formatTime'
import type { DetectionEvent } from '../types/ws-messages'

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
  personality: boolean
  weather: boolean
}

export interface PersonalityProfile {
  age: number
  fitness: number
  injured: boolean
}

export const DEFAULT_PERSONALITY: PersonalityProfile = {
  age: 35,
  fitness: 3,
  injured: false,
}

export const DEFAULT_LAYERS: LayerState = {
  topography: true,
  roads: false,
  personality: false,
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
  liveStartTickCount: number
  engineTickVersion: number
  wsStatus: WsStatus
  simulationRunning: boolean
  layers: LayerState
  personality: PersonalityProfile
  metadata: GridMetadata | null
  grid: Float32Array | null
  gridVersion: number
  droneRoute: DroneRoute | null
  detectionFlash: DetectionEvent | null
  pinnedLkp: LatLon | null
  draftLkp: LatLon | null
  lkpTimestamp: string | null
  pace: number
  stepSec: number
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
  setMission: (
    id: string,
    lkp: LatLon,
    mode: MissionMode,
    pace: number,
    stepSec?: number,
    liveStartTickCount?: number,
  ) => void
  setStepSec: (stepSec: number) => void
  setSimulationRunning: (running: boolean) => void
  resetMission: () => void
  setWsStatus: (status: WsStatus) => void
  setWsSend: (fn: ((payload: unknown) => void) | null) => void
  setLayers: (layers: Partial<LayerState>) => void
  setPersonality: (profile: Partial<PersonalityProfile>) => void
  setEngineTick: (mpp: LatLon, tickCount: number, layers?: Partial<LayerState>) => void
  setHeatmapFull: (metadata: GridMetadata, probabilities: number[]) => void
  applyHeatmapDelta: (cells: { row: number; col: number; probability: number }[]) => void
  setDroneRoute: (route: DroneRoute | null) => void
  setDetectionFlash: (detection: DetectionEvent | null) => void
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
  liveStartTickCount: 0,
  engineTickVersion: 0,
  wsStatus: 'idle',
  simulationRunning: true,
  layers: { ...DEFAULT_LAYERS },
  personality: { ...DEFAULT_PERSONALITY },
  metadata: null,
  grid: null,
  gridVersion: 0,
  droneRoute: null,
  detectionFlash: null,
  pinnedLkp: null,
  draftLkp: null,
  lkpTimestamp: defaultLkpTimestamp(),
  pace: 1,
  stepSec: BASE_STEP_SEC,
  wsSend: null,
  terrainData: null,
  terrainField: null,
  terrainVersion: 0,
  terrainMaskMode: false,
  terrainMaskThreshold: 0.5,

  setDraftLkp: (draftLkp) => set({ draftLkp }),

  setPinnedLkp: (pinnedLkp) => set({ pinnedLkp }),

  setMode: (mode) => set({ mode }),

  setPace: (pace) => {
    const clamped = Math.max(0.1, Math.min(120, pace))
    set({ pace: clamped, stepSec: BASE_STEP_SEC * clamped })
  },

  setStepSec: (stepSec) => set({ stepSec }),

  setLkpTimestamp: (lkpTimestamp) => set({ lkpTimestamp }),

  setMission: (id, lkp, mode, pace, stepSec, liveStartTickCount = 0) =>
    set((state) => ({
      missionId: id,
      lkp,
      mpp: lkp,
      mppTrail: [lkp],
      status: 'searching',
      mode,
      tickCount: liveStartTickCount,
      liveStartTickCount,
      engineTickVersion: 0,
      simulationRunning: mode === 'live',
      layers: state.layers,
      pace,
      stepSec: stepSec ?? BASE_STEP_SEC * pace,
      terrainField: null,
      // Preserve grid/metadata when already loaded (REST prefetch before WS connect)
      metadata: state.metadata,
      grid: state.grid,
      gridVersion: state.gridVersion,
      droneRoute: null,
      detectionFlash: null,
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
      liveStartTickCount: 0,
      engineTickVersion: 0,
      wsStatus: 'idle',
      simulationRunning: true,
      layers: { ...DEFAULT_LAYERS },
      personality: { ...DEFAULT_PERSONALITY },
      metadata: null,
      grid: null,
      gridVersion: 0,
      droneRoute: null,
      detectionFlash: null,
      pinnedLkp: null,
      draftLkp: null,
      lkpTimestamp: defaultLkpTimestamp(),
      pace: 1,
      stepSec: BASE_STEP_SEC,
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

  setPersonality: (profile) =>
    set((state) => ({
      personality: { ...state.personality, ...profile },
    })),

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

  setDetectionFlash: (detectionFlash) => set({ detectionFlash }),

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
        terrainField: stillValid ? state.terrainField : null,
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

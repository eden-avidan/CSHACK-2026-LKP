import { z } from 'zod'

const latLonSchema = z.object({
  lat: z.number(),
  lon: z.number(),
})

const gridBoundsSchema = z.object({
  north: z.number(),
  south: z.number(),
  east: z.number(),
  west: z.number(),
})

const gridMetadataSchema = z.object({
  origin: latLonSchema,
  resolution_m: z.number(),
  rows: z.number(),
  cols: z.number(),
  crs_epsg: z.number(),
  bounds: gridBoundsSchema,
})

const cellDeltaSchema = z.object({
  row: z.number(),
  col: z.number(),
  probability: z.number(),
})

const layerStateSchema = z.object({
  topography: z.boolean(),
  roads: z.boolean(),
  subject_injured: z.boolean(),
  weather: z.boolean(),
})

export const heatmapFullSchema = z.object({
  type: z.literal('heatmap_full'),
  mission_id: z.string(),
  timestamp: z.string(),
  metadata: gridMetadataSchema,
  probabilities: z.array(z.number()),
})

export const heatmapDeltaSchema = z.object({
  type: z.literal('heatmap_delta'),
  mission_id: z.string(),
  timestamp: z.string(),
  cells: z.array(cellDeltaSchema),
})

export const detectionEventSchema = z.object({
  type: z.literal('detection_event'),
  mission_id: z.string(),
  asset_id: z.string(),
  timestamp: z.string(),
  target: z.object({
    lat: z.number(),
    lon: z.number(),
    confidence: z.number(),
  }),
})

export const engineTickSchema = z.object({
  event: z.literal('engine_tick'),
  tick_count: z.number(),
  lkp_coords: latLonSchema,
  mpp_coords: latLonSchema,
  layers: layerStateSchema.optional(),
  particle_matrix: z.array(z.array(z.number())),
})

export const wsMessageSchema = z.union([
  heatmapFullSchema,
  heatmapDeltaSchema,
  detectionEventSchema,
  engineTickSchema,
])

export type WsMessage = z.infer<typeof wsMessageSchema>
export type HeatmapFull = z.infer<typeof heatmapFullSchema>
export type HeatmapDelta = z.infer<typeof heatmapDeltaSchema>
export type EngineTick = z.infer<typeof engineTickSchema>

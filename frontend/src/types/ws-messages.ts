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

const gridCornersSchema = z.object({
  nw: latLonSchema,
  ne: latLonSchema,
  se: latLonSchema,
  sw: latLonSchema,
})

const gridMetadataSchema = z.object({
  origin: latLonSchema,
  resolution_m: z.number(),
  rows: z.number(),
  cols: z.number(),
  crs_epsg: z.number(),
  bounds: gridBoundsSchema,
  corners: gridCornersSchema,
})

const cellDeltaSchema = z.object({
  row: z.number(),
  col: z.number(),
  probability: z.number(),
})

const layerStateSchema = z.object({
  topography: z.boolean(),
  roads: z.boolean(),
  personality: z.boolean(),
  weather: z.boolean(),
  sea_drift: z.boolean(),
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
  person_found: z.literal(true),
  confidence: z.number(),
  confidence_percent: z.number(),
  frame: z.number().nullable().optional(),
  bbox: z.array(z.number()).nullable().optional(),
  position: z.object({
    lat: z.number(),
    lon: z.number(),
  }).nullable().optional(),
})

export const droneTrackItemSchema = z.object({
  asset_id: z.string(),
  found: z.boolean().optional(),
  active: z.boolean().optional(),
  position: latLonSchema.nullable().optional(),
  path: z.array(z.array(z.number())),
})

export const droneTrackSchema = z.object({
  type: z.literal('drone_track'),
  mission_id: z.string(),
  asset_id: z.string().optional(),
  timestamp: z.string(),
  position: latLonSchema.nullable().optional(),
  path: z.array(z.array(z.number())),
  drones: z.array(droneTrackItemSchema).optional(),
})

export const engineTickSchema = z.object({
  event: z.literal('engine_tick'),
  tick_count: z.number(),
  lkp_coords: latLonSchema,
  mpp_coords: latLonSchema,
  layers: layerStateSchema.partial().optional(),
  particle_matrix: z.array(z.array(z.number())).optional(),
})

export const wsMessageSchema = z.union([
  heatmapFullSchema,
  heatmapDeltaSchema,
  detectionEventSchema,
  droneTrackSchema,
  engineTickSchema,
])

export type WsMessage = z.infer<typeof wsMessageSchema>
export type HeatmapFull = z.infer<typeof heatmapFullSchema>
export type HeatmapDelta = z.infer<typeof heatmapDeltaSchema>
export type EngineTick = z.infer<typeof engineTickSchema>
export type DetectionEvent = z.infer<typeof detectionEventSchema>
export type DroneTrack = z.infer<typeof droneTrackSchema>
export type DroneTrackItem = z.infer<typeof droneTrackItemSchema>

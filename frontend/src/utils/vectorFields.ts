import type { GridMetadata } from '../types/geo'
import type { TerrainData } from '../stores/missionStore'

type GeoFeature = GeoJSON.Feature<GeoJSON.Geometry>
type FeatureCollection = GeoJSON.FeatureCollection

const MIN_SPEED = 0.02
const BASE_LENGTH_M = 600

export const WIND_ARROW_COLOR = '#a3e635'
export const CURRENT_ARROW_COLOR = '#22d3ee'
export const LKP_WIND_COLOR = '#facc15'
export const LKP_CURRENT_COLOR = '#fb923c'

function cellLatLon(meta: GridMetadata, row: number, col: number): [number, number] {
  const { bounds, rows, cols } = meta
  const lon = bounds.west + ((col + 0.5) / cols) * (bounds.east - bounds.west)
  const lat = bounds.north - ((row + 0.5) / rows) * (bounds.north - bounds.south)
  return [lon, lat]
}

function metersToDelta(lat: number, uEast: number, vNorth: number, lengthM: number): [number, number] {
  const speed = Math.hypot(uEast, vNorth)
  if (speed < 1e-9) return [0, 0]
  const scale = lengthM / speed
  const mPerDegLat = 111_320
  const mPerDegLon = 111_320 * Math.cos((lat * Math.PI) / 180)
  return [(uEast * scale) / mPerDegLon, (vNorth * scale) / mPerDegLat]
}

function arrowHead(
  tipLon: number,
  tipLat: number,
  uEast: number,
  vNorth: number,
  headLenM: number,
  color: string,
): GeoFeature[] {
  const speed = Math.hypot(uEast, vNorth)
  if (speed < 1e-9) return []
  const ux = uEast / speed
  const uy = vNorth / speed
  const wing = 0.42
  const leftU = -ux * wing - uy * wing
  const leftV = -uy * wing + ux * wing
  const rightU = -ux * wing + uy * wing
  const rightV = -uy * wing - ux * wing
  const [dLeftLon, dLeftLat] = metersToDelta(tipLat, leftU, leftV, headLenM)
  const [dRightLon, dRightLat] = metersToDelta(tipLat, rightU, rightV, headLenM)
  return [
    {
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: [
          [tipLon - dLeftLon, tipLat - dLeftLat],
          [tipLon, tipLat],
          [tipLon - dRightLon, tipLat - dRightLat],
        ],
      },
      properties: { kind: 'head', color },
    },
  ]
}

function shaftFeature(
  lon0: number,
  lat0: number,
  uEast: number,
  vNorth: number,
  lengthM: number,
  speed: number,
  color: string,
  kind: string = 'shaft',
): GeoFeature[] {
  const [dLon, dLat] = metersToDelta(lat0, uEast, vNorth, lengthM)
  const tipLon = lon0 + dLon
  const tipLat = lat0 + dLat
  const shaft: GeoFeature = {
    type: 'Feature',
    geometry: {
      type: 'LineString',
      coordinates: [
        [lon0, lat0],
        [tipLon, tipLat],
      ],
    },
    properties: { kind, speed, color },
  }
  return [shaft, ...arrowHead(tipLon, tipLat, uEast, vNorth, lengthM * 0.22, color)]
}

function buildVectorGeoJson(
  data: TerrainData,
  uKey: 'wind_u' | 'current_u',
  vKey: 'wind_v' | 'current_v',
  opts: {
    cellFilter: (land: number | undefined, idx: number) => boolean
    arrowColor: string
    lkpColor: string
    lkpVector?: { u: number; v: number; speed: number } | null
  },
): FeatureCollection {
  const { metadata, rows, cols, fields } = data
  const u = fields[uKey]
  const v = fields[vKey]
  const land = fields.is_land
  if (!u || !v) {
    return { type: 'FeatureCollection', features: [] }
  }

  let maxSpeed = MIN_SPEED
  for (let i = 0; i < u.length; i++) {
    if (!opts.cellFilter(land?.[i], i)) continue
    maxSpeed = Math.max(maxSpeed, Math.hypot(u[i], v[i]))
  }
  if (opts.lkpVector) {
    maxSpeed = Math.max(maxSpeed, opts.lkpVector.speed)
  }

  const step = Math.max(1, Math.floor(Math.min(rows, cols) / 14))
  const features: GeoFeature[] = []

  for (let row = 0; row < rows; row += step) {
    for (let col = 0; col < cols; col += step) {
      const idx = row * cols + col
      if (!opts.cellFilter(land?.[idx], idx)) continue
      const uVal = u[idx]
      const vVal = v[idx]
      const speed = Math.hypot(uVal, vVal)
      if (speed < MIN_SPEED) continue
      const [lon, lat] = cellLatLon(metadata, row, col)
      const lengthM = BASE_LENGTH_M * (0.35 + 0.65 * (speed / maxSpeed))
      features.push(...shaftFeature(lon, lat, uVal, vVal, lengthM, speed, opts.arrowColor))
    }
  }

  const centerRow = Math.floor(rows / 2)
  const centerCol = Math.floor(cols / 2)
  const centerIdx = centerRow * cols + centerCol
  const lkpU = opts.lkpVector?.u ?? u[centerIdx]
  const lkpV = opts.lkpVector?.v ?? v[centerIdx]
  const lkpSpeed = opts.lkpVector?.speed ?? Math.hypot(lkpU, lkpV)
  if (lkpSpeed >= MIN_SPEED) {
    const [lon, lat] = cellLatLon(metadata, centerRow, centerCol)
    const lengthM = BASE_LENGTH_M * 1.35
    features.push(
      ...shaftFeature(lon, lat, lkpU, lkpV, lengthM, lkpSpeed, opts.lkpColor, 'lkp'),
    )
  }

  return { type: 'FeatureCollection', features }
}

/** Arrow overlay for per-cell wind across the full grid (spatial mock W → N). */
export function buildWindVectorGeoJson(data: TerrainData): FeatureCollection {
  return buildVectorGeoJson(data, 'wind_u', 'wind_v', {
    cellFilter: () => true,
    arrowColor: WIND_ARROW_COLOR,
    lkpColor: LKP_WIND_COLOR,
  })
}

/** Arrow overlay for sea-surface current on water cells. */
export function buildCurrentVectorGeoJson(data: TerrainData): FeatureCollection {
  const marine = data.marine_current
  return buildVectorGeoJson(data, 'current_u', 'current_v', {
    cellFilter: (land) => land === undefined || land < 0.5,
    arrowColor: CURRENT_ARROW_COLOR,
    lkpColor: LKP_CURRENT_COLOR,
    lkpVector: marine
      ? { u: marine.u_east_mps, v: marine.v_north_mps, speed: marine.speed_mps }
      : null,
  })
}

export function buildVectorFieldGeoJson(
  data: TerrainData,
  fieldId: string,
): FeatureCollection {
  if (fieldId === 'wind_vectors') return buildWindVectorGeoJson(data)
  if (fieldId === 'current_vectors') return buildCurrentVectorGeoJson(data)
  return { type: 'FeatureCollection', features: [] }
}

export function formatMarineCurrentSummary(marine: TerrainData['marine_current']): string | null {
  if (!marine) return null
  const src = marine.source === 'open_meteo' ? 'Open-Meteo' : 'fallback'
  return `${marine.speed_mps.toFixed(2)} m/s toward ${marine.direction_deg.toFixed(0)}° (${src})`
}

import type { GridMetadata } from '../types/geo'
import type { TerrainData } from '../stores/missionStore'

type GeoFeature = GeoJSON.Feature<GeoJSON.Geometry>
type FeatureCollection = GeoJSON.FeatureCollection

const MIN_SPEED = 0.02
const BASE_LENGTH_M = 600

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
      properties: { kind: 'head' },
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
    properties: { kind: 'shaft', speed },
  }
  return [shaft, ...arrowHead(tipLon, tipLat, uEast, vNorth, lengthM * 0.22)]
}

/** Build Mapbox GeoJSON arrow features for water-cell current u/v fields. */
export function buildCurrentVectorGeoJson(data: TerrainData): FeatureCollection {
  const { metadata, rows, cols, fields, marine_current: marine } = data
  const u = fields.current_u
  const v = fields.current_v
  const land = fields.is_land
  if (!u || !v) {
    return { type: 'FeatureCollection', features: [] }
  }

  let maxSpeed = MIN_SPEED
  for (let i = 0; i < u.length; i++) {
    const isWater = !land || land[i] < 0.5
    if (!isWater) continue
    maxSpeed = Math.max(maxSpeed, Math.hypot(u[i], v[i]))
  }
  if (marine) {
    maxSpeed = Math.max(maxSpeed, marine.speed_mps)
  }

  const step = Math.max(1, Math.floor(Math.min(rows, cols) / 14))
  const features: GeoFeature[] = []

  for (let row = 0; row < rows; row += step) {
    for (let col = 0; col < cols; col += step) {
      const idx = row * cols + col
      const isWater = !land || land[idx] < 0.5
      if (!isWater) continue
      const uVal = u[idx]
      const vVal = v[idx]
      const speed = Math.hypot(uVal, vVal)
      if (speed < MIN_SPEED) continue
      const [lon, lat] = cellLatLon(metadata, row, col)
      const lengthM = BASE_LENGTH_M * (0.35 + 0.65 * (speed / maxSpeed))
      features.push(...shaftFeature(lon, lat, uVal, vVal, lengthM, speed))
    }
  }

  // Prominent LKP reference arrow (live API vector at pin).
  if (marine && marine.speed_mps >= MIN_SPEED) {
    const centerRow = Math.floor(rows / 2)
    const centerCol = Math.floor(cols / 2)
    const [lon, lat] = cellLatLon(metadata, centerRow, centerCol)
    const lengthM = BASE_LENGTH_M * 1.35
    features.push(
      ...shaftFeature(lon, lat, marine.u_east_mps, marine.v_north_mps, lengthM, marine.speed_mps).map(
        (f) => ({
          ...f,
          properties: { ...f.properties, kind: 'lkp', source: marine.source },
        }),
      ),
    )
  }

  return { type: 'FeatureCollection', features }
}

export function formatMarineCurrentSummary(marine: TerrainData['marine_current']): string | null {
  if (!marine) return null
  const src = marine.source === 'open_meteo' ? 'Open-Meteo' : 'fallback'
  return `${marine.speed_mps.toFixed(2)} m/s toward ${marine.direction_deg.toFixed(0)}° (${src})`
}

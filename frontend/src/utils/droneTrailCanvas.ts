import type { Map as MapboxMap } from 'mapbox-gl'

/** Active sortie comet-tail length (wall clock). */
export const TRAIL_FADE_MS = 12_000
/** Finished sortie trail + icon fade out faster. */
export const INACTIVE_TRAIL_FADE_MS = 7_000

const TRAIL_GLOW_WIDTH = 10
const TRAIL_CORE_WIDTH = 3.5
const TRAIL_GLOW_ALPHA = 0.42
const TRAIL_CORE_ALPHA = 0.98

export type TrailSegment = {
  fromLon: number
  fromLat: number
  toLon: number
  toLat: number
  revealWallMs: number
  found: boolean
}

const TRAIL_COLORS = {
  clean: { core: [74, 222, 128] as const, glow: [34, 197, 94] as const },
  found: { core: [251, 191, 36] as const, glow: [245, 158, 11] as const },
  blue: { core: [56, 189, 248] as const, glow: [14, 165, 233] as const },
}

export const SECOND_DRONE_ASSET_ID = 'drone-2'

function trailPalette(assetId: string, found: boolean) {
  if (assetId === SECOND_DRONE_ASSET_ID) return TRAIL_COLORS.blue
  return found ? TRAIL_COLORS.found : TRAIL_COLORS.clean
}

export function trailOpacity(
  revealWallMs: number,
  nowMs: number,
  fastFade = false,
): number {
  const fadeMs = fastFade ? INACTIVE_TRAIL_FADE_MS : TRAIL_FADE_MS
  const age = nowMs - revealWallMs
  if (age >= fadeMs) return 0
  return Math.max(0, 1 - age / fadeMs)
}

function coordsEqual(
  aLon: number,
  aLat: number,
  bLon: number,
  bLat: number,
  eps = 1e-9,
): boolean {
  return Math.abs(aLon - bLon) < eps && Math.abs(aLat - bLat) < eps
}

function syncTrailSegments(
  cache: Map<string, TrailSegment[]>,
  assetId: string,
  path: number[][],
  found: boolean,
  nowMs: number,
  stepSec: number,
): TrailSegment[] {
  let prev = cache.get(assetId) ?? []
  if (path.length < 2) {
    cache.set(assetId, [])
    return []
  }

  if (
    prev.length > 0 &&
    !coordsEqual(prev[0].fromLon, prev[0].fromLat, path[0][0], path[0][1])
  ) {
    prev = []
    cache.set(assetId, [])
  }

  const segmentCount = path.length - 1
  if (segmentCount < prev.length) {
    prev = []
    cache.set(assetId, [])
  }

  const segments: TrailSegment[] = []
  for (let i = 0; i < segmentCount; i++) {
    const fromLon = path[i][0]
    const fromLat = path[i][1]
    const toLon = path[i + 1][0]
    const toLat = path[i + 1][1]

    if (i < prev.length) {
      const cached = prev[i]
      if (
        coordsEqual(cached.fromLon, cached.fromLat, fromLon, fromLat) &&
        coordsEqual(cached.toLon, cached.toLat, toLon, toLat)
      ) {
        segments.push(
          i === segmentCount - 1
            ? { ...cached, toLon, toLat, found }
            : cached,
        )
        continue
      }
      prev = prev.slice(0, i)
      cache.set(assetId, prev)
    }

    const newCount = segmentCount - prev.length
    const indexInBatch = i - prev.length
    const ageOffset = (newCount - 1 - indexInBatch) * stepSec

    segments.push({
      fromLon,
      fromLat,
      toLon,
      toLat,
      revealWallMs: nowMs - ageOffset * stepSec * 1000,
      found,
    })
  }

  cache.set(assetId, segments)
  return segments
}

function rgba([r, g, b]: readonly [number, number, number], alpha: number): string {
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

function drawSegment(
  ctx: CanvasRenderingContext2D,
  from: { x: number; y: number },
  to: { x: number; y: number },
  assetId: string,
  found: boolean,
  opacity: number,
): void {
  if (opacity <= 0.01) return
  const palette = trailPalette(assetId, found)

  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  ctx.beginPath()
  ctx.moveTo(from.x, from.y)
  ctx.lineTo(to.x, to.y)
  ctx.strokeStyle = rgba(palette.glow, opacity * TRAIL_GLOW_ALPHA)
  ctx.lineWidth = TRAIL_GLOW_WIDTH
  ctx.stroke()

  ctx.beginPath()
  ctx.moveTo(from.x, from.y)
  ctx.lineTo(to.x, to.y)
  ctx.strokeStyle = rgba(palette.core, opacity * TRAIL_CORE_ALPHA)
  ctx.lineWidth = TRAIL_CORE_WIDTH
  ctx.stroke()
}

export type DroneTrailDraw = {
  assetId: string
  segments: TrailSegment[]
  headLon: number
  headLat: number
  found: boolean
  active: boolean
}

export function markerOpacityForTrail(
  draw: DroneTrailDraw | undefined,
  nowMs: number,
): number {
  if (!draw) return 0
  if (draw.active) return 1
  if (draw.segments.length === 0) return 0
  const newest = draw.segments[draw.segments.length - 1]
  return trailOpacity(newest.revealWallMs, nowMs, true)
}

export function drawDroneTrails(
  map: MapboxMap,
  ctx: CanvasRenderingContext2D,
  trails: DroneTrailDraw[],
  nowMs: number = performance.now(),
): void {
  const dpr = window.devicePixelRatio || 1
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, map.getCanvas().clientWidth, map.getCanvas().clientHeight)

  for (const trail of trails) {
    const fastFade = !trail.active
    for (const seg of trail.segments) {
      const opacity = trailOpacity(seg.revealWallMs, nowMs, fastFade)
      if (opacity <= 0.01) continue

      const from = map.project([seg.fromLon, seg.fromLat])
      const to = map.project([seg.toLon, seg.toLat])
      drawSegment(ctx, from, to, trail.assetId, seg.found, opacity)
    }
  }
}

export type DroneTrailInput = {
  asset_id: string
  found: boolean
  active?: boolean
  position?: { lon: number; lat: number } | null
  path: number[][]
}

export function normalizeDronePath(d: DroneTrailInput): {
  path: number[][]
  anchor: { lon: number; lat: number }
} | null {
  if (!d.position && d.path.length === 0) return null

  let path = d.path.map((pt) => [pt[0], pt[1]] as [number, number])

  if (d.position) {
    if (path.length === 0) {
      path = [[d.position.lon, d.position.lat]]
    } else {
      const [lon, lat] = path[path.length - 1]
      if (!coordsEqual(lon, lat, d.position.lon, d.position.lat)) {
        path = [...path, [d.position.lon, d.position.lat]]
      } else {
        path[path.length - 1] = [d.position.lon, d.position.lat]
      }
    }
  }

  const [lon, lat] = path[path.length - 1]
  return { path, anchor: { lon, lat } }
}

export function buildTrailDraws(
  cache: Map<string, TrailSegment[]>,
  drones: DroneTrailInput[],
  nowMs: number,
  stepSec: number,
): DroneTrailDraw[] {
  const active = new Set(drones.map((d) => d.asset_id))
  for (const id of cache.keys()) {
    if (!active.has(id)) cache.delete(id)
  }

  return drones.flatMap((d) => {
    const normalized = normalizeDronePath(d)
    if (!normalized || normalized.path.length < 2) return []

    const { path, anchor } = normalized
    const segments = syncTrailSegments(
      cache,
      d.asset_id,
      path,
      !!d.found,
      nowMs,
      stepSec,
    )
    if (segments.length === 0) return []

    const lastIdx = segments.length - 1
    segments[lastIdx] = {
      ...segments[lastIdx],
      toLon: anchor.lon,
      toLat: anchor.lat,
    }

    return [
      {
        assetId: d.asset_id,
        segments,
        headLon: anchor.lon,
        headLat: anchor.lat,
        found: !!d.found,
        active: d.active !== false,
      },
    ]
  })
}

export function mountTrailCanvas(map: MapboxMap): {
  canvas: HTMLCanvasElement
  ctx: CanvasRenderingContext2D
  resize: () => void
  remove: () => void
} {
  const canvas = document.createElement('canvas')
  canvas.className = 'drone-trail-canvas'
  canvas.style.position = 'absolute'
  canvas.style.top = '0'
  canvas.style.left = '0'
  canvas.style.width = '100%'
  canvas.style.height = '100%'
  canvas.style.pointerEvents = 'none'
  canvas.style.zIndex = '2'

  const container = map.getCanvasContainer()
  container.appendChild(canvas)

  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('2D canvas context unavailable')

  const resize = () => {
    const dpr = window.devicePixelRatio || 1
    const w = map.getCanvas().clientWidth
    const h = map.getCanvas().clientHeight
    canvas.width = Math.max(1, Math.floor(w * dpr))
    canvas.height = Math.max(1, Math.floor(h * dpr))
  }

  resize()

  return { canvas, ctx, resize, remove: () => canvas.remove() }
}

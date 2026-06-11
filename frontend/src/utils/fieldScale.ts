import type { TerrainFieldKind } from '../stores/missionStore'

const EPS = 1e-9

export interface FieldRange {
  min: number
  max: number
}

export interface FieldRenderOptions {
  forceMask?: boolean
  maskThreshold?: number // normalized 0..1 on scalar fields
}

const CONTINUOUS_TERRAIN_FIELDS = new Set([
  'elevation',
  'slope',
  'reachability',
  'reachability_rel',
  'road_proximity',
  'current_speed',
  'current_heading',
])

/** Min/max over finite values. */
export function computeFieldRange(values: number[], fieldId?: string): FieldRange {
  let min = Infinity
  let max = -Infinity
  const usePercentile =
    fieldId === 'elevation' || fieldId === 'slope' || fieldId === 'reachability'
  if (usePercentile) {
    const finite = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
    if (finite.length === 0) return { min: 0, max: 1 }
    const lo = finite[Math.floor(finite.length * 0.02)] ?? finite[0]
    const hi = finite[Math.floor(finite.length * 0.98)] ?? finite[finite.length - 1]
    if (hi - lo < EPS) return { min: lo, max: lo + 1 }
    return { min: lo, max: hi }
  }
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) continue
    if (v < min) min = v
    if (v > max) max = v
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 }
  if (max - min < EPS) return { min, max: min + 1 }
  return { min, max }
}

type RGB = [number, number, number]

// Turbo-like multi-stop ramp: low = blue, mid = green/yellow, high = red.
const STOPS: { t: number; rgb: RGB }[] = [
  { t: 0.0, rgb: [48, 18, 153] },
  { t: 0.25, rgb: [33, 144, 230] },
  { t: 0.5, rgb: [38, 200, 130] },
  { t: 0.7, rgb: [222, 214, 53] },
  { t: 0.85, rgb: [243, 138, 35] },
  { t: 1.0, rgb: [200, 30, 30] },
]

function ramp(t: number): RGB {
  const clamped = Math.max(0, Math.min(1, t))
  for (let i = 1; i < STOPS.length; i++) {
    if (clamped <= STOPS[i].t) {
      const a = STOPS[i - 1]
      const b = STOPS[i]
      const f = (clamped - a.t) / (b.t - a.t || 1)
      return [
        Math.round(a.rgb[0] + (b.rgb[0] - a.rgb[0]) * f),
        Math.round(a.rgb[1] + (b.rgb[1] - a.rgb[1]) * f),
        Math.round(a.rgb[2] + (b.rgb[2] - a.rgb[2]) * f),
      ]
    }
  }
  return STOPS[STOPS.length - 1].rgb
}

/**
 * Map one field value to RGBA.
 * - scalar: ramp colormap; near-zero values fade to transparent.
 * - mask: highlight cells; `is_land` is inverted so WATER is highlighted.
 */
export function fieldValueToRGBA(
  value: number,
  range: FieldRange,
  kind: TerrainFieldKind,
  fieldId: string,
  options?: FieldRenderOptions,
): [number, number, number, number] {
  const forceMask = options?.forceMask ?? false
  const maskThreshold = Math.max(0, Math.min(1, options?.maskThreshold ?? 0.5))
  if (kind === 'mask' || forceMask) {
    if (forceMask && kind !== 'mask') {
      const t = (value - range.min) / (range.max - range.min || 1)
      if (t < maskThreshold) return [0, 0, 0, 0]
      return [255, 90, 50, 190]
    }
    const on = fieldId === 'is_land' ? value < 0.5 : value > 0.5
    if (!on) return [0, 0, 0, 0]
    if (fieldId === 'is_land') return [30, 120, 220, 150] // water highlight
    return [255, 70, 200, 170] // road / generic mask highlight
  }

  const t = (value - range.min) / (range.max - range.min || 1)
  const clampedT = Math.max(0, Math.min(1, t))
  if (clampedT <= 0.01 && !CONTINUOUS_TERRAIN_FIELDS.has(fieldId)) return [0, 0, 0, 0]
  const [r, g, b] = ramp(clampedT)
  const alpha = Math.min(0.85, 0.35 + clampedT * 0.55)
  return [r, g, b, Math.round(alpha * 255)]
}

function sampleBilinear(
  values: number[],
  rows: number,
  cols: number,
  rowF: number,
  colF: number,
): number {
  const r0 = Math.max(0, Math.min(rows - 1, Math.floor(rowF)))
  const c0 = Math.max(0, Math.min(cols - 1, Math.floor(colF)))
  const r1 = Math.min(rows - 1, r0 + 1)
  const c1 = Math.min(cols - 1, c0 + 1)
  const dr = rowF - r0
  const dc = colF - c0
  const v00 = values[r0 * cols + c0]
  const v01 = values[r0 * cols + c1]
  const v10 = values[r1 * cols + c0]
  const v11 = values[r1 * cols + c1]
  return (
    v00 * (1 - dr) * (1 - dc) +
    v01 * (1 - dr) * dc +
    v10 * dr * (1 - dc) +
    v11 * dr * dc
  )
}

const UPSCALE = 3

/** Render a field array onto a canvas (upscaled). Masks use nearest-neighbor. */
export function paintField(
  values: number[],
  rows: number,
  cols: number,
  kind: TerrainFieldKind,
  fieldId: string,
  canvas: HTMLCanvasElement,
  options?: FieldRenderOptions,
): void {
  const range = computeFieldRange(values, fieldId)
  const outRows = rows * UPSCALE
  const outCols = cols * UPSCALE
  if (canvas.width !== outCols || canvas.height !== outRows) {
    canvas.width = outCols
    canvas.height = outRows
  }
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  const img = ctx.createImageData(outCols, outRows)

  for (let outRow = 0; outRow < outRows; outRow++) {
    const rowF = (outRow + 0.5) / UPSCALE - 0.5
    for (let outCol = 0; outCol < outCols; outCol++) {
      const colF = (outCol + 0.5) / UPSCALE - 0.5
      let value: number
      if (kind === 'mask' || options?.forceMask) {
        const r = Math.max(0, Math.min(rows - 1, Math.round(rowF)))
        const c = Math.max(0, Math.min(cols - 1, Math.round(colF)))
        value = values[r * cols + c]
      } else {
        value = sampleBilinear(values, rows, cols, rowF, colF)
      }
      const [r, g, b, a] = fieldValueToRGBA(value, range, kind, fieldId, options)
      const off = (outRow * outCols + outCol) * 4
      img.data[off] = r
      img.data[off + 1] = g
      img.data[off + 2] = b
      img.data[off + 3] = a
    }
  }
  ctx.putImageData(img, 0, 0)
}

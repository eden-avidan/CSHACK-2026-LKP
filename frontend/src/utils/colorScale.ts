const BLUE: [number, number, number] = [33, 102, 172]
const YELLOW: [number, number, number] = [253, 231, 37]
const ORANGE: [number, number, number] = [194, 85, 10]
const RED: [number, number, number] = [158, 10, 14]
const DARK_RED: [number, number, number] = [58, 4, 8]

const EPS = 1e-8

/** Gentle stretch — smooth ramp, not banded steps. */
const COLOR_GAMMA = 0.72

/** Blue tail stays light; dark-red peak is fully opaque. */
const ALPHA_MIN = 0.14
const ALPHA_MAX = 1.0

/** Low → high: blue → yellow → orange → red → dark red. */
const COLOR_STOPS: Array<{ t: number; rgb: [number, number, number] }> = [
  { t: 0, rgb: BLUE },
  { t: 0.25, rgb: YELLOW },
  { t: 0.5, rgb: ORANGE },
  { t: 0.75, rgb: RED },
  { t: 1, rgb: DARK_RED },
]

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function lerpColor(c1: [number, number, number], c2: [number, number, number], t: number): [number, number, number] {
  return [
    Math.round(lerp(c1[0], c2[0], t)),
    Math.round(lerp(c1[1], c2[1], t)),
    Math.round(lerp(c1[2], c2[2], t)),
  ]
}

/** Smooth five-stop ramp with eased transitions between hues. */
function fiveStopRamp(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t))

  for (let i = 0; i < COLOR_STOPS.length - 1; i++) {
    const a = COLOR_STOPS[i]
    const b = COLOR_STOPS[i + 1]
    if (x <= b.t) {
      const span = b.t - a.t
      const local = span <= EPS ? 1 : (x - a.t) / span
      const eased = local * local * (3 - 2 * local)
      return lerpColor(a.rgb, b.rgb, eased)
    }
  }

  return DARK_RED
}

/** Opacity rises smoothly toward the warm end. */
function alphaForValue(colorT: number): number {
  const warm = Math.max(0, colorT)
  return Math.min(ALPHA_MAX, ALPHA_MIN + Math.pow(warm, 1.15) * (ALPHA_MAX - ALPHA_MIN))
}

export interface ColorRange {
  min: number
  max: number
}

/** Min/max of the grid — used to map every cell into [0, 1] before coloring. */
export function computeColorRange(grid: Float32Array): ColorRange {
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < grid.length; i++) {
    const v = grid[i]
    if (v < min) min = v
    if (v > max) max = v
  }
  if (!Number.isFinite(min) || !Number.isFinite(max) || max - min <= EPS) {
    return { min: 0, max: 1 }
  }
  return { min, max }
}

/** Map a raw cell value to [0, 1] using the grid-wide min/max. */
export function normalizeToUnit(value: number, range: ColorRange): number {
  const span = range.max - range.min
  if (span <= EPS) return 0
  return Math.max(0, Math.min(1, (value - range.min) / span))
}

export function probabilityToRGBA(p: number, range: ColorRange): [number, number, number, number] {
  const linear = normalizeToUnit(p, range)
  if (linear <= EPS) return [0, 0, 0, 0]

  const t = Math.pow(linear, COLOR_GAMMA)
  const rgb = fiveStopRamp(t)
  const alpha = alphaForValue(t)

  return [rgb[0], rgb[1], rgb[2], Math.round(alpha * 255)]
}

export function gridMax(grid: Float32Array): number {
  let max = EPS
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > max) max = grid[i]
  }
  return max
}

export function gridMin(grid: Float32Array): number {
  let min = Infinity
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] < min) min = grid[i]
  }
  return Number.isFinite(min) ? min : 0
}

/** Returns true when non-zero cell count is below 1% of grid (possible boundary loss). */
export function isLowMassGrid(grid: Float32Array): boolean {
  let nonZero = 0
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > EPS) nonZero++
  }
  return nonZero > 0 && nonZero / grid.length < 0.01
}

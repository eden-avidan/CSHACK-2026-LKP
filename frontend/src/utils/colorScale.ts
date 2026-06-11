const BLUE: [number, number, number] = [33, 102, 172]
const YELLOW: [number, number, number] = [253, 231, 37]
const RED: [number, number, number] = [178, 24, 43]
const EPS = 1e-8

/** Stretch mid/low values so the blue→yellow→red ramp is more visible. */
const COLOR_GAMMA = 0.45

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

  // Gamma < 1 pushes more of the dynamic range into saturated hues.
  const t = Math.pow(linear, COLOR_GAMMA)

  let rgb: [number, number, number]
  if (t < 0.5) {
    rgb = lerpColor(BLUE, YELLOW, t * 2)
  } else {
    rgb = lerpColor(YELLOW, RED, (t - 0.5) * 2)
  }

  const alpha = Math.min(0.92, 0.12 + linear * 0.88)
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

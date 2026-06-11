const BLUE: [number, number, number] = [33, 102, 172]
const YELLOW: [number, number, number] = [253, 231, 37]
const RED: [number, number, number] = [178, 24, 43]
const EPS = 1e-8

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
  max: number
  logMin: number
  logMax: number
}

/** Scale colors against grid peak so the tail fades out naturally — no vignette needed. */
export function computeColorRange(grid: Float32Array): ColorRange {
  let max = EPS
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > max) max = grid[i]
  }
  return { max, logMin: Math.log(EPS), logMax: Math.log(max) }
}

export function probabilityToRGBA(p: number, range: ColorRange): [number, number, number, number] {
  if (p < EPS) return [0, 0, 0, 0]

  const rel = p / range.max
  if (rel < 0.008) return [0, 0, 0, 0]

  const span = range.logMax - range.logMin
  const t = span > 1e-12 ? (Math.log(p + EPS) - range.logMin) / span : 1
  const clamped = Math.max(0, Math.min(1, t))

  let rgb: [number, number, number]
  if (clamped < 0.5) {
    rgb = lerpColor(BLUE, YELLOW, clamped * 2)
  } else {
    rgb = lerpColor(YELLOW, RED, (clamped - 0.5) * 2)
  }

  const alpha = Math.min(0.75, Math.pow(rel, 0.55) * 0.8)
  return [rgb[0], rgb[1], rgb[2], Math.round(alpha * 255)]
}

export function gridMax(grid: Float32Array): number {
  let max = EPS
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > max) max = grid[i]
  }
  return max
}

/** Returns true when non-zero cell count is below 1% of grid (possible boundary loss). */
export function isLowMassGrid(grid: Float32Array): boolean {
  let nonZero = 0
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > EPS) nonZero++
  }
  return nonZero > 0 && nonZero / grid.length < 0.01
}

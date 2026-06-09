const BLUE: [number, number, number] = [33, 102, 172]
const YELLOW: [number, number, number] = [253, 231, 37]
const RED: [number, number, number] = [178, 24, 43]
const EPS = 1e-8

/** Smooth fade to transparent near grid edges (matches backend kde_edge_fade_cells). */
export function edgeFadeMultiplier(
  row: number,
  col: number,
  rows: number,
  cols: number,
  fadeCells = 22,
): number {
  const dist = Math.min(row, col, rows - 1 - row, cols - 1 - col)
  if (dist >= fadeCells) return 1
  const t = dist / fadeCells
  return t * t * (3 - 2 * t)
}

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
  logMin: number
  logMax: number
}

/** Percentile stretch so low/mid/high probability cells get distinct colors. */
export function computeColorRange(grid: Float32Array): ColorRange {
  const values: number[] = []
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > EPS) values.push(grid[i])
  }

  if (values.length === 0) {
    return { logMin: Math.log(EPS), logMax: Math.log(EPS) }
  }

  values.sort((a, b) => a - b)
  const p5 = values[Math.floor(values.length * 0.05)] ?? values[0]
  const p95 = values[Math.floor(values.length * 0.95)] ?? values[values.length - 1]
  const lo = Math.max(p5, EPS)
  const hi = Math.max(p95, lo * 10)

  return { logMin: Math.log(lo), logMax: Math.log(hi) }
}

export function probabilityToRGBA(p: number, range: ColorRange): [number, number, number, number] {
  if (p < EPS) return [0, 0, 0, 0]

  const span = range.logMax - range.logMin
  const t = span > 1e-12 ? (Math.log(p + EPS) - range.logMin) / span : 1
  const clamped = Math.max(0, Math.min(1, t))

  let rgb: [number, number, number]
  if (clamped < 0.5) {
    rgb = lerpColor(BLUE, YELLOW, clamped * 2)
  } else {
    rgb = lerpColor(YELLOW, RED, (clamped - 0.5) * 2)
  }

  // Low-probability cells stay more transparent so satellite terrain shows through
  const alpha = clamped < 0.08 ? clamped * 2.5 : Math.min(0.75, 0.15 + clamped * 0.6)
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

import type { GridMetadata, LatLon } from '../types/geo'

/**
 * Convert between grid cell indices and WGS84 lat/lon using the four actual
 * grid corners the backend ships in `metadata.corners`.
 *
 * Convention (matches the backend grid in app/geospatial/grid.py):
 *   row 0 = north edge,  row rows-1 = south edge
 *   col 0 = west  edge,  col cols-1 = east  edge
 *   (row, col) refers to the cell centroid.
 *
 * Accuracy: bilinear interpolation across the NW-NE-SE-SW quad.  Verified
 * within ~0.5 m of the backend's exact `cell_centroid_latlon` for a 128x128
 * grid at 50 m resolution near a UTM zone edge (worst case).  Do NOT linearly
 * interpolate inside `metadata.bounds` — that's the axis-aligned bbox of the
 * corners and drifts O(100 m) at corner cells.
 */

export function cellLatLon(
  row: number,
  col: number,
  metadata: GridMetadata,
): LatLon {
  const { rows, cols, corners } = metadata
  const u = (col + 0.5) / cols
  const v = (row + 0.5) / rows
  return bilinear(u, v, corners)
}

/** Inverse of `cellLatLon`. Returns null for points outside the grid quad.
 *  Uses bbox-bounds as an initial guess, then refines with one Newton step
 *  on the bilinear forward map (sub-cell accurate for any realistic grid). */
export function latLonToCell(
  lat: number,
  lon: number,
  metadata: GridMetadata,
): { row: number; col: number } | null {
  const { rows, cols, bounds, corners } = metadata

  let u = (lon - bounds.west) / (bounds.east - bounds.west)
  let v = (bounds.north - lat) / (bounds.north - bounds.south)

  for (let iter = 0; iter < 4; iter++) {
    const cur = bilinear(u, v, corners)
    const errLat = lat - cur.lat
    const errLon = lon - cur.lon

    const du = 1e-4
    const dv = 1e-4
    const pu = bilinear(u + du, v, corners)
    const pv = bilinear(u, v + dv, corners)
    const dLatDu = (pu.lat - cur.lat) / du
    const dLonDu = (pu.lon - cur.lon) / du
    const dLatDv = (pv.lat - cur.lat) / dv
    const dLonDv = (pv.lon - cur.lon) / dv

    const det = dLatDu * dLonDv - dLatDv * dLonDu
    if (Math.abs(det) < 1e-20) break
    const ddu = ( dLonDv * errLat - dLatDv * errLon) / det
    const ddv = (-dLonDu * errLat + dLatDu * errLon) / det
    u += ddu
    v += ddv
    if (Math.abs(ddu) < 1e-9 && Math.abs(ddv) < 1e-9) break
  }

  if (u < 0 || u > 1 || v < 0 || v > 1) return null

  const row = Math.min(rows - 1, Math.max(0, Math.floor(v * rows)))
  const col = Math.min(cols - 1, Math.max(0, Math.floor(u * cols)))
  return { row, col }
}

function bilinear(
  u: number,
  v: number,
  corners: GridMetadata['corners'],
): LatLon {
  const topLat = corners.nw.lat * (1 - u) + corners.ne.lat * u
  const topLon = corners.nw.lon * (1 - u) + corners.ne.lon * u
  const botLat = corners.sw.lat * (1 - u) + corners.se.lat * u
  const botLon = corners.sw.lon * (1 - u) + corners.se.lon * u
  return {
    lat: topLat * (1 - v) + botLat * v,
    lon: topLon * (1 - v) + botLon * v,
  }
}

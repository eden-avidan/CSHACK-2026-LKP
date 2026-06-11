export interface LatLon {
  lat: number
  lon: number
}

export interface GridBounds {
  north: number
  south: number
  east: number
  west: number
}

export interface GridCorners {
  nw: LatLon
  ne: LatLon
  se: LatLon
  sw: LatLon
}

export interface GridMetadata {
  origin: LatLon
  resolution_m: number
  rows: number
  cols: number
  crs_epsg: number
  bounds: GridBounds
  corners: GridCorners
}

/** Default map center: Haifa, Israel */
export const HAIFA_CENTER: LatLon = { lat: 32.7940, lon: 34.9896 }
export const HAIFA_MAP_VIEW: { center: [number, number]; zoom: number } = {
  center: [HAIFA_CENTER.lon, HAIFA_CENTER.lat],
  zoom: 13,
}

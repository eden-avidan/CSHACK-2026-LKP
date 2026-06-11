import { useCallback, useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import type { LatLon } from '../../types/geo'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN ?? ''

const LKP_SOURCE = 'lkp-crosshair'
const LKP_LAYER = 'lkp-crosshair-layer'
const DRONE_ROUTE_SOURCE = 'drone-route'
const DRONE_ROUTE_LAYER = 'drone-route-layer'
const DRONE_TRACK_SOURCE = 'drone-track'
const DRONE_TRACK_LAYER = 'drone-track-layer'
const DRONE_TRACK_GLOW_LAYER = 'drone-track-glow-layer'

interface TrackingOverlayProps {
  map: mapboxgl.Map | null
}

function crosshairFeatures(lon: number, lat: number, sizeDeg = 0.00045) {
  return [
    {
      type: 'Feature' as const,
      geometry: {
        type: 'LineString' as const,
        coordinates: [
          [lon - sizeDeg, lat],
          [lon + sizeDeg, lat],
        ],
      },
      properties: {},
    },
    {
      type: 'Feature' as const,
      geometry: {
        type: 'LineString' as const,
        coordinates: [
          [lon, lat - sizeDeg],
          [lon, lat + sizeDeg],
        ],
      },
      properties: {},
    },
  ]
}

function ensureTrackingLayers(map: mapboxgl.Map) {
  if (!map.getSource(LKP_SOURCE)) {
    map.addSource(LKP_SOURCE, {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: LKP_LAYER,
      type: 'line',
      source: LKP_SOURCE,
      paint: { 'line-color': '#ff3333', 'line-width': 4 },
    })
  }

  if (!map.getSource(DRONE_ROUTE_SOURCE)) {
    map.addSource(DRONE_ROUTE_SOURCE, {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: DRONE_ROUTE_LAYER,
      type: 'line',
      source: DRONE_ROUTE_SOURCE,
      paint: {
        'line-color': '#38bdf8',
        'line-width': 5,
        'line-dasharray': [2, 1.5],
        'line-opacity': 0.95,
      },
    })
  }

  // Flown drone track — soft glow underlay + crisp solid line on top.
  if (!map.getSource(DRONE_TRACK_SOURCE)) {
    map.addSource(DRONE_TRACK_SOURCE, {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: DRONE_TRACK_GLOW_LAYER,
      type: 'line',
      source: DRONE_TRACK_SOURCE,
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint: {
        // Amber for the drone that found the person, green for "clean" sweeps.
        'line-color': ['case', ['get', 'found'], '#f59e0b', '#22c55e'],
        'line-width': 9,
        'line-opacity': 0.25,
        'line-blur': 4,
      },
    })
    map.addLayer({
      id: DRONE_TRACK_LAYER,
      type: 'line',
      source: DRONE_TRACK_SOURCE,
      layout: { 'line-cap': 'round', 'line-join': 'round' },
      paint: {
        'line-color': ['case', ['get', 'found'], '#fbbf24', '#4ade80'],
        'line-width': 3,
        'line-opacity': 0.95,
      },
    })
  }

  for (const id of [
    DRONE_ROUTE_LAYER,
    DRONE_TRACK_GLOW_LAYER,
    DRONE_TRACK_LAYER,
    LKP_LAYER,
  ]) {
    if (map.getLayer(id)) {
      map.moveLayer(id)
    }
  }
}

function createDroneElement(found = false): HTMLDivElement {
  const rotor = found ? '#f59e0b' : '#22c55e'
  const hub = found ? '#b45309' : '#16a34a'
  const el = document.createElement('div')
  el.className = `drone-marker${found ? ' drone-marker--found' : ''}`
  el.innerHTML = `
    <div class="drone-pulse"></div>
    <svg class="drone-icon" viewBox="0 0 48 48" width="34" height="34" aria-hidden="true">
      <g fill="none" stroke="#eafff1" stroke-width="2.4" stroke-linecap="round">
        <line x1="24" y1="24" x2="12" y2="12" />
        <line x1="24" y1="24" x2="36" y2="12" />
        <line x1="24" y1="24" x2="12" y2="36" />
        <line x1="24" y1="24" x2="36" y2="36" />
      </g>
      <g fill="${rotor}" stroke="#eafff1" stroke-width="2">
        <circle cx="12" cy="12" r="6" />
        <circle cx="36" cy="12" r="6" />
        <circle cx="12" cy="36" r="6" />
        <circle cx="36" cy="36" r="6" />
      </g>
      <circle cx="24" cy="24" r="5.5" fill="${hub}" stroke="#eafff1" stroke-width="2" />
      <circle cx="24" cy="16.5" r="1.8" fill="#fef08a" />
    </svg>`
  return el
}

function pathHeadingDeg(path: number[][]): number {
  if (path.length < 2) return 0
  const [lon1, lat1] = path[path.length - 2]
  const [lon2, lat2] = path[path.length - 1]
  const dLon = ((lon2 - lon1) * Math.PI) / 180
  const y = Math.sin(dLon) * Math.cos((lat2 * Math.PI) / 180)
  const x =
    Math.cos((lat1 * Math.PI) / 180) * Math.sin((lat2 * Math.PI) / 180) -
    Math.sin((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.cos(dLon)
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360
}

export function TrackingOverlay({ map }: TrackingOverlayProps) {
  const missionId = useMissionStore((s) => s.missionId)
  const lkp = useMissionStore((s) => s.lkp)
  const pinnedLkp = useMissionStore((s) => s.pinnedLkp)
  const draftLkp = useMissionStore((s) => s.draftLkp)
  const gridVersion = useMissionStore((s) => s.gridVersion)
  const droneRoute = useMissionStore((s) => s.droneRoute)
  const dronePosition = useMissionStore((s) => s.dronePosition)
  const dronePath = useMissionStore((s) => s.dronePath)
  const drones = useMissionStore((s) => s.drones)
  const droneMarkersRef = useRef<Map<string, mapboxgl.Marker>>(new Map())
  const [overlayReady, setOverlayReady] = useState(false)

  const displayLkp: LatLon | null = missionId ? lkp : pinnedLkp ?? draftLkp

  const applyTracking = useCallback(() => {
    if (!map || !overlayReady) return

    ensureTrackingLayers(map)

    const lkpSrc = map.getSource(LKP_SOURCE) as mapboxgl.GeoJSONSource
    const droneRouteSrc = map.getSource(DRONE_ROUTE_SOURCE) as mapboxgl.GeoJSONSource
    const droneTrackSrc = map.getSource(DRONE_TRACK_SOURCE) as mapboxgl.GeoJSONSource

    if (displayLkp) {
      lkpSrc.setData({
        type: 'FeatureCollection',
        features: crosshairFeatures(displayLkp.lon, displayLkp.lat),
      })
    } else {
      lkpSrc.setData({ type: 'FeatureCollection', features: [] })
    }

    droneRouteSrc.setData({
      type: 'FeatureCollection',
      features: droneRoute
        ? [
            {
              type: 'Feature',
              geometry: droneRoute,
              properties: { status: 'planned' },
            },
          ]
        : [],
    })

    // Prefer the per-drone list (multiple sorties one after another); fall back
    // to the single active drone for backward compatibility.
    const droneList =
      drones.length > 0
        ? drones
        : dronePosition || dronePath.length >= 2
          ? [{ asset_id: 'drone', found: false, active: true, position: dronePosition, path: dronePath }]
          : []

    droneTrackSrc.setData({
      type: 'FeatureCollection',
      features: droneList
        .filter((d) => d.path.length >= 2)
        .map((d) => ({
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: d.path },
          properties: { status: 'flown', found: !!d.found, asset_id: d.asset_id },
        })),
    })

    // One marker per drone, keyed by asset_id; drop markers no longer present.
    const markers = droneMarkersRef.current
    const seen = new Set<string>()
    for (const d of droneList) {
      if (!d.position) continue
      seen.add(d.asset_id)
      let marker = markers.get(d.asset_id)
      if (!marker) {
        marker = new mapboxgl.Marker({
          element: createDroneElement(!!d.found),
          rotationAlignment: 'map',
          pitchAlignment: 'map',
        })
        markers.set(d.asset_id, marker)
      }
      marker
        .setLngLat([d.position.lon, d.position.lat])
        .setRotation(pathHeadingDeg(d.path))
        .addTo(map)
    }
    for (const [id, marker] of markers) {
      if (!seen.has(id)) {
        marker.remove()
        markers.delete(id)
      }
    }
  }, [map, overlayReady, displayLkp, droneRoute, dronePosition, dronePath, drones])

  useEffect(() => {
    if (!map) {
      setOverlayReady(false)
      return
    }

    const init = () => {
      ensureTrackingLayers(map)
      setOverlayReady(true)
    }

    if (map.isStyleLoaded()) init()
    else map.once('load', init)

    const markers = droneMarkersRef.current
    return () => {
      setOverlayReady(false)
      for (const marker of markers.values()) marker.remove()
      markers.clear()
    }
  }, [map])

  useEffect(() => {
    applyTracking()
  }, [applyTracking, gridVersion, pinnedLkp?.lat, pinnedLkp?.lon, draftLkp?.lat, draftLkp?.lon])

  return null
}

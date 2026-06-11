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
        'line-color': '#22c55e',
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
        'line-color': '#4ade80',
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

function createDroneElement(): HTMLDivElement {
  const el = document.createElement('div')
  el.className = 'drone-marker'
  el.innerHTML = `
    <div class="drone-pulse"></div>
    <svg class="drone-icon" viewBox="0 0 48 48" width="34" height="34" aria-hidden="true">
      <g fill="none" stroke="#eafff1" stroke-width="2.4" stroke-linecap="round">
        <line x1="24" y1="24" x2="12" y2="12" />
        <line x1="24" y1="24" x2="36" y2="12" />
        <line x1="24" y1="24" x2="12" y2="36" />
        <line x1="24" y1="24" x2="36" y2="36" />
      </g>
      <g fill="#22c55e" stroke="#eafff1" stroke-width="2">
        <circle cx="12" cy="12" r="6" />
        <circle cx="36" cy="12" r="6" />
        <circle cx="12" cy="36" r="6" />
        <circle cx="36" cy="36" r="6" />
      </g>
      <circle cx="24" cy="24" r="5.5" fill="#16a34a" stroke="#eafff1" stroke-width="2" />
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
  const droneMarkerRef = useRef<mapboxgl.Marker | null>(null)
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

    droneTrackSrc.setData({
      type: 'FeatureCollection',
      features:
        dronePath.length >= 2
          ? [
              {
                type: 'Feature',
                geometry: { type: 'LineString', coordinates: dronePath },
                properties: { status: 'flown' },
              },
            ]
          : [],
    })

    if (dronePosition) {
      if (!droneMarkerRef.current) {
        droneMarkerRef.current = new mapboxgl.Marker({
          element: createDroneElement(),
          rotationAlignment: 'map',
          pitchAlignment: 'map',
        })
      }
      droneMarkerRef.current
        .setLngLat([dronePosition.lon, dronePosition.lat])
        .setRotation(pathHeadingDeg(dronePath))
        .addTo(map)
    } else if (droneMarkerRef.current) {
      droneMarkerRef.current.remove()
    }
  }, [map, overlayReady, displayLkp, droneRoute, dronePosition, dronePath])

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

    return () => {
      setOverlayReady(false)
      droneMarkerRef.current?.remove()
      droneMarkerRef.current = null
    }
  }, [map])

  useEffect(() => {
    applyTracking()
  }, [applyTracking, gridVersion, pinnedLkp?.lat, pinnedLkp?.lon, draftLkp?.lat, draftLkp?.lon])

  return null
}

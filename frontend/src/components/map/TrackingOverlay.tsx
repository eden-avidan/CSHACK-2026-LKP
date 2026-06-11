import { useCallback, useEffect, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import type { LatLon } from '../../types/geo'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN ?? ''

const LKP_SOURCE = 'lkp-crosshair'
const LKP_LAYER = 'lkp-crosshair-layer'
const DRONE_ROUTE_SOURCE = 'drone-route'
const DRONE_ROUTE_LAYER = 'drone-route-layer'

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

  for (const id of [DRONE_ROUTE_LAYER, LKP_LAYER]) {
    if (map.getLayer(id)) {
      map.moveLayer(id)
    }
  }
}

export function TrackingOverlay({ map }: TrackingOverlayProps) {
  const missionId = useMissionStore((s) => s.missionId)
  const lkp = useMissionStore((s) => s.lkp)
  const pinnedLkp = useMissionStore((s) => s.pinnedLkp)
  const draftLkp = useMissionStore((s) => s.draftLkp)
  const gridVersion = useMissionStore((s) => s.gridVersion)
  const droneRoute = useMissionStore((s) => s.droneRoute)
  const [overlayReady, setOverlayReady] = useState(false)

  const displayLkp: LatLon | null = missionId ? lkp : pinnedLkp ?? draftLkp

  const applyTracking = useCallback(() => {
    if (!map || !overlayReady) return

    ensureTrackingLayers(map)

    const lkpSrc = map.getSource(LKP_SOURCE) as mapboxgl.GeoJSONSource
    const droneRouteSrc = map.getSource(DRONE_ROUTE_SOURCE) as mapboxgl.GeoJSONSource

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
  }, [map, overlayReady, displayLkp, droneRoute])

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

    return () => setOverlayReady(false)
  }, [map])

  useEffect(() => {
    applyTracking()
  }, [applyTracking, gridVersion, pinnedLkp?.lat, pinnedLkp?.lon, draftLkp?.lat, draftLkp?.lon])

  return null
}

import { useCallback, useEffect, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import type { LatLon } from '../../types/geo'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN ?? ''

const LKP_SOURCE = 'lkp-crosshair'
const LKP_LAYER = 'lkp-crosshair-layer'
const MPP_SOURCE = 'mpp-marker'
const MPP_LAYER = 'mpp-marker-layer'
const DRIFT_SOURCE = 'drift-line'
const DRIFT_LAYER = 'drift-line-layer'

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

  if (!map.getSource(MPP_SOURCE)) {
    map.addSource(MPP_SOURCE, {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: MPP_LAYER,
      type: 'circle',
      source: MPP_SOURCE,
      paint: {
        'circle-radius': 10,
        'circle-color': '#22c55e',
        'circle-opacity': 0.9,
        'circle-stroke-width': 2,
        'circle-stroke-color': '#ffffff',
      },
    })
  }

  if (!map.getSource(DRIFT_SOURCE)) {
    map.addSource(DRIFT_SOURCE, {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: DRIFT_LAYER,
      type: 'line',
      source: DRIFT_SOURCE,
      paint: {
        'line-color': '#86efac',
        'line-width': 4,
        'line-dasharray': [2, 2],
        'line-opacity': 0.95,
      },
    })
  }

  for (const id of [DRIFT_LAYER, MPP_LAYER, LKP_LAYER]) {
    if (map.getLayer(id)) {
      map.moveLayer(id)
    }
  }
}

export function TrackingOverlay({ map }: TrackingOverlayProps) {
  const missionId = useMissionStore((s) => s.missionId)
  const lkp = useMissionStore((s) => s.lkp)
  const pendingLkp = useMissionStore((s) => s.pendingLkp)
  const mpp = useMissionStore((s) => s.mpp)
  const mppTrail = useMissionStore((s) => s.mppTrail)
  const engineTickVersion = useMissionStore((s) => s.engineTickVersion)
  const gridVersion = useMissionStore((s) => s.gridVersion)
  const [overlayReady, setOverlayReady] = useState(false)
  const [mppScreen, setMppScreen] = useState<{ x: number; y: number } | null>(null)

  const displayLkp: LatLon | null = missionId ? lkp : pendingLkp

  const applyTracking = useCallback(() => {
    if (!map || !overlayReady) return

    ensureTrackingLayers(map)

    const lkpSrc = map.getSource(LKP_SOURCE) as mapboxgl.GeoJSONSource
    const mppSrc = map.getSource(MPP_SOURCE) as mapboxgl.GeoJSONSource
    const driftSrc = map.getSource(DRIFT_SOURCE) as mapboxgl.GeoJSONSource

    if (displayLkp) {
      lkpSrc.setData({
        type: 'FeatureCollection',
        features: crosshairFeatures(displayLkp.lon, displayLkp.lat),
      })
    } else {
      lkpSrc.setData({ type: 'FeatureCollection', features: [] })
    }

    if (missionId && displayLkp && mpp) {
      mppSrc.setData({
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            geometry: { type: 'Point', coordinates: [mpp.lon, mpp.lat] },
            properties: {},
          },
        ],
      })
      const path =
        mppTrail.length >= 2
          ? mppTrail
          : [displayLkp, mpp]
      driftSrc.setData({
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            geometry: {
              type: 'LineString',
              coordinates: path.map((p) => [p.lon, p.lat]),
            },
            properties: {},
          },
        ],
      })
    } else {
      mppSrc.setData({ type: 'FeatureCollection', features: [] })
      driftSrc.setData({ type: 'FeatureCollection', features: [] })
    }
  }, [map, overlayReady, missionId, displayLkp, mpp, mppTrail])

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
  }, [applyTracking, engineTickVersion, gridVersion, pendingLkp?.lat, pendingLkp?.lon])

  useEffect(() => {
    if (!map || !missionId || !mpp) {
      setMppScreen(null)
      return
    }

    const update = () => {
      const p = map.project([mpp.lon, mpp.lat])
      setMppScreen({ x: p.x, y: p.y })
    }

    update()
    map.on('move', update)
    map.on('zoom', update)
    map.on('resize', update)
    return () => {
      map.off('move', update)
      map.off('zoom', update)
      map.off('resize', update)
    }
  }, [map, missionId, mpp, engineTickVersion])

  return mppScreen ? (
    <div
      className="mpp-pulse-ring"
      style={{ left: mppScreen.x, top: mppScreen.y }}
      aria-hidden
    />
  ) : null
}

import { useCallback, useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import type { LatLon } from '../../types/geo'
import type { DroneTrackItem } from '../../types/ws-messages'
import {
  buildTrailDraws,
  drawDroneTrails,
  markerOpacityForTrail,
  mountTrailCanvas,
  normalizeDronePath,
  SECOND_DRONE_ASSET_ID,
  type TrailSegment,
} from '../../utils/droneTrailCanvas'

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

function removeLegacyTrackLayers(map: mapboxgl.Map) {
  for (const id of [
    'drone-track-layer',
    'drone-track-glow-layer',
    'drone-track-found-layer',
    'drone-track-found-glow-layer',
  ]) {
    if (map.getLayer(id)) map.removeLayer(id)
  }
  if (map.getSource('drone-track')) map.removeSource('drone-track')
}

function ensureTrackingLayers(map: mapboxgl.Map) {
  removeLegacyTrackLayers(map)

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
    if (map.getLayer(id)) map.moveLayer(id)
  }
}

function dronePalette(assetId: string, found: boolean) {
  if (assetId === SECOND_DRONE_ASSET_ID) {
    return { rotor: '#38bdf8', hub: '#0284c7', tone: 'blue' as const }
  }
  if (found) {
    return { rotor: '#f59e0b', hub: '#b45309', tone: 'found' as const }
  }
  return { rotor: '#22c55e', hub: '#16a34a', tone: 'clean' as const }
}

function createDroneElement(assetId: string, found = false): HTMLDivElement {
  const { rotor, hub, tone } = dronePalette(assetId, found)
  const el = document.createElement('div')
  el.className = `drone-marker${tone === 'found' ? ' drone-marker--found' : ''}${tone === 'blue' ? ' drone-marker--blue' : ''}`
  el.dataset.droneTone = tone
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

function syncDroneElement(el: HTMLElement, assetId: string, found: boolean): void {
  const { tone } = dronePalette(assetId, found)
  if (el.dataset.droneTone === tone) return
  const fresh = createDroneElement(assetId, found)
  el.className = fresh.className
  el.dataset.droneTone = tone
  el.innerHTML = fresh.innerHTML
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

function resolveDroneList(
  drones: DroneTrackItem[],
  dronePosition: LatLon | null,
  dronePath: number[][],
): Array<{
  asset_id: string
  found: boolean
  active: boolean
  position?: LatLon | null
  path: number[][]
}> {
  if (drones.length > 0) {
    return drones.map((d) => ({
      asset_id: d.asset_id,
      found: !!d.found,
      active: d.active !== false,
      position: d.position,
      path: d.path,
    }))
  }
  if (dronePosition || dronePath.length >= 2) {
    return [
      {
        asset_id: 'drone',
        found: false,
        active: true,
        position: dronePosition,
        path: dronePath,
      },
    ]
  }
  return []
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
  const tickCount = useMissionStore((s) => s.tickCount)
  const stepSec = useMissionStore((s) => s.stepSec)
  const simulationRunning = useMissionStore((s) => s.simulationRunning)
  const engineTickVersion = useMissionStore((s) => s.engineTickVersion)
  const droneMarkersRef = useRef<Map<string, mapboxgl.Marker>>(new Map())
  const segmentCacheRef = useRef<Map<string, TrailSegment[]>>(new Map())
  const trailCanvasRef = useRef<ReturnType<typeof mountTrailCanvas> | null>(null)
  const frozenNowMsRef = useRef<number | null>(null)
  const rafRef = useRef(0)
  const [overlayReady, setOverlayReady] = useState(false)

  useEffect(() => {
    if (!simulationRunning && frozenNowMsRef.current === null) {
      frozenNowMsRef.current = performance.now()
    }
    if (simulationRunning) {
      frozenNowMsRef.current = null
    }
  }, [simulationRunning])

  const trailNowMs = useCallback(
    () => frozenNowMsRef.current ?? performance.now(),
    [],
  )

  useEffect(() => {
    segmentCacheRef.current.clear()
  }, [missionId])

  const paintTrails = useCallback(() => {
    const trail = trailCanvasRef.current
    if (!map || !trail) return

    const state = useMissionStore.getState()
    const droneList = resolveDroneList(
      state.drones,
      state.dronePosition,
      state.dronePath,
    )
    const nowMs = trailNowMs()

    const draws = buildTrailDraws(
      segmentCacheRef.current,
      droneList,
      nowMs,
      state.stepSec,
    )
    drawDroneTrails(map, trail.ctx, draws, nowMs)

    const markers = droneMarkersRef.current
    for (const d of droneList) {
      if (!d.position) continue
      const marker = markers.get(d.asset_id)
      if (!marker) continue
      const draw = draws.find((t) => t.assetId === d.asset_id)
      const opacity = markerOpacityForTrail(draw, nowMs)
      const el = marker.getElement()
      el.style.opacity = opacity <= 0.02 ? '0' : String(opacity)
      el.style.visibility = opacity <= 0.02 ? 'hidden' : 'visible'
    }
  }, [map, trailNowMs])

  const applyTracking = useCallback(() => {
    if (!map || !overlayReady) return

    ensureTrackingLayers(map)

    const lkpSrc = map.getSource(LKP_SOURCE) as mapboxgl.GeoJSONSource
    const droneRouteSrc = map.getSource(DRONE_ROUTE_SOURCE) as mapboxgl.GeoJSONSource

    const displayLkp: LatLon | null = missionId ? lkp : pinnedLkp ?? draftLkp

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

    const droneList = resolveDroneList(drones, dronePosition, dronePath)

    const markers = droneMarkersRef.current
    const seen = new Set<string>()
    for (const d of droneList) {
      const normalized = normalizeDronePath(d)
      if (!normalized?.anchor) continue
      const { anchor, path } = normalized
      seen.add(d.asset_id)
      let marker = markers.get(d.asset_id)
      if (!marker) {
        marker = new mapboxgl.Marker({
          element: createDroneElement(d.asset_id, d.found),
          rotationAlignment: 'map',
          pitchAlignment: 'map',
        })
        markers.set(d.asset_id, marker)
      } else {
        syncDroneElement(marker.getElement(), d.asset_id, d.found)
      }
      marker
        .setLngLat([anchor.lon, anchor.lat])
        .setRotation(pathHeadingDeg(path))
        .addTo(map)
      if (d.active) {
        marker.getElement().style.opacity = '1'
        marker.getElement().style.visibility = 'visible'
      }
    }
    for (const [id, marker] of markers) {
      if (!seen.has(id)) {
        marker.remove()
        markers.delete(id)
      }
    }

    paintTrails()
  }, [
    map,
    overlayReady,
    missionId,
    lkp,
    pinnedLkp,
    draftLkp,
    droneRoute,
    dronePosition,
    dronePath,
    drones,
    paintTrails,
  ])

  useEffect(() => {
    if (!map) {
      setOverlayReady(false)
      return
    }

    const init = () => {
      ensureTrackingLayers(map)
      trailCanvasRef.current = mountTrailCanvas(map)
      setOverlayReady(true)
    }

    if (map.isStyleLoaded()) init()
    else map.once('load', init)

    const onResize = () => {
      trailCanvasRef.current?.resize()
      paintTrails()
    }
    const onMove = () => paintTrails()

    map.on('resize', onResize)
    map.on('move', onMove)

    const markers = droneMarkersRef.current
    return () => {
      map.off('resize', onResize)
      map.off('move', onMove)
      trailCanvasRef.current?.remove()
      trailCanvasRef.current = null
      setOverlayReady(false)
      for (const marker of markers.values()) marker.remove()
      markers.clear()
    }
  }, [map, paintTrails])

  useEffect(() => {
    applyTracking()
  }, [
    applyTracking,
    gridVersion,
    engineTickVersion,
    tickCount,
    stepSec,
    drones,
    dronePath,
    pinnedLkp?.lat,
    pinnedLkp?.lon,
    draftLkp?.lat,
    draftLkp?.lon,
  ])

  useEffect(() => {
    if (!map || !overlayReady) return

    const loop = () => {
      paintTrails()
      rafRef.current = requestAnimationFrame(loop)
    }

    rafRef.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(rafRef.current)
  }, [map, overlayReady, paintTrails])

  return null
}

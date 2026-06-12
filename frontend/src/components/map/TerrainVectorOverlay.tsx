import { useEffect } from 'react'
import type { GeoJSONSource, Map } from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import { buildVectorFieldGeoJson } from '../../utils/vectorFields'

const SOURCE_ID = 'terrain-vector-source'
const SHAFT_LAYER = 'terrain-vector-shaft'
const HEAD_LAYER = 'terrain-vector-head'
const LKP_LAYER = 'terrain-vector-lkp'

const VECTOR_FIELDS = new Set(['wind_vectors', 'current_vectors'])

function removeVectorLayers(map: Map) {
  for (const id of [LKP_LAYER, HEAD_LAYER, SHAFT_LAYER]) {
    if (map.getLayer(id)) map.removeLayer(id)
  }
  if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID)
}

function ensureLayers(map: Map) {
  if (!map.getSource(SOURCE_ID)) {
    map.addSource(SOURCE_ID, {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    })
    map.addLayer({
      id: SHAFT_LAYER,
      type: 'line',
      source: SOURCE_ID,
      filter: ['==', ['get', 'kind'], 'shaft'],
      paint: {
        'line-color': ['get', 'color'],
        'line-width': ['interpolate', ['linear'], ['get', 'speed'], 0, 1.5, 2, 3.5],
        'line-opacity': 0.85,
      },
    })
    map.addLayer({
      id: HEAD_LAYER,
      type: 'line',
      source: SOURCE_ID,
      filter: ['==', ['get', 'kind'], 'head'],
      paint: {
        'line-color': ['get', 'color'],
        'line-width': 2.5,
        'line-opacity': 0.9,
      },
    })
    map.addLayer({
      id: LKP_LAYER,
      type: 'line',
      source: SOURCE_ID,
      filter: ['==', ['get', 'kind'], 'lkp'],
      paint: {
        'line-color': ['get', 'color'],
        'line-width': 5,
        'line-opacity': 1,
      },
    })
  }
  for (const id of [SHAFT_LAYER, HEAD_LAYER, LKP_LAYER]) {
    if (map.getLayer(id)) map.moveLayer(id)
  }
}

interface TerrainVectorOverlayProps {
  map: Map | null
}

export function TerrainVectorOverlay({ map }: TerrainVectorOverlayProps) {
  const terrainData = useMissionStore((s) => s.terrainData)
  const terrainField = useMissionStore((s) => s.terrainField)
  const terrainVersion = useMissionStore((s) => s.terrainVersion)

  useEffect(() => {
    if (!map) return

    const active =
      terrainData && terrainField && VECTOR_FIELDS.has(terrainField)
    if (!active) {
      removeVectorLayers(map)
      return
    }

    const sync = () => {
      ensureLayers(map)
      const geojson = buildVectorFieldGeoJson(terrainData, terrainField)
      const src = map.getSource(SOURCE_ID) as GeoJSONSource | undefined
      src?.setData(geojson)
      map.triggerRepaint()
    }

    if (map.isStyleLoaded()) sync()
    else map.once('load', sync)
  }, [map, terrainData, terrainField, terrainVersion])

  useEffect(() => {
    return () => {
      if (map) removeVectorLayers(map)
    }
  }, [map])

  return null
}

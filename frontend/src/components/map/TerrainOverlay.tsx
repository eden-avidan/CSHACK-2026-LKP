import { useEffect, useRef } from 'react'
import type { Map, ImageSource } from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import { paintField } from '../../utils/fieldScale'

const SOURCE_ID = 'terrain-field-source'
const LAYER_ID = 'terrain-field-layer'

interface TerrainOverlayProps {
  map: Map | null
}

function removeTerrainLayer(map: Map) {
  if (map.getLayer(LAYER_ID)) map.removeLayer(LAYER_ID)
  if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID)
}

export function TerrainOverlay({ map }: TerrainOverlayProps) {
  const terrainData = useMissionStore((s) => s.terrainData)
  const terrainField = useMissionStore((s) => s.terrainField)
  const terrainMaskMode = useMissionStore((s) => s.terrainMaskMode)
  const terrainMaskThreshold = useMissionStore((s) => s.terrainMaskThreshold)
  const terrainVersion = useMissionStore((s) => s.terrainVersion)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const lastBoundsRef = useRef<string | null>(null)

  useEffect(() => {
    if (!map) return

    const active =
      terrainData &&
      terrainField &&
      terrainData.fields[terrainField] &&
      terrainData.available.find((f) => f.id === terrainField)?.kind !== 'vector'
    if (!active) {
      if (map.getLayer(LAYER_ID) || map.getSource(SOURCE_ID)) removeTerrainLayer(map)
      lastBoundsRef.current = null
      return
    }

    const values = terrainData.fields[terrainField]
    const meta = terrainData.available.find((f) => f.id === terrainField)
    const kind = meta?.kind ?? 'scalar'
    const forceMask = terrainMaskMode && kind === 'scalar'

    const syncLayer = () => {
      if (!canvasRef.current) canvasRef.current = document.createElement('canvas')
      const canvas = canvasRef.current
      paintField(values, terrainData.rows, terrainData.cols, kind, terrainField, canvas, {
        forceMask,
        maskThreshold: terrainMaskThreshold,
      })

      const b = terrainData.metadata.bounds
      const coords: [[number, number], [number, number], [number, number], [number, number]] = [
        [b.west, b.north],
        [b.east, b.north],
        [b.east, b.south],
        [b.west, b.south],
      ]
      const boundsKey = `${b.west},${b.south},${b.east},${b.north}`
      const dataUrl = canvas.toDataURL()
      const crisp = kind === 'mask' || forceMask

      if (!map.getSource(SOURCE_ID)) {
        map.addSource(SOURCE_ID, { type: 'image', url: dataUrl, coordinates: coords })
        map.addLayer({
          id: LAYER_ID,
          type: 'raster',
          source: SOURCE_ID,
          paint: {
            'raster-opacity': 0.8,
            'raster-fade-duration': 0,
            'raster-resampling': crisp ? 'nearest' : 'linear',
          },
        })
        lastBoundsRef.current = boundsKey
      } else {
        const src = map.getSource(SOURCE_ID) as ImageSource
        src.updateImage({ url: dataUrl })
        if (lastBoundsRef.current !== boundsKey) {
          src.setCoordinates(coords)
          lastBoundsRef.current = boundsKey
        }
        if (map.getLayer(LAYER_ID)) {
          map.setPaintProperty(LAYER_ID, 'raster-resampling', crisp ? 'nearest' : 'linear')
        }
      }
      map.triggerRepaint()
    }

    if (map.isStyleLoaded()) {
      syncLayer()
    } else {
      map.once('load', syncLayer)
    }
  }, [map, terrainData, terrainField, terrainMaskMode, terrainMaskThreshold, terrainVersion])

  useEffect(() => {
    return () => {
      if (map) removeTerrainLayer(map)
    }
  }, [map])

  return null
}

import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import { useHeatmapPainter } from '../../hooks/useHeatmap'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN ?? ''

const SOURCE_ID = 'heatmap-source'
const LAYER_ID = 'heatmap-layer'

interface HeatmapCanvasProps {
  map: mapboxgl.Map | null
}

function removeHeatmapLayer(map: mapboxgl.Map) {
  if (map.getLayer(LAYER_ID)) map.removeLayer(LAYER_ID)
  if (map.getSource(SOURCE_ID)) map.removeSource(SOURCE_ID)
}

export function HeatmapCanvas({ map }: HeatmapCanvasProps) {
  const missionId = useMissionStore((s) => s.missionId)
  const metadata = useMissionStore((s) => s.metadata)
  const grid = useMissionStore((s) => s.grid)
  const gridVersion = useMissionStore((s) => s.gridVersion)
  const { paintSync } = useHeatmapPainter()
  const lastBoundsRef = useRef<string | null>(null)

  useEffect(() => {
    if (!map) return
    if (missionId === null && metadata === null && grid === null) {
      removeHeatmapLayer(map)
      lastBoundsRef.current = null
    }
  }, [map, missionId, metadata, grid])

  useEffect(() => {
    if (!map || !metadata || !grid) return

    const syncLayer = () => {
      const canvas = paintSync(grid, metadata.rows, metadata.cols)
      if (!canvas) return

      const { corners } = metadata
      const coords: [[number, number], [number, number], [number, number], [number, number]] = [
        [corners.nw.lon, corners.nw.lat],
        [corners.ne.lon, corners.ne.lat],
        [corners.se.lon, corners.se.lat],
        [corners.sw.lon, corners.sw.lat],
      ]

      const boundsKey = [
        corners.nw.lat,
        corners.nw.lon,
        corners.ne.lat,
        corners.ne.lon,
        corners.se.lat,
        corners.se.lon,
        corners.sw.lat,
        corners.sw.lon,
      ].join(',')
      const dataUrl = canvas.toDataURL()

      if (!map.getSource(SOURCE_ID)) {
        map.addSource(SOURCE_ID, {
          type: 'image',
          url: dataUrl,
          coordinates: coords,
        })
        map.addLayer(
          {
            id: LAYER_ID,
            type: 'raster',
            source: SOURCE_ID,
            paint: { 'raster-opacity': 0.90, 'raster-fade-duration': 0 },
          },
          undefined,
        )
        lastBoundsRef.current = boundsKey
      } else {
        const src = map.getSource(SOURCE_ID) as mapboxgl.ImageSource
        src.updateImage({ url: dataUrl })
        if (lastBoundsRef.current !== boundsKey) {
          src.setCoordinates(coords)
          lastBoundsRef.current = boundsKey
        }
      }

      map.triggerRepaint()
    }

    if (map.isStyleLoaded()) {
      syncLayer()
    } else {
      map.once('load', syncLayer)
    }
  }, [map, metadata, grid, gridVersion, paintSync])

  useEffect(() => {
    return () => {
      if (map) removeHeatmapLayer(map)
    }
  }, [map])

  return null
}

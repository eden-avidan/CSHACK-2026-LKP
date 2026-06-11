import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'
import { HAIFA_MAP_VIEW } from '../../types/geo'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN ?? ''

interface MapContainerProps {
  onMapReady: (map: mapboxgl.Map) => void
}

export function MapContainer({ onMapReady }: MapContainerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<mapboxgl.Map | null>(null)
  const setDraftLkp = useMissionStore((s) => s.setDraftLkp)

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return

    const map = new mapboxgl.Map({
      container: containerRef.current,
      style: 'mapbox://styles/mapbox/satellite-streets-v12',
      center: HAIFA_MAP_VIEW.center,
      zoom: HAIFA_MAP_VIEW.zoom,
    })

    map.addControl(new mapboxgl.NavigationControl(), 'top-right')

    map.on('load', () => {
      mapRef.current = map
      onMapReady(map)
    })

    map.on('click', (e) => {
      setDraftLkp({ lat: e.lngLat.lat, lon: e.lngLat.lng })
    })

    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [onMapReady, setDraftLkp])

  return <div ref={containerRef} className="map-container" />
}

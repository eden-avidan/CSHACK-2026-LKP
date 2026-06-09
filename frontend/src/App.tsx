import { useCallback, useState } from 'react'
import type { Map } from 'mapbox-gl'
import { MapContainer } from './components/map/MapContainer'
import { HeatmapCanvas } from './components/map/HeatmapCanvas'
import { TrackingOverlay } from './components/map/TrackingOverlay'
import { MissionControl } from './components/panels/MissionControl'
import { LayerControls } from './components/panels/LayerControls'
import { HeatmapLegend } from './components/panels/HeatmapLegend'

export default function App() {
  const [map, setMap] = useState<Map | null>(null)
  const onMapReady = useCallback((m: Map) => setMap(m), [])

  return (
    <div className="app">
      <header className="header">
        <h1>RescuEdge</h1>
        <span className="subtitle">SAR Command Center</span>
      </header>
      <div className="layout">
        <aside className="sidebar left">
          <MissionControl map={map} />
          <LayerControls />
        </aside>
        <main className="map-area">
          <MapContainer onMapReady={onMapReady} />
          <HeatmapCanvas map={map} />
          <TrackingOverlay map={map} />
        </main>
      </div>
      <footer className="footer">
        <HeatmapLegend />
      </footer>
    </div>
  )
}

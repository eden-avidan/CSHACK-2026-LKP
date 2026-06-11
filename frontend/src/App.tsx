import { useCallback, useState } from 'react'
import type { Map } from 'mapbox-gl'
import { MapContainer } from './components/map/MapContainer'
import { HeatmapCanvas } from './components/map/HeatmapCanvas'
import { TerrainOverlay } from './components/map/TerrainOverlay'
import { CurrentVectorOverlay } from './components/map/CurrentVectorOverlay'
import { TrackingOverlay } from './components/map/TrackingOverlay'
import { HeatmapSidebar } from './components/panels/HeatmapSidebar'
import { LayerControls } from './components/panels/LayerControls'
import { TerrainInspector } from './components/panels/TerrainInspector'
import { HeatmapLegend } from './components/panels/HeatmapLegend'
import { DetectionFlash } from './components/map/DetectionFlash'

export default function App() {
  const [map, setMap] = useState<Map | null>(null)
  const onMapReady = useCallback((m: Map) => setMap(m), [])

  return (
    <div className="app">
      <header className="header">
        <h1>Adar</h1>
        <span className="subtitle">SAR Command Center</span>
      </header>
      <div className="layout">
        <aside className="sidebar left">
          <HeatmapSidebar map={map} />
          <LayerControls />
          <TerrainInspector />
        </aside>
        <main className="map-area">
          <MapContainer onMapReady={onMapReady} />
          <TerrainOverlay map={map} />
          <CurrentVectorOverlay map={map} />
          <HeatmapCanvas map={map} />
          <TrackingOverlay map={map} />
          <DetectionFlash map={map} />
        </main>
      </div>
      <footer className="footer">
        <HeatmapLegend />
      </footer>
    </div>
  )
}

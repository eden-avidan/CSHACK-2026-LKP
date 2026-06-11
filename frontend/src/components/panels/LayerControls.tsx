import { useCallback } from 'react'
import { useMissionStore, type LayerState } from '../../stores/missionStore'

const LAYER_CONFIG: {
  key: keyof LayerState
  label: string
  hint: string
}[] = [
  {
    key: 'topography',
    label: 'Topography / Reachability',
    hint: 'Tobler hiking + Dijkstra isochrones from LKP; steep/cliff/valley/ridge class weights; blocks water cells.',
  },
  {
    key: 'roads',
    label: 'Trail Magnetism',
    hint: 'Within 80 m of OSM roads: strong velocity snap + displacement pull along tangent; KDE boost up to +120%.',
  },
  {
    key: 'subject_injured',
    label: 'Subject Injured',
    hint: 'Velocity ×0.25 and random walk variance halved — simulates sheltering / limited mobility.',
  },
  {
    key: 'weather',
    label: 'Weather / Wind',
    hint: 'Adds mock wind 4 m/s north + 2.5 m/s east to all particles (off by default for coastal missions).',
  },
]

export function LayerControls() {
  const missionId = useMissionStore((s) => s.missionId)
  const layers = useMissionStore((s) => s.layers)
  const wsStatus = useMissionStore((s) => s.wsStatus)
  const setLayers = useMissionStore((s) => s.setLayers)
  const wsSend = useMissionStore((s) => s.wsSend)

  const sendLayers = useCallback(
    (next: LayerState) => {
      if (wsSend && wsStatus === 'open') {
        wsSend({ event: 'update_layers', layers: next })
      }
    },
    [wsSend, wsStatus],
  )

  const toggleLayer = useCallback(
    (key: keyof LayerState) => {
      const next = { ...layers, [key]: !layers[key] }
      if (!Object.values(next).some(Boolean)) {
        next.topography = true
      }
      setLayers(next)
      sendLayers(next)
    },
    [layers, setLayers, sendLayers],
  )

  const togglesDisabled = false

  return (
    <div className="layer-controls-sidebar" aria-label="Simulation layer toggles">
      <h3>Probability Layers</h3>
      {!missionId && (
        <p className="layer-idle-hint">Set layers before creating a mission — they apply on start.</p>
      )}
      <ul className="layer-toggle-list">
        {LAYER_CONFIG.map(({ key, label, hint }) => (
          <li key={key}>
            <label className="layer-toggle">
              <input
                type="checkbox"
                checked={layers[key]}
                onChange={() => toggleLayer(key)}
                disabled={togglesDisabled}
              />
              <span>{label}</span>
            </label>
            <p className="layer-hint">{hint}</p>
          </li>
        ))}
      </ul>
      {missionId && wsStatus !== 'open' && (
        <p className="layer-ws-hint">Connecting WebSocket…</p>
      )}
    </div>
  )
}

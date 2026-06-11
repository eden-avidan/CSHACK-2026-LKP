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
    key: 'personality',
    label: 'Personality',
    hint: 'Age, fitness, and injury adjust how far probability spreads from the LKP (mobility heuristic).',
  },
  {
    key: 'weather',
    label: 'Weather / Wind',
    hint: 'Adds mock wind 4 m/s north + 2.5 m/s east to all particles (off by default for coastal missions).',
  },
  {
    key: 'sea_drift',
    label: 'Sea Drift / Current',
    hint: 'Advects probability across water cells along the live Open-Meteo marine current (auto-enabled when the LKP is offshore).',
  },
]

export function LayerControls() {
  const missionId = useMissionStore((s) => s.missionId)
  const layers = useMissionStore((s) => s.layers)
  const personality = useMissionStore((s) => s.personality)
  const setLayers = useMissionStore((s) => s.setLayers)
  const setPersonality = useMissionStore((s) => s.setPersonality)

  const toggleLayer = useCallback(
    (key: keyof LayerState) => {
      const next = { ...layers, [key]: !layers[key] }
      if (!Object.values(next).some(Boolean)) {
        next.topography = true
      }
      setLayers(next)
    },
    [layers, setLayers],
  )

  const togglesDisabled = Boolean(missionId)

  return (
    <div className="layer-controls-sidebar" aria-label="Simulation layer toggles">
      <h3>Probability Layers</h3>
      {missionId ? (
        <p className="layer-idle-hint">
          Locked while this mission runs — the grid is built iteratively from the layers chosen at
          start. Use New Pin to change them.
        </p>
      ) : (
        <p className="layer-idle-hint">Set layers before Run Heatmap — they apply for the whole simulation.</p>
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
            {key === 'personality' && layers.personality && !missionId && (
              <div className="personality-fields">
                <label className="field">
                  <span>Age (years)</span>
                  <input
                    type="number"
                    min={1}
                    max={120}
                    value={personality.age}
                    onChange={(e) =>
                      setPersonality({ age: Math.max(1, Math.min(120, Number(e.target.value) || 1)) })
                    }
                  />
                </label>
                <label className="field">
                  <span>Fitness — {personality.fitness}/5</span>
                  <input
                    type="range"
                    min={1}
                    max={5}
                    step={1}
                    value={personality.fitness}
                    onChange={(e) => setPersonality({ fitness: Number(e.target.value) })}
                  />
                </label>
                <label className="layer-toggle">
                  <input
                    type="checkbox"
                    checked={personality.injured}
                    onChange={(e) => setPersonality({ injured: e.target.checked })}
                  />
                  <span>Injured</span>
                </label>
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

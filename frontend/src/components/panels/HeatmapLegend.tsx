import { useMissionStore } from '../../stores/missionStore'
import { gridMax, isLowMassGrid } from '../../utils/colorScale'

export function HeatmapLegend() {
  const grid = useMissionStore((s) => s.grid)

  let peakLabel = '—'
  let lowMass = false
  if (grid) {
    const max = gridMax(grid)
    peakLabel = max > 1e-6 ? `${(max * 100).toFixed(2)}% peak` : 'no data'
    lowMass = isLowMassGrid(grid)
  }

  return (
    <div className="heatmap-legend" aria-label="Probability heatmap legend">
      <span className="legend-label">Low prob</span>
      <div className="legend-bar" />
      <span className="legend-label">High prob</span>
      <span className="legend-peak">{peakLabel}</span>
      {lowMass && (
        <span className="legend-warn">Low mass — check boundary</span>
      )}
      <span className="legend-note">Roads & slope bias active when terrain loads</span>
    </div>
  )
}

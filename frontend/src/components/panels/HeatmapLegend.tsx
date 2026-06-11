import { useMissionStore } from '../../stores/missionStore'
import { gridMax, gridMin, isLowMassGrid } from '../../utils/colorScale'

export function HeatmapLegend() {
  const grid = useMissionStore((s) => s.grid)

  let peakLabel = '—'
  let lowMass = false
  if (grid) {
    const min = gridMin(grid)
    const max = gridMax(grid)
    const span = max - min
    peakLabel =
      span > 1e-6
        ? `display 0–1 (raw ${min.toExponential(1)}…${max.toExponential(1)})`
        : 'no data'
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

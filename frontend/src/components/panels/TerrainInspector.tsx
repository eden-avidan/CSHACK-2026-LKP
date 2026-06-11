import { useCallback, useMemo, useState } from 'react'
import { useMissionStore } from '../../stores/missionStore'
import type { TerrainData } from '../../stores/missionStore'
import { computeFieldRange } from '../../utils/fieldScale'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? 'http://localhost:8000'

export function TerrainInspector() {
  const missionId = useMissionStore((s) => s.missionId)
  const pinnedLkp = useMissionStore((s) => s.pinnedLkp)
  const draftLkp = useMissionStore((s) => s.draftLkp)
  const terrainData = useMissionStore((s) => s.terrainData)
  const terrainField = useMissionStore((s) => s.terrainField)
  const terrainMaskMode = useMissionStore((s) => s.terrainMaskMode)
  const terrainMaskThreshold = useMissionStore((s) => s.terrainMaskThreshold)
  const setTerrainData = useMissionStore((s) => s.setTerrainData)
  const setTerrainField = useMissionStore((s) => s.setTerrainField)
  const setTerrainMaskMode = useMissionStore((s) => s.setTerrainMaskMode)
  const setTerrainMaskThreshold = useMissionStore((s) => s.setTerrainMaskThreshold)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const target = pinnedLkp ?? draftLkp
  const missionLoaded = Boolean(missionId && terrainData)

  const inspect = useCallback(async () => {
    if (!target) {
      setError('Pin or click a position on the map first')
      return
    }
    setLoading(true)
    setError(null)
    try {
      let res: Response
      try {
        res = await fetch(`${BACKEND_URL}/terrain/inspect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lkp: target }),
        })
      } catch (e) {
        throw new Error(
          `Cannot reach backend at ${BACKEND_URL}: ${
            e instanceof Error ? e.message : String(e)
          }`,
        )
      }
      if (!res.ok) throw new Error(await res.text())
      const data = (await res.json()) as TerrainData
      setTerrainData(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to inspect terrain')
    } finally {
      setLoading(false)
    }
  }, [target, setTerrainData])

  const clear = useCallback(() => {
    if (missionId) return
    setTerrainData(null)
    setError(null)
  }, [missionId, setTerrainData])

  const selectedRange = useMemo(() => {
    if (!terrainData || !terrainField) return null
    const values = terrainData.fields[terrainField]
    if (!values) return null
    return computeFieldRange(values, terrainField)
  }, [terrainData, terrainField])

  const selectedMeta = terrainData?.available.find((f) => f.id === terrainField)

  const gridHint = terrainData
    ? `${terrainData.rows}×${terrainData.cols} cells · ${terrainData.metadata.resolution_m.toFixed(0)} m/cell`
    : null

  return (
    <div className="terrain-inspector" aria-label="Terrain data inspector">
      <h3>Terrain Inspector</h3>
      <p className="layer-idle-hint">
        Per-cell inputs the grid engine uses at mission start (roads, elevation, slope,
        reachability, land, wind, sea current).
      </p>

      {missionId ? (
        <p className="layer-idle-hint">
          {missionLoaded
            ? `Loaded with Run Heatmap — engine grid (${gridHint}). Values are fixed at pin time.`
            : 'Run Heatmap to load init cell data for this mission.'}
        </p>
      ) : (
        <>
          <button type="button" className="pin-btn" onClick={inspect} disabled={loading || !target}>
            {loading ? 'Fetching terrain…' : 'Preview at pin (debug grid)'}
          </button>
          {!target && <p className="layer-idle-hint">Click the map to choose a point.</p>}
          {terrainData && !missionId && (
            <p className="layer-idle-hint">
              Debug preview ({gridHint}) — finer than the mission grid. Run Heatmap to use exact
              engine values.
            </p>
          )}
        </>
      )}

      {error && <p className="error">{error}</p>}

      {terrainData && (
        <>
          {terrainData.warnings?.length ? (
            <div className="terrain-warning">
              {terrainData.warnings.map((w) => (
                <p key={w} className="error">
                  {w}
                </p>
              ))}
            </div>
          ) : null}

          <ul className="terrain-field-list">
            <li>
              <label className="layer-toggle">
                <input
                  type="radio"
                  name="terrain-field"
                  checked={terrainField === null}
                  onChange={() => setTerrainField(null)}
                />
                <span>Off (hide overlay)</span>
              </label>
            </li>
            {terrainData.available.map((field) => (
              <li key={field.id}>
                <label className="layer-toggle">
                  <input
                    type="radio"
                    name="terrain-field"
                    checked={terrainField === field.id}
                    onChange={() => setTerrainField(field.id)}
                  />
                  <span>
                    {field.label}
                    {field.unit ? ` (${field.unit})` : ''}
                  </span>
                </label>
                {field.description && <p className="layer-hint">{field.description}</p>}
              </li>
            ))}
          </ul>

          {selectedMeta?.kind === 'scalar' ? (
            <div className="terrain-mask-controls">
              <label className="layer-toggle">
                <input
                  type="checkbox"
                  checked={terrainMaskMode}
                  onChange={(e) => setTerrainMaskMode(e.target.checked)}
                />
                <span>Mask mode (binary cells)</span>
              </label>
              {terrainMaskMode ? (
                <label className="field">
                  <span>Mask threshold — {(terrainMaskThreshold * 100).toFixed(0)}%</span>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={terrainMaskThreshold}
                    onChange={(e) => setTerrainMaskThreshold(Number(e.target.value))}
                  />
                </label>
              ) : null}
            </div>
          ) : null}

          {selectedMeta && selectedRange && (
            <div className="terrain-range">
              <p>
                {selectedMeta.kind === 'mask'
                  ? selectedMeta.id === 'is_land'
                    ? 'Highlighting WATER cells'
                    : 'Highlighting active cells'
                  : `Range: ${selectedRange.min.toFixed(2)} – ${selectedRange.max.toFixed(2)}${
                      selectedMeta.unit ? ` ${selectedMeta.unit}` : ''
                    }`}
              </p>
              {terrainData.field_stats?.[selectedMeta.id] ? (
                <p>
                  Coverage:{' '}
                  {(terrainData.field_stats[selectedMeta.id].nonzero_frac * 100).toFixed(1)}%
                </p>
              ) : null}
            </div>
          )}

          {!missionId && (
            <button type="button" className="secondary" onClick={clear}>
              Clear terrain data
            </button>
          )}
        </>
      )}
    </div>
  )
}

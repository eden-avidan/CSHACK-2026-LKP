import { useCallback, useRef } from 'react'
import { computeColorRange, probabilityToRGBA } from '../utils/colorScale'

const UPSCALE = 4

function sampleGrid(
  grid: Float32Array,
  rows: number,
  cols: number,
  rowF: number,
  colF: number,
): number {
  const r0 = Math.max(0, Math.min(rows - 1, Math.floor(rowF)))
  const c0 = Math.max(0, Math.min(cols - 1, Math.floor(colF)))
  const r1 = Math.min(rows - 1, r0 + 1)
  const c1 = Math.min(cols - 1, c0 + 1)
  const dr = rowF - r0
  const dc = colF - c0
  const v00 = grid[r0 * cols + c0]
  const v01 = grid[r0 * cols + c1]
  const v10 = grid[r1 * cols + c0]
  const v11 = grid[r1 * cols + c1]
  return (
    v00 * (1 - dr) * (1 - dc) +
    v01 * (1 - dr) * dc +
    v10 * dr * (1 - dc) +
    v11 * dr * dc
  )
}

export function useHeatmapPainter() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const paintSync = useCallback((grid: Float32Array, rows: number, cols: number): HTMLCanvasElement | null => {
    let canvas = canvasRef.current
    if (!canvas) {
      canvas = document.createElement('canvas')
      canvasRef.current = canvas
    }

    const outRows = rows * UPSCALE
    const outCols = cols * UPSCALE
    if (canvas.width !== outCols || canvas.height !== outRows) {
      canvas.width = outCols
      canvas.height = outRows
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    const imageData = ctx.createImageData(outCols, outRows)
    const range = computeColorRange(grid)

    for (let outRow = 0; outRow < outRows; outRow++) {
      const rowF = ((outRow + 0.5) / UPSCALE) - 0.5
      for (let outCol = 0; outCol < outCols; outCol++) {
        const colF = ((outCol + 0.5) / UPSCALE) - 0.5
        const p = sampleGrid(grid, rows, cols, rowF, colF)
        const [r, g, b, a] = probabilityToRGBA(p, range)
        const offset = (outRow * outCols + outCol) * 4
        imageData.data[offset] = r
        imageData.data[offset + 1] = g
        imageData.data[offset + 2] = b
        imageData.data[offset + 3] = a
      }
    }

    ctx.putImageData(imageData, 0, 0)
    return canvas
  }, [])

  const getCanvas = useCallback(() => canvasRef.current, [])

  return { paintSync, getCanvas }
}

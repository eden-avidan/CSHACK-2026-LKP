import { useCallback, useRef } from 'react'
import { computeColorRange, edgeFadeMultiplier, probabilityToRGBA } from '../utils/colorScale'

export function useHeatmapPainter() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  const paintSync = useCallback((grid: Float32Array, rows: number, cols: number): HTMLCanvasElement | null => {
    let canvas = canvasRef.current
    if (!canvas) {
      canvas = document.createElement('canvas')
      canvasRef.current = canvas
    }

    if (canvas.width !== cols || canvas.height !== rows) {
      canvas.width = cols
      canvas.height = rows
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return null

    const imageData = ctx.createImageData(cols, rows)
    const range = computeColorRange(grid)

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const i = row * cols + col
        const edge = edgeFadeMultiplier(row, col, rows, cols)
        const [r, g, b, a] = probabilityToRGBA(grid[i], range)
        const offset = i * 4
        imageData.data[offset] = r
        imageData.data[offset + 1] = g
        imageData.data[offset + 2] = b
        imageData.data[offset + 3] = Math.round(a * edge)
      }
    }

    ctx.putImageData(imageData, 0, 0)
    return canvas
  }, [])

  const getCanvas = useCallback(() => canvasRef.current, [])

  return { paintSync, getCanvas }
}

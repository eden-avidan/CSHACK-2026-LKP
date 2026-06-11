import { useEffect, useMemo, useState } from 'react'
import type { Map } from 'mapbox-gl'
import { useMissionStore } from '../../stores/missionStore'

const FLASH_DURATION_MS = 3000

interface DetectionFlashProps {
  map: Map | null
}

export function DetectionFlash({ map }: DetectionFlashProps) {
  const detection = useMissionStore((s) => s.detectionFlash)
  const setDetectionFlash = useMissionStore((s) => s.setDetectionFlash)
  const [screenPosition, setScreenPosition] = useState<{ x: number; y: number } | null>(null)

  const animationKey = useMemo(
    () => detection ? `${detection.mission_id}:${detection.frame ?? detection.timestamp}` : '',
    [detection],
  )

  useEffect(() => {
    if (!detection) return
    const timeout = window.setTimeout(() => setDetectionFlash(null), FLASH_DURATION_MS)
    return () => clearTimeout(timeout)
  }, [detection, setDetectionFlash])

  useEffect(() => {
    if (!map || !detection?.position) {
      setScreenPosition(null)
      return
    }

    const updatePosition = () => {
      const point = map.project([detection.position!.lon, detection.position!.lat])
      setScreenPosition({ x: point.x, y: point.y })
    }

    updatePosition()
    map.on('move', updatePosition)
    map.on('zoom', updatePosition)
    map.on('resize', updatePosition)
    return () => {
      map.off('move', updatePosition)
      map.off('zoom', updatePosition)
      map.off('resize', updatePosition)
    }
  }, [map, detection])

  if (!detection) return null

  if (detection.position && screenPosition) {
    return (
      <div
        key={animationKey}
        className="detection-location-flash"
        style={{ left: screenPosition.x, top: screenPosition.y }}
        aria-label={`Person detected with ${detection.confidence_percent.toFixed(1)} percent confidence`}
      >
        <span />
        <span />
        <span />
      </div>
    )
  }

  return (
    <div
      key={animationKey}
      className="detection-viewport-flash"
      aria-label={`Person detected with ${detection.confidence_percent.toFixed(1)} percent confidence`}
    />
  )
}

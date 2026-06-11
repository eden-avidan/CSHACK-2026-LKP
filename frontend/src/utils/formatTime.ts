/** Format seconds as H:MM:SS or M:SS for sidebar timers. */
export function formatDuration(totalSec: number): string {
  const s = Math.max(0, Math.floor(totalSec))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) {
    return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
  }
  return `${m}:${String(sec).padStart(2, '0')}`
}

/** Matches backend BASE_STEP_SEC — simulated seconds per tick at pace 1×. */
export const BASE_STEP_SEC = 10

function pad2(n: number): string {
  return String(n).padStart(2, '0')
}

/** Format a Date for `<input type="datetime-local">` value. */
export function toDatetimeLocalValue(d: Date): string {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(d.getMinutes())}`
}

/** Simulated scenario clock: LKP time + tick_count × step_sec. */
export function simulatedDateTime(
  lkpTimestamp: string,
  tickCount: number,
  stepSec: number,
): Date {
  const baseMs = new Date(lkpTimestamp).getTime()
  return new Date(baseMs + tickCount * stepSec * 1000)
}

/** Human-readable date/time for offline simulation display. */
export function formatSimulatedDateTime(date: Date): string {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

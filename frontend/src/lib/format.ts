export function formatNumber(n: number, decimals = 2): string {
  return n.toFixed(decimals)
}

export function formatPercent(n: number): string {
  return (n * 100).toFixed(1) + '%'
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.round(seconds % 60)
  return `${mins}m ${secs}s`
}

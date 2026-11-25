// Ported from dashboard_app.py - purely presentational, so kept client-side
// rather than round-tripped through the API.
export const SEVERITY_COLORS: Record<number, string> = {
  1: '#ff4b4b',
  2: '#f59e0b',
  3: '#d2a106',
  4: '#58a6ff',
  5: '#2ea043',
}

export const SEVERITY_ICONS: Record<number, string> = {
  1: '🔴',
  2: '🟠',
  3: '🟡',
  4: '🔵',
  5: '🟢',
}

export const ACCENT_COLORS: Record<string, string> = {
  blue: '#58a6ff',
  orange: '#f59e0b',
  green: '#2ea043',
  purple: '#a371f7',
  red: '#ff4b4b',
}

export const DASHBOARD_VERSION = 'v1.1 (React)'

export const AUTO_RUN_LIMIT_DEFAULT = 10

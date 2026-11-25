import React from 'react'
import { ACCENT_COLORS, SEVERITY_COLORS } from '../constants'

export function MetricCard({ label, value, accent = 'blue' }: { label: string; value: React.ReactNode; accent?: string }) {
  const color = ACCENT_COLORS[accent] ?? accent
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-accent" style={{ background: color }} />
    </div>
  )
}

export function MetricGrid({ children }: { children: React.ReactNode }) {
  return <div className="metric-grid">{children}</div>
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="section-label">{children}</div>
}

export function SeverityLegend({ meanings }: { meanings: Record<string, string> }) {
  return (
    <div className="legend-row">
      {[1, 2, 3, 4, 5].map((r) => (
        <span key={r} className="legend-chip">
          <span className="legend-dot" style={{ background: SEVERITY_COLORS[r] }} />
          {r} · {meanings[String(r)] ?? meanings[r as any] ?? ''}
        </span>
      ))}
    </div>
  )
}

export function Callout({ kind, children }: { kind: 'info' | 'warn' | 'error' | 'success'; children: React.ReactNode }) {
  return <div className={`callout callout-${kind}`}>{children}</div>
}

export function ProgressBar({ current, total, label }: { current: number; total: number; label: string }) {
  const pct = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0
  return (
    <div className="progress-wrap">
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="small-muted" style={{ marginTop: 6 }}>
        {label}
      </p>
    </div>
  )
}

export function Expander({ title, children, defaultOpen = false }: { title: React.ReactNode; children: React.ReactNode; defaultOpen?: boolean }) {
  return (
    <details className="expander" open={defaultOpen}>
      <summary>{title}</summary>
      <div className="expander-body">{children}</div>
    </details>
  )
}

export function RuleCard({ title, body, accent = 'blue' }: { title: string; body: string; accent?: 'blue' | 'green' | 'orange' }) {
  if (!body) return null
  return (
    <div className={`rule-doc-card accent-${accent}`}>
      <div className="rule-doc-title">{title}</div>
      <div className="rule-doc-body">{body}</div>
    </div>
  )
}

export function severityRankColor(rank: number | null | undefined) {
  if (rank == null) return undefined
  return SEVERITY_COLORS[rank]
}

/** Formats an ISO timestamp (from the backend, always UTC) in Pakistan
 * Standard Time - PKT is a fixed UTC+5 offset year-round (no DST), so this
 * is just a display-layer conversion; the API itself keeps returning plain
 * UTC/ISO timestamps. */
export function formatPKT(iso: string | null | undefined): string {
  if (!iso) return '-'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Karachi',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h12',
  }).formatToParts(d)
  const get = (type: string) => parts.find((p) => p.type === type)?.value ?? ''
  const ampm = get('dayPeriod').toUpperCase().replace(/\./g, '')
  return `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}:${get('second')} ${ampm} PKT`
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return '-'
  if (seconds < 60) return `${seconds.toFixed(1)} sec`
  const minutes = Math.floor(seconds / 60)
  const secs = seconds - minutes * 60
  return `${minutes} min ${secs.toFixed(0)} sec`
}

export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: { key: string; label: string }[]
  active: string
  onChange: (key: string) => void
}) {
  return (
    <div className="tabs-row">
      {tabs.map((t) => (
        <button
          key={t.key}
          className={`tab-btn ${active === t.key ? 'active' : ''}`}
          onClick={() => onChange(t.key)}
          type="button"
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

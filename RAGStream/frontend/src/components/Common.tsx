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

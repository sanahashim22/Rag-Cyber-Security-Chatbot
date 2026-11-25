import type { MetaInfo } from '../api/types'
import { SeverityLegend } from './Common'

export function Sidebar({ meta }: { meta: MetaInfo | undefined }) {
  if (!meta) {
    return (
      <div className="app-sidebar">
        <div className="info-card">Loading…</div>
      </div>
    )
  }

  const badge = meta.ollama_is_local ? '🖥️ Local (CPU)' : '🌐 Remote (GPU test)'

  return (
    <div className="app-sidebar">
      <div className="info-card">
        <div className="info-card-title">⚙️ Ollama target</div>
        <p style={{ margin: 0 }}>
          {badge}
          <br />
          <code>{meta.ollama_host}</code>
        </p>
      </div>

      <div className="info-card">
        <div className="info-card-title">📚 Loaded standards</div>
        <ul>
          {meta.doc_names.length > 0 ? (
            meta.doc_names.map((n) => <li key={n}>{n}</li>)
          ) : (
            <li>Built-in NIST CSF 2.0 summary</li>
          )}
        </ul>
      </div>

      <div className="info-card">
        <div className="info-card-title">ℹ️ How this works</div>
        <p style={{ margin: 0 }}>
          Each alert is ranked 1-5 for severity, 3 times per model, citing its own Snort rule documentation (exact
          sid lookup) plus retrieved NIST/ISO context. An independent judge model (<code>{meta.judge_model}</code>)
          scores how well each justification is actually supported by the evidence.
        </p>
      </div>

      <div className="panel">
        <div className="info-card-title">🎯 Severity scale</div>
        <SeverityLegend meanings={meta.severity_meanings} />
      </div>
    </div>
  )
}

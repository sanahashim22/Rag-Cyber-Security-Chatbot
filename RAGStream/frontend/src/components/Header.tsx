import { DASHBOARD_VERSION } from '../constants'

export function Header() {
  return (
    <div className="hero-card">
      <div className="hero-title-row">
        <div className="hero-title">🛡️ Snort Alert Severity Ranking Dashboard</div>
        <span className="version-pill">{DASHBOARD_VERSION}</span>
      </div>
      <div className="hero-caption">
        AI-assisted Snort/Suricata alert severity ranking, grounded in NIST CSF 2.0, ISO/IEC 27001:2022, and
        each alert's own rule documentation.
      </div>
    </div>
  )
}

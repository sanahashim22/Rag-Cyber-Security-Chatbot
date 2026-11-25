import { useState } from 'react'
import type { ScanResult } from '../api/types'

const ALL_FILES_CHOICE = 'All files (combined)'

export { ALL_FILES_CHOICE }

export function FileDropdown({
  scan,
  value,
  onChange,
}: {
  scan: ScanResult
  value: string
  onChange: (v: string) => void
}) {
  return (
    <label>
      📄 Which file do you want to rank?
      <br />
      <select value={value} onChange={(e) => onChange(e.target.value)} style={{ minWidth: 320 }}>
        <option value={ALL_FILES_CHOICE}>{ALL_FILES_CHOICE}</option>
        {scan.file_summaries.map((f) => (
          <option key={f.name} value={f.name}>
            {f.name} ({f.count})
          </option>
        ))}
      </select>
    </label>
  )
}

export function AutoRunLimitConfirm({
  fileChoice,
  totalParsed,
  autoRunLimit,
  onConfirm,
}: {
  fileChoice: string
  totalParsed: number
  autoRunLimit: number
  onConfirm: (n: number) => void
}) {
  const [n, setN] = useState(Math.min(autoRunLimit, totalParsed))

  return (
    <div className="panel">
      <div className="info-card-title">⚠️ {totalParsed} alerts found</div>
      <p>
        <b>{fileChoice}</b> has <b>{totalParsed} alerts</b>. Ranking all of them (3 runs per alert, x4 models in
        the 4-Model Comparison tab) can take a while on local models. Choose how many to process below.
      </p>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <input
          type="number"
          min={1}
          max={totalParsed}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
          style={{ width: 120 }}
        />
        <button className="btn-primary" onClick={() => onConfirm(n)}>
          ▶ Process entered number
        </button>
        <button onClick={() => onConfirm(totalParsed)}>Process all {totalParsed}</button>
      </div>
    </div>
  )
}

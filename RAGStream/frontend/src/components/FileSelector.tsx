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
  // Kept as a raw string, not a number, while the field is being edited -
  // a controlled <input type="number"> whose value is coerced through
  // Number() on every keystroke snaps an emptied field straight to "0"
  // (Number("") === 0), and typing after that produces "02" etc: the DOM
  // silently ignores value updates that parse to the same number as what's
  // already displayed (e.g. assigning "2" onto a field showing "02" is a
  // no-op, since both parse to 2). Parsing only happens when the value is
  // actually used (onConfirm), with a clamp as a fallback for "" or NaN.
  const [nText, setNText] = useState(String(Math.min(autoRunLimit, totalParsed)))

  const clamp = (raw: string) => {
    const parsed = Number(raw)
    if (!raw || Number.isNaN(parsed)) return 1
    return Math.min(Math.max(1, Math.trunc(parsed)), totalParsed)
  }

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
          value={nText}
          onChange={(e) => setNText(e.target.value)}
          style={{ width: 120 }}
        />
        <button className="btn-primary" onClick={() => onConfirm(clamp(nText))}>
          ▶ Process entered number
        </button>
        <button onClick={() => onConfirm(totalParsed)}>Process all {totalParsed}</button>
      </div>
    </div>
  )
}

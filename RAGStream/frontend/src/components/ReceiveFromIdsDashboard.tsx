import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { Callout, Expander } from './Common'

export function ReceiveFromIdsDashboard({
  baseUrl,
  token,
  onChange,
}: {
  baseUrl: string
  token: string
  onChange: (baseUrl: string, token: string) => void
}) {
  const [mode, setMode] = useState<'window' | 'last_n'>('window')
  const [window, setWindowLen] = useState('5 minutes')
  const [lastN, setLastN] = useState(5)
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null)
  const queryClient = useQueryClient()

  const { data: lengths } = useQuery({ queryKey: ['ids-window-lengths'], queryFn: api.idsWindowLengths })

  const pull = useMutation({
    mutationFn: () =>
      api.pullAlerts({
        base_url: baseUrl,
        token,
        length: mode === 'window' ? window : undefined,
        last_n: mode === 'last_n' ? lastN : undefined,
      }),
    onSuccess: (res) => {
      setMessage({ ok: res.ok, text: res.message })
      if (res.ok) queryClient.invalidateQueries({ queryKey: ['alerts-scan'] })
    },
  })

  return (
    <Expander title="📥 Receive from IDS-Dashboard">
      <p className="small-muted">
        Pulls alerts from IDS-Dashboard's export API and saves them into <code>alert_inputs/</code> - same as
        dropping a file in by hand.
      </p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 10 }}>
        <label>
          IDS-Dashboard base URL
          <br />
          <input type="text" value={baseUrl} onChange={(e) => onChange(e.target.value, token)} style={{ width: 280 }} />
        </label>
        <label>
          IDS_API_TOKEN
          <br />
          <input type="password" value={token} onChange={(e) => onChange(baseUrl, e.target.value)} style={{ width: 200 }} />
        </label>
      </div>

      <div className="checkbox-row" style={{ marginBottom: 10 }}>
        <label>
          <input type="radio" checked={mode === 'window'} onChange={() => setMode('window')} /> Time window
        </label>
        <label>
          <input type="radio" checked={mode === 'last_n'} onChange={() => setMode('last_n')} /> Last N alerts
        </label>
      </div>

      {mode === 'window' ? (
        <select value={window} onChange={(e) => setWindowLen(e.target.value)}>
          {(lengths?.lengths ?? ['5 minutes']).map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
      ) : (
        <input
          type="number"
          min={1}
          max={150000}
          value={lastN}
          onChange={(e) => setLastN(Number(e.target.value))}
          style={{ width: 120 }}
        />
      )}

      <div style={{ marginTop: 12 }}>
        <button className="btn-primary" onClick={() => pull.mutate()} disabled={pull.isPending}>
          {pull.isPending ? 'Pulling…' : '▶ Pull alerts now'}
        </button>
      </div>

      {message && <Callout kind={message.ok ? 'success' : 'error'}>{message.text}</Callout>}
    </Expander>
  )
}

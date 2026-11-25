import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { Callout } from './Common'

/** Polls /alerts/scan every 4s - same role as dashboard_app.py's
 * st.fragment(run_every=4) alert-folder watcher, just over HTTP. */
export function AlertInputsStatus() {
  const { data: scan } = useQuery({
    queryKey: ['alerts-scan'],
    queryFn: api.scanAlerts,
    refetchInterval: 4000,
  })

  if (!scan) return null

  if (scan.file_summaries.length === 0) {
    return (
      <Callout kind="info">
        Drop one or more Snort/Suricata alerts JSON files into the folder below - a single alert object, a list
        of alerts, or an <code>{'{"alerts": [...]}'}</code> export are all accepted, per file. This updates
        automatically every few seconds:
        <br />
        <code>{scan.alert_inputs_dir}</code>
      </Callout>
    )
  }

  return (
    <>
      {scan.warnings.map((w, i) => (
        <Callout kind="warn" key={i}>
          {w}
        </Callout>
      ))}
      <Callout kind="success">
        Found {scan.file_summaries.length} file(s) in <code>alert_inputs/</code>, <b>{scan.total_parsed} alerts
        combined</b>:{' '}
        {scan.file_summaries.map((f) => `${f.name} (${f.count})`).join(', ')}
      </Callout>
    </>
  )
}

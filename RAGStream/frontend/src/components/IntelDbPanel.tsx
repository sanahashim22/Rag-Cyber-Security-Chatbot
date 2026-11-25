import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import { Callout, Expander } from './Common'

export function IntelDbPanel({ baseUrl, token }: { baseUrl: string; token: string }) {
  const queryClient = useQueryClient()
  const [remoteInfo, setRemoteInfo] = useState<{ ok: boolean; info?: any; error?: string } | null>(null)
  const [downloadJobId, setDownloadJobId] = useState<string | null>(null)

  const { data: local } = useQuery({ queryKey: ['intel-local'], queryFn: api.intelLocal })

  const checkRemote = useMutation({
    mutationFn: () => api.intelRemote({ base_url: baseUrl, token }),
    onSuccess: setRemoteInfo,
  })

  const startDownload = useMutation({
    mutationFn: () => api.intelDownloadStart({ base_url: baseUrl, token }),
    onSuccess: (r) => setDownloadJobId(r.job_id),
  })

  const { data: downloadJob } = useQuery({
    queryKey: ['intel-download-job', downloadJobId],
    queryFn: () => api.intelDownloadStatus(downloadJobId!),
    enabled: !!downloadJobId,
    refetchInterval: (query) => (query.state.data?.status === 'running' ? 1000 : false),
  })

  if (downloadJob?.status === 'done' && downloadJobId) {
    // one-shot: refresh local status, then stop tracking this job
    queryClient.invalidateQueries({ queryKey: ['intel-local'] })
  }

  return (
    <Expander title="🗄️ Threat-intel database (intel.db)">
      <p className="small-muted">
        Shared database from the IDS sensor PC (Talos/Snort rule docs, CVEs with CVSS/EPSS/CISA-KEV, MITRE
        ATT&CK). When present, ranking looks up each alert's rule here FIRST, falling back to{' '}
        <code>rule_docs_preprocessed_by_sid.json</code> only for a sid intel.db doesn't have. Manual refresh
        only - it's about 1.4GB.
      </p>

      {!local?.downloaded ? (
        <Callout kind="info">
          Not downloaded yet - ranking will use <code>rule_docs_preprocessed_by_sid.json</code> only until you
          download it.
        </Callout>
      ) : (
        <Callout kind="success">
          Local copy: <b>{Math.round((local.status!.bytes ?? 0) / (1 << 20))} MB</b>, downloaded{' '}
          {local.status!.modified}. Sources:{' '}
          {Object.entries(local.status!.sources)
            .map(([n, d]) => `${n} (${d})`)
            .join(', ') || 'unknown'}
        </Callout>
      )}

      <div style={{ display: 'flex', gap: 10 }}>
        <button onClick={() => checkRemote.mutate()} disabled={checkRemote.isPending}>
          🔍 Check IDS-Dashboard for updates
        </button>
        <button
          className="btn-primary"
          onClick={() => startDownload.mutate()}
          disabled={startDownload.isPending || downloadJob?.status === 'running'}
        >
          ⬇ Download / update intel.db
        </button>
      </div>

      {remoteInfo &&
        (remoteInfo.ok ? (
          <Callout kind="info">
            On IDS-Dashboard: <b>{Math.round((remoteInfo.info.bytes ?? 0) / (1 << 20))} MB</b>, last modified{' '}
            {remoteInfo.info.modified}.
          </Callout>
        ) : (
          <Callout kind="error">{remoteInfo.error}</Callout>
        ))}

      {downloadJob && downloadJob.status === 'running' && (
        <Callout kind="info">{downloadJob.progress.message}</Callout>
      )}
      {downloadJob && downloadJob.status === 'done' && (
        <Callout kind={downloadJob.result?.ok ? 'success' : 'error'}>{downloadJob.result?.message}</Callout>
      )}
    </Expander>
  )
}

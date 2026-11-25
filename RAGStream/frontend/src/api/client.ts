import type {
  AlertEntry,
  ComparisonResult,
  EntriesResult,
  IntelLocalStatus,
  JobStatus,
  MetaInfo,
  ScanResult,
  SingleRunResult,
} from './types'

// Always relative - works unchanged whether loaded from the Vite dev
// server (proxied to the backend, see vite.config.ts) or from the built
// static bundle served same-origin by FastAPI (localhost OR an ngrok URL).
const BASE = '/api'

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!resp.ok) {
    let detail = resp.statusText
    try {
      const body = await resp.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* ignore */
    }
    throw new Error(`${resp.status} ${detail}`)
  }
  return resp.json()
}

export const api = {
  meta: () => req<MetaInfo>('/meta'),

  scanAlerts: () => req<ScanResult>('/alerts/scan'),
  getEntries: (file?: string) =>
    req<EntriesResult>(`/alerts/entries${file ? `?file=${encodeURIComponent(file)}` : ''}`),
  pullAlerts: (body: { base_url: string; token: string; length?: string; last_n?: number }) =>
    req<{ ok: boolean; message: string }>('/alerts/pull', { method: 'POST', body: JSON.stringify(body) }),
  idsWindowLengths: () => req<{ lengths: string[] }>('/alerts/ids-dashboard-window-lengths'),

  intelLocal: () => req<IntelLocalStatus>('/intel/local'),
  intelRemote: (body: { base_url: string; token: string }) =>
    req<{ ok: boolean; info?: any; error?: string }>('/intel/remote', { method: 'POST', body: JSON.stringify(body) }),
  intelDownloadStart: (body: { base_url: string; token: string }) =>
    req<{ job_id: string }>('/intel/download', { method: 'POST', body: JSON.stringify(body) }),
  intelDownloadStatus: (jobId: string) => req<JobStatus<{ ok: boolean; message: string }>>(`/intel/download/jobs/${jobId}`),

  models: () => req<{ models: string[]; auto_run_limit: number }>('/ranking/models'),
  runRanking: (body: {
    entries: AlertEntry[]
    model: string
    run_judge: boolean
    run_mismatch: boolean
    run_correlation: boolean
  }) => req<{ job_id: string }>('/ranking/run', { method: 'POST', body: JSON.stringify(body) }),
  runComparison: (body: {
    entries: AlertEntry[]
    run_judge: boolean
    run_mismatch: boolean
    run_correlation: boolean
  }) => req<{ job_id: string }>('/ranking/compare', { method: 'POST', body: JSON.stringify(body) }),
  rankingJob: (jobId: string) => req<JobStatus<SingleRunResult>>(`/ranking/jobs/${jobId}`),
  comparisonJob: (jobId: string) => req<JobStatus<ComparisonResult>>(`/ranking/jobs/${jobId}`),

  chatSend: (question: string, model: string) =>
    req<{ question: string; answer: string }>('/chat/message', {
      method: 'POST',
      body: JSON.stringify({ question, model }),
    }),
  chatHistory: () => req<{ history: { question: string; answer: string }[] }>('/chat/history'),
  chatClear: () => req<{ ok: boolean }>('/chat/history', { method: 'DELETE' }),
}

export function downloadBlob(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function downloadBase64(filename: string, base64: string, mime: string) {
  const bytes = atob(base64)
  const arr = new Uint8Array(bytes.length)
  for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i)
  const blob = new Blob([arr], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

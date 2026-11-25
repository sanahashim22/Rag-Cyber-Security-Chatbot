import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { AlertEntry, MetaInfo } from '../api/types'
import { Callout, ProgressBar, Tabs } from './Common'
import { RankingOverview, ReasoningView, ReportView, TimingView } from './RankingShared'

export function SingleModelPanel({
  entries,
  models,
  meta,
  onBusyChange,
  onModelChange,
}: {
  entries: AlertEntry[]
  models: string[]
  meta: MetaInfo | undefined
  onBusyChange: (busy: boolean) => void
  onModelChange?: (model: string) => void
}) {
  const [model, setModel] = useState(models[0])

  useEffect(() => {
    onModelChange?.(model)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model])
  const [runJudge, setRunJudge] = useState(false)
  const [runMismatch, setRunMismatch] = useState(false)
  const [runCorrelation, setRunCorrelation] = useState(false)
  // Raw string while editing, not a number - see FileSelector.tsx's
  // AutoRunLimitConfirm for why a controlled number input coerced through
  // Number() on every keystroke breaks.
  const [repeatsText, setRepeatsText] = useState('1')
  const repeats = Math.max(1, Math.trunc(Number(repeatsText)) || 1)
  const [jobId, setJobId] = useState<string | null>(null)
  const [tab, setTab] = useState('ranking')

  const start = useMutation({
    mutationFn: () =>
      api.runRanking({
        entries, model, run_judge: runJudge, run_mismatch: runMismatch, run_correlation: runCorrelation, repeats,
      }),
    onSuccess: (r) => setJobId(r.job_id),
  })

  const { data: job } = useQuery({
    queryKey: ['ranking-job', jobId],
    queryFn: () => api.rankingJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data?.status === 'running' ? 1500 : false),
  })

  useEffect(() => {
    onBusyChange(job?.status === 'running' || start.isPending)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status, start.isPending])

  const result = job?.status === 'done' ? job.result : null

  return (
    <div>
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-end', flexWrap: 'wrap', marginBottom: 22 }}>
        <label>
          <span className="field-label">Ranking model</span>
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <button className="btn-primary" onClick={() => start.mutate()} disabled={job?.status === 'running' || start.isPending}>
          ▶ Run ranking
        </button>
      </div>

      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 24 }}>
        <label className="checkbox-row">
          <input type="checkbox" checked={runJudge} onChange={(e) => setRunJudge(e.target.checked)} /> Judge LLM
        </label>
        <label className="checkbox-row">
          <input type="checkbox" checked={runMismatch} onChange={(e) => setRunMismatch(e.target.checked)} /> Mismatch
          justification
        </label>
        <label className="checkbox-row">
          <input type="checkbox" checked={runCorrelation} onChange={(e) => setRunCorrelation(e.target.checked)} />{' '}
          Combined/correlated analysis
        </label>
      </div>

      <div style={{ marginBottom: 24 }}>
        <label>
          <span className="field-label">Runs per alert</span>
          <input
            type="number"
            min={1}
            value={repeatsText}
            onChange={(e) => setRepeatsText(e.target.value)}
            style={{ width: 120 }}
          />
        </label>
        <p className="small-muted" style={{ marginTop: 6 }}>
          How many times to re-rank each alert with the same model, to check consistency. Each run is a separate
          LLM call, so more runs takes proportionally longer.
        </p>
      </div>

      {job?.status === 'running' && (
        <ProgressBar current={job.progress.current} total={job.progress.total} label={`⏳ ${job.progress.message}`} />
      )}
      {job?.status === 'error' && <Callout kind="error">{job.error}</Callout>}

      {!result ? (
        job?.status !== 'running' && (
          <Callout kind="info">Select a model and click <b>Run ranking</b> to rank these alerts.</Callout>
        )
      ) : (
        <>
          <Tabs
            active={tab}
            onChange={setTab}
            tabs={[
              { key: 'ranking', label: '📋 Ranking' },
              { key: 'reasoning', label: '🧠 Reasoning' },
              { key: 'report', label: '📄 Report & Material' },
              { key: 'timing', label: '⏱️ Timing' },
            ]}
          />
          {tab === 'ranking' && (
            <RankingOverview
              stats={result.stats}
              alertTable={result.alert_table}
              correlation={result.correlation}
              judgeModel={meta?.judge_model ?? 'gemma2:2b'}
              runJudge={runJudge}
            />
          )}
          {tab === 'reasoning' && <ReasoningView processed={result.processed} />}
          {tab === 'report' && (
            <ReportView markdown={result.report_markdown} filename={`snort_ranking_report_${model.replace(':', '_')}.md`} />
          )}
          {tab === 'timing' && (
            <TimingView
              processed={result.processed}
              correlation={result.correlation}
              timingText={result.timing_text}
              filename={`timing_report_${model.replace(':', '_')}.txt`}
            />
          )}
        </>
      )}
    </div>
  )
}

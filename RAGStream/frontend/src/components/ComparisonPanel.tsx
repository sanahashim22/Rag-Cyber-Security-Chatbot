import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, downloadBase64 } from '../api/client'
import type { AlertEntry } from '../api/types'
import { Callout, Tabs, severityRankColor } from './Common'
import { ReasoningView, ReportView, TimingView } from './RankingShared'

export function ComparisonPanel({
  entries,
  models,
  onBusyChange,
}: {
  entries: AlertEntry[]
  models: string[]
  onBusyChange: (busy: boolean) => void
}) {
  const [runJudge, setRunJudge] = useState(true)
  const [runMismatch, setRunMismatch] = useState(true)
  const [runCorrelation, setRunCorrelation] = useState(true)
  const [jobId, setJobId] = useState<string | null>(null)
  const [tab, setTab] = useState('ranking')
  const [activeModel, setActiveModel] = useState(models[0])

  const start = useMutation({
    mutationFn: () => api.runComparison({ entries, run_judge: runJudge, run_mismatch: runMismatch, run_correlation: runCorrelation }),
    onSuccess: (r) => setJobId(r.job_id),
  })

  const { data: job } = useQuery({
    queryKey: ['comparison-job', jobId],
    queryFn: () => api.comparisonJob(jobId!),
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
      <div style={{ marginBottom: 10 }}>
        <button className="btn-primary" onClick={() => start.mutate()} disabled={job?.status === 'running' || start.isPending}>
          ▶ Run all 4 models
        </button>
      </div>

      <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 14 }}>
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

      {job?.status === 'running' && <Callout kind="info">⏳ {job.progress.message}</Callout>}
      {job?.status === 'error' && <Callout kind="error">{job.error}</Callout>}

      {!result ? (
        <Callout kind="info">
          Click <b>Run all 4 models</b> to rank every alert with all 4 models (3 runs each) and compare them side
          by side. This runs 12 full passes per alert, so it takes roughly 4x longer than the single-model tab.
        </Callout>
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
            <>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Consistency (3/3)</th>
                    <th>Avg Judge Score</th>
                    <th>Rank 1</th>
                    <th>Rank 2</th>
                    <th>Rank 3</th>
                    <th>Rank 4</th>
                    <th>Rank 5</th>
                  </tr>
                </thead>
                <tbody>
                  {result.model_summary.map((s) => (
                    <tr key={s.model}>
                      <td>{s.model}</td>
                      <td>
                        {s.consistent_count}/{s.total_alerts}
                      </td>
                      <td>{s.avg_judge != null ? s.avg_judge.toFixed(3) : '-'}</td>
                      {s.rank_counts.map((c, i) => (
                        <td key={i}>{c}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>

              <table className="data-table" style={{ marginTop: 16 }}>
                <thead>
                  <tr>
                    <th>Alert ID</th>
                    <th>Source</th>
                    <th>Alert Message</th>
                    <th>Actual Ranking</th>
                    <th>SID</th>
                    {models.map((m) => (
                      <th key={m}>{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.comparison_rows.map((row, i) => (
                    <tr key={i}>
                      <td>{row.alert_id}</td>
                      <td>{row.source}</td>
                      <td>{row.alert_message}</td>
                      <td>{row.actual_ranking}</td>
                      <td>{row.sid}</td>
                      {models.map((m) => {
                        const mr = row.models[m]
                        return (
                          <td key={m} style={{ background: severityRankColor(mr.majority_rank) + '33' }}>
                            <b>{mr.majority_rank}</b> ({mr.ranks.join(',')}) · judge {mr.avg_judge_score ?? '-'}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>

              <div style={{ marginTop: 12 }}>
                <button
                  onClick={() => downloadBase64('model_comparison.xlsx', result.excel_base64, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                >
                  ⬇ Download comparison workbook (.xlsx)
                </button>
              </div>
            </>
          )}

          {tab === 'reasoning' && (
            <>
              <p className="small-muted">
                Rule documentation and retrieved NIST/ISO context are identical for every model on a given alert, so
                they're shown once per alert; each model tab below has its own 3 runs.
              </p>
              <ModelTabs models={models} active={activeModel} onChange={setActiveModel} />
              <ReasoningView processed={result.per_model[activeModel].processed} />
            </>
          )}

          {tab === 'report' && (
            <>
              <ModelTabs models={models} active={activeModel} onChange={setActiveModel} />
              <ReportView
                markdown={result.per_model[activeModel].report_markdown}
                filename={`snort_ranking_report_${activeModel.replace(':', '_')}.md`}
              />
            </>
          )}

          {tab === 'timing' && (
            <>
              <ModelTabs models={models} active={activeModel} onChange={setActiveModel} />
              <TimingView
                processed={result.per_model[activeModel].processed}
                correlation={result.per_model[activeModel].correlation}
                timingText={result.timing_text}
                filename="timing_report_4_model_comparison.txt"
              />
            </>
          )}
        </>
      )}
    </div>
  )
}

function ModelTabs({ models, active, onChange }: { models: string[]; active: string; onChange: (m: string) => void }) {
  return (
    <div className="tabs-row">
      {models.map((m) => (
        <button key={m} className={`tab-btn ${active === m ? 'active' : ''}`} onClick={() => onChange(m)} type="button">
          {m}
        </button>
      ))}
    </div>
  )
}

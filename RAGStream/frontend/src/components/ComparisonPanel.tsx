import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { api, downloadBase64 } from '../api/client'
import type { AlertEntry } from '../api/types'
import { Callout, ProgressBar, SectionLabel, Tabs, severityRankColor } from './Common'
import { CorrelationSection, ReasoningView, ReportView, TimingView } from './RankingShared'

export function ComparisonPanel({
  entries,
  models,
  onBusyChange,
}: {
  entries: AlertEntry[]
  models: string[]
  onBusyChange: (busy: boolean) => void
}) {
  const [runJudge, setRunJudge] = useState(false)
  const [runMismatch, setRunMismatch] = useState(false)
  const [runCorrelation, setRunCorrelation] = useState(false)
  const [repeatsText, setRepeatsText] = useState('1')
  const repeats = Math.max(1, Math.trunc(Number(repeatsText)) || 1)
  const [jobId, setJobId] = useState<string | null>(null)
  const [tab, setTab] = useState('ranking')
  const [activeModel, setActiveModel] = useState(models[0])
  const [correlationModel, setCorrelationModel] = useState(models[0])

  const start = useMutation({
    mutationFn: () =>
      api.runComparison({
        entries, run_judge: runJudge, run_mismatch: runMismatch, run_correlation: runCorrelation, repeats,
      }),
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
      <div className="panel options-card">
        <div className="info-card-title">⚙️ Comparison options</div>

        <button className="btn-primary" onClick={() => start.mutate()} disabled={job?.status === 'running' || start.isPending}>
          ▶ Run all 4 models
        </button>

        <div>
          <span className="field-label">Optional extra checks</span>
          <div className="checkbox-group">
            <label className={`toggle-chip ${runJudge ? 'checked' : ''}`}>
              <input type="checkbox" checked={runJudge} onChange={(e) => setRunJudge(e.target.checked)} />
              <span className="chip-dot" /> Judge LLM
            </label>
            <label className={`toggle-chip ${runMismatch ? 'checked' : ''}`}>
              <input type="checkbox" checked={runMismatch} onChange={(e) => setRunMismatch(e.target.checked)} />
              <span className="chip-dot" /> Mismatch justification
            </label>
            <label className={`toggle-chip ${runCorrelation ? 'checked' : ''}`}>
              <input type="checkbox" checked={runCorrelation} onChange={(e) => setRunCorrelation(e.target.checked)} />
              <span className="chip-dot" /> Combined/correlated analysis
            </label>
          </div>
        </div>

        <div>
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
            How many times to re-rank each alert with each model, to check consistency. Each run is a separate LLM
            call, so more runs takes proportionally longer.
          </p>
        </div>
      </div>

      {job?.status === 'running' && (
        <ProgressBar current={job.progress.current} total={job.progress.total} label={`⏳ ${job.progress.message}`} />
      )}
      {job?.status === 'error' && <Callout kind="error">{job.error}</Callout>}

      {!result ? (
        job?.status !== 'running' && (
          <Callout kind="info">
            Click <b>Run all 4 models</b> to rank every alert with all 4 models ({repeats} run(s) each) and compare
            them side by side. This runs {models.length * repeats} full pass(es) per alert, so it takes roughly 4x
            longer than the single-model tab.
          </Callout>
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
            <>
              <SectionLabel>Model summary</SectionLabel>
              <div className="table-scroll">
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
              </div>
              <p className="small-muted" style={{ marginTop: 10 }}>
                These are pipeline-measured stats only (consistency across runs, independent judge score) - there's
                no external "correct answer" baked in, so no model is labeled best/worst here.
              </p>

              <SectionLabel>Per-alert comparison</SectionLabel>
              <div className="table-scroll">
                <table className="data-table">
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
                        <td style={{ minWidth: 220 }}>{row.alert_message}</td>
                        <td>{row.actual_ranking}</td>
                        <td>{row.sid}</td>
                        {models.map((m) => {
                          const mr = row.models[m]
                          return (
                            <td key={m} style={{ background: severityRankColor(mr.majority_rank) + '26', minWidth: 130 }}>
                              <div className="model-cell">
                                <span className="rank-badge">{mr.majority_rank}</span>
                                <span className="model-detail">
                                  {mr.ranks.length === 1
                                    ? `rank: ${mr.ranks[0]}`
                                    : `ranks across ${mr.ranks.length} runs: ${mr.ranks.join(', ')}`}
                                </span>
                                <span className="model-detail">judge {mr.avg_judge_score ?? '-'}</span>
                              </div>
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="small-muted" style={{ marginTop: 10 }}>
                Each model's colored column is its majority rank across {repeats} run(s); "runs" shows the raw
                values and "judge" the average judge score.
              </p>

              <div style={{ marginTop: 16 }}>
                <button
                  onClick={() => downloadBase64('model_comparison.xlsx', result.excel_base64, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                >
                  ⬇ Download comparison workbook (.xlsx)
                </button>
              </div>

              {models.some((m) => result.per_model[m].correlation !== null) && (
                <>
                  <SectionLabel>Combined / correlated alert groups (per model)</SectionLabel>
                  <p className="small-muted">
                    Each model gets its own extra pass looking at the whole batch together and suggesting alert_id
                    groups that collectively look more dangerous than any single alert alone - entirely a 🔗
                    AI-suggested judgment call, shown separately per model since both the grouping and the combined
                    rank can differ between models.
                  </p>
                  <ModelTabs models={models} active={correlationModel} onChange={setCorrelationModel} />
                  <CorrelationSection correlation={result.per_model[correlationModel].correlation} showHeading={false} />
                </>
              )}
            </>
          )}

          {tab === 'reasoning' && (
            <>
              <p className="small-muted">
                Rule documentation and retrieved NIST/ISO context are identical for every model on a given alert, so
                they're shown once per alert; each model tab below has its own{' '}
                {result.per_model[activeModel].processed[0]?.runs.length ?? repeats} run(s).
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
    <div className="tabs-row" style={{ marginTop: 18 }}>
      {models.map((m) => (
        <button key={m} className={`tab-btn ${active === m ? 'active' : ''}`} onClick={() => onChange(m)} type="button">
          {m}
        </button>
      ))}
    </div>
  )
}

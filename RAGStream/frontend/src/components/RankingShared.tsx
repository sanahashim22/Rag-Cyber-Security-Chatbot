import ReactMarkdown from 'react-markdown'
import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from 'recharts'
import type { AggregateStats, AlertTableRow, CorrelationResult, ProcessedAlert, RunDetail } from '../api/types'
import { SEVERITY_COLORS, SEVERITY_ICONS } from '../constants'
import { downloadBlob } from '../api/client'
import { Callout, Expander, MetricCard, MetricGrid, SectionLabel, RuleCard, formatDuration, severityRankColor } from './Common'

const STAGE_LABELS: Record<string, string> = {
  rag_retrieval: 'RAG retrieval (NIST/ISO chunks)',
  rule_doc_lookup: 'Rule doc lookup (sid exact match)',
  ranking_llm_call: 'Ranking LLM call',
  mismatch_llm_call: 'Mismatch justification LLM call',
  judge_llm_call: 'Judge LLM call',
  correlation_llm_call: 'Combined/correlated alert check',
  correlation_verify_llm_call: 'Combined-event self-check (per group)',
}
const STAGE_ORDER = Object.keys(STAGE_LABELS)

export function modeRank(ranks: number[]): number {
  const counts: Record<number, number> = {}
  ranks.forEach((r) => (counts[r] = (counts[r] ?? 0) + 1))
  const best = Math.max(...Object.values(counts))
  for (const r of ranks) if (counts[r] === best) return r
  return ranks[0]
}

export function AlertTable({ rows }: { rows: AlertTableRow[] }) {
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          <tr>
            <th>Alert ID</th>
            <th>Source</th>
            <th>Alert Message</th>
            <th>Actual Ranking</th>
            <th>SID</th>
            <th>Majority Rank</th>
            <th>Ranks (3 runs)</th>
            <th>Consistent</th>
            <th>Avg Judge Score</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              <td>{r.alert_id}</td>
              <td>{r.source}</td>
              <td style={{ minWidth: 240 }}>{r.alert_message}</td>
              <td>{r.actual_ranking}</td>
              <td>{r.sid}</td>
              <td style={{ background: severityRankColor(r.majority_rank) + '26', fontWeight: 700 }}>
                {r.majority_rank}
              </td>
              <td>{r.ranks.join(', ')}</td>
              <td>{r.consistent ? '✅ Yes' : '❌ No'}</td>
              <td>{r.avg_judge_score ?? '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function CorrelationSection({ correlation }: { correlation: CorrelationResult | null }) {
  if (correlation === null) return null
  const groups = correlation.groups ?? []
  return (
    <>
      <SectionLabel>Combined / correlated alert groups</SectionLabel>
      <p className="small-muted">
        Looks at ALL of this batch's alerts together (not one at a time) and lets the model suggest groups of
        alert_ids that collectively tell a more dangerous story than any one alert alone. This is entirely a 🔗
        AI-suggested judgment call based on the alerts' own content, not a code-guaranteed match.
      </p>
      {groups.length === 0 ? (
        <Callout kind="info">No correlated alert groups found in this batch.</Callout>
      ) : (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Status</th>
                <th>Alert IDs</th>
                <th>Combined Rank</th>
                <th style={{ width: '45%' }}>Combined Event</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((g, i) => (
                <tr key={i}>
                  <td>🔗 AI-suggested</td>
                  <td>{g.alert_ids.join(', ')}</td>
                  <td style={{ background: severityRankColor(g.combined_rank ?? undefined) + '26', fontWeight: 700 }}>
                    {g.combined_rank ?? '-'}
                  </td>
                  <td>{g.event_description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

export function RankingOverview({
  stats,
  alertTable,
  correlation,
  judgeModel,
  runJudge,
}: {
  stats: AggregateStats
  alertTable: AlertTableRow[]
  correlation: CorrelationResult | null
  judgeModel: string
  runJudge: boolean
}) {
  const distData = [1, 2, 3, 4, 5].map((r) => ({ rank: String(r), count: stats.rank_counts[r - 1] }))

  return (
    <>
      <MetricGrid>
        <MetricCard label="Alerts Ranked" value={stats.total_alerts} accent="blue" />
        <MetricCard label="Consistent (3/3)" value={`${stats.consistent_count}/${stats.total_alerts}`} accent="green" />
        <MetricCard label="Avg Judge Score" value={stats.avg_judge != null ? stats.avg_judge.toFixed(2) : '-'} accent="purple" />
        <MetricCard label="Judge Model" value={runJudge ? judgeModel : 'Skipped'} accent="red" />
      </MetricGrid>

      <SectionLabel>Severity rank distribution (all runs)</SectionLabel>
      <BarChart width={600} height={240} data={distData}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27354a" />
        <XAxis dataKey="rank" stroke="#8b949e" />
        <YAxis stroke="#8b949e" allowDecimals={false} />
        <Tooltip contentStyle={{ background: '#111826', border: '1px solid #27354a' }} />
        <Bar dataKey="count" fill={SEVERITY_COLORS[1]} name="# judgments" />
      </BarChart>

      <SectionLabel>Per-alert ranking</SectionLabel>
      <AlertTable rows={alertTable} />

      <CorrelationSection correlation={correlation} />
    </>
  )
}

export function AlertSharedContext({ runs }: { runs: RunDetail[] }) {
  const ruleDocBlock = runs[0]?.rule_doc_block ?? ''
  const retrieved = runs[0]?.retrieved_context ?? ''
  const chunks = retrieved
    .slice(0, 4000)
    .split('<<<CHUNK_BREAK>>>')
    .map((c) => c.trim())
    .filter(Boolean)

  return (
    <>
      {ruleDocBlock ? (
        <RuleCard title="Rule documentation used (exact sid lookup, not RAG-retrieved)" body={ruleDocBlock} accent="blue" />
      ) : (
        <p className="small-muted">
          No rule documentation entry found for this alert's sid - model relied on alert data + retrieved context
          only.
        </p>
      )}
      {chunks.length > 0 && (
        <RuleCard
          title={`NIST/ISO context retrieved (${chunks.length} chunks, same for all runs)`}
          body={chunks.map((c, i) => `(${i + 1}) ${c}`).join('\n\n')}
          accent="green"
        />
      )}
    </>
  )
}

export function RunDetailCard({ index, run }: { index: number; run: RunDetail }) {
  const rank = run.model_severity_rank
  return (
    <div className="run-card">
      <div className="run-card-head">
        {SEVERITY_ICONS[rank] ?? ''} Run {index} — {rank} · judge score {run.judge_score != null ? run.judge_score.toFixed(2) : '-'}
      </div>
      <p>{run.model_justification || '(no justification returned)'}</p>
      {run.mismatch_with_snort && run.mismatch_justification && (
        <Callout kind="warn">
          ⚠️ Disagrees with Snort priority
          {run.expected_new_rank_range ? ` (Snort implies rank ${run.expected_new_rank_range[0]}-${run.expected_new_rank_range[1]})` : ''}:{' '}
          {run.mismatch_justification}
        </Callout>
      )}
      {run.judge_reasoning && <p className="small-muted">🧑⚖️ Judge reasoning: {run.judge_reasoning}</p>}
    </div>
  )
}

export function ReasoningView({ processed }: { processed: ProcessedAlert[] }) {
  return (
    <>
      {processed.map((item, i) => {
        const alert = item.alert_entry.alert ?? {}
        const majority = modeRank(item.ranks)
        const tag = item.consistent ? 'consistent' : 'varied across runs'
        return (
          <Expander
            key={i}
            title={`${SEVERITY_ICONS[majority] ?? ''} Alert ${item.alert_entry.alert_id} — ${alert.alert_message ?? '(no message)'} · ranks [${item.ranks.join(', ')}] (${tag})`}
          >
            <AlertSharedContext runs={item.runs} />
            <SectionLabel>Run-by-run detail</SectionLabel>
            {item.runs.map((r, ri) => (
              <RunDetailCard key={ri} index={ri + 1} run={r} />
            ))}
          </Expander>
        )
      })}
    </>
  )
}

export function ReportView({ markdown, filename }: { markdown: string; filename: string }) {
  return (
    <>
      <button onClick={() => downloadBlob(filename, markdown, 'text/markdown')}>⬇ Download full report (.md)</button>
      <div style={{ marginTop: 14 }}>
        <ReactMarkdown>{markdown}</ReactMarkdown>
      </div>
    </>
  )
}

function timingStageTotals(processed: ProcessedAlert[]): Record<string, number> {
  const totals: Record<string, number> = {}
  for (const item of processed) {
    const t = item.timings
    if (t.retrieval) totals['rag_retrieval'] = (totals['rag_retrieval'] ?? 0) + t.retrieval.duration_sec
    for (const stageList of t.runs) {
      for (const s of stageList) totals[s.stage] = (totals[s.stage] ?? 0) + s.duration_sec
    }
  }
  return totals
}

function correlationTimingTotal(correlation: CorrelationResult | null): number {
  if (!correlation?.timings) return 0
  return correlation.timings.reduce((acc, t) => acc + t.duration_sec, 0)
}

function mergeCorrelationIntoTotals(totals: Record<string, number>, correlation: CorrelationResult | null) {
  for (const t of correlation?.timings ?? []) totals[t.stage] = (totals[t.stage] ?? 0) + t.duration_sec
  return totals
}

export function AlertTimingDetail({ item }: { item: ProcessedAlert }) {
  const t = item.timings
  return (
    <>
      <p className="small-muted">
        ⏱️ Alert span: {t.alert_start_ts} → {t.alert_end_ts} (total {formatDuration(t.alert_duration_sec)})
      </p>
      {t.retrieval && (
        <p className="small-muted">
          🔍 RAG retrieval: {t.retrieval.start_ts} → {t.retrieval.end_ts} ({formatDuration(t.retrieval.duration_sec)}) -
          shared by all 3 runs below
        </p>
      )}
      {t.runs.map((stageList, ri) =>
        stageList.length === 0 ? null : (
          <div key={ri}>
            <b>Run {ri + 1}</b>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Start</th>
                  <th>End</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {stageList.map((s, si) => (
                  <tr key={si}>
                    <td>{STAGE_LABELS[s.stage] ?? s.stage}</td>
                    <td>{s.start_ts}</td>
                    <td>{s.end_ts}</td>
                    <td>{formatDuration(s.duration_sec)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
      )}
    </>
  )
}

export function TimingView({
  processed,
  correlation,
  timingText,
  filename,
}: {
  processed: ProcessedAlert[]
  correlation: CorrelationResult | null
  timingText: string
  filename: string
}) {
  const alertDurations = processed.map((p) => p.timings.alert_duration_sec ?? 0)
  const corrTotal = correlationTimingTotal(correlation)
  const totalDuration = alertDurations.reduce((a, b) => a + b, 0) + corrTotal
  const avgDuration = processed.length ? alertDurations.reduce((a, b) => a + b, 0) / processed.length : 0
  const totals = mergeCorrelationIntoTotals(timingStageTotals(processed), correlation)

  const chartData = STAGE_ORDER.filter((s) => totals[s] != null).map((s) => ({
    stage: STAGE_LABELS[s],
    seconds: Math.round(totals[s] * 100) / 100,
  }))

  return (
    <>
      <MetricGrid>
        <MetricCard label="Total batch time" value={formatDuration(totalDuration)} accent="blue" />
        <MetricCard label="Avg time / alert" value={formatDuration(avgDuration)} accent="purple" />
        <MetricCard label="Alerts timed" value={processed.length} accent="green" />
        <MetricCard
          label="Combined/correlated check"
          value={correlation !== null ? formatDuration(corrTotal) : 'Skipped'}
          accent="red"
        />
      </MetricGrid>

      <SectionLabel>Time spent per stage (summed across all alerts & runs)</SectionLabel>
      <BarChart width={700} height={Math.max(160, chartData.length * 40)} data={chartData} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" stroke="#27354a" />
        <XAxis type="number" stroke="#8b949e" />
        <YAxis type="category" dataKey="stage" width={220} stroke="#8b949e" />
        <Tooltip contentStyle={{ background: '#111826', border: '1px solid #27354a' }} />
        <Bar dataKey="seconds" fill={SEVERITY_COLORS[1]} />
      </BarChart>

      <SectionLabel>Per-alert timing detail</SectionLabel>
      {processed.map((item, i) => {
        const alert = item.alert_entry.alert ?? {}
        return (
          <Expander key={i} title={`Alert ${item.alert_entry.alert_id} — ${alert.alert_message ?? ''} ⏱️ ${formatDuration(item.timings.alert_duration_sec)}`}>
            <AlertTimingDetail item={item} />
          </Expander>
        )
      })}

      <div style={{ marginTop: 10 }}>
        <button onClick={() => downloadBlob(filename, timingText, 'text/plain')}>⬇ Download full timing report (.txt)</button>
      </div>
    </>
  )
}

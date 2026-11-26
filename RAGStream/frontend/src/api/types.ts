// Shapes verified directly against the running backend (api/ranking_service.py,
// snort_ranking.py) - see RunDetail for exactly what rank_one_alert() returns.

export interface AlertEntry {
  alert_id: number | string
  alert: Record<string, any>
  [key: string]: any
}

export interface ScanResult {
  file_summaries: { name: string; count: number }[]
  warnings: string[]
  fhash: string
  total_parsed: number
  alert_inputs_dir: string
}

export interface EntriesResult {
  entries: AlertEntry[]
  warnings: string[]
  fhash: string
}

export interface RunDetail {
  model_severity_rank: number
  model_justification: string
  model_defaulted: boolean
  retrieved_context: string
  rag_relevance_score: number
  rule_doc_sid: any
  rule_doc_block: string
  rule_doc_source: string
  snort_priority: any
  expected_new_rank_range: [number, number] | null
  mismatch_with_snort: boolean
  mismatch_justification: string | null
  reported_sid: any
  sid_match: boolean
  judge_score: number | null
  judge_reasoning: string | null
  judge_model: string | null
  timings: TimingStage[]
}

export interface TimingStage {
  stage: string
  start_ts: string
  end_ts: string
  duration_sec: number
  attempts?: TimingAttempt[]
}

export interface TimingAttempt {
  round?: number
  attempt: number
  start_ts: string
  end_ts: string
  duration_sec: number
  model_load_sec?: number
  prompt_tokens?: number
  prompt_eval_sec?: number
  response_eval_sec?: number
  outcome: string
}

export interface AlertTimings {
  alert_start_ts: string
  alert_end_ts: string
  alert_duration_sec: number
  retrieval: TimingStage | null
  runs: TimingStage[][]
}

export interface ProcessedAlert {
  alert_entry: AlertEntry
  runs: RunDetail[]
  ranks: number[]
  consistent: boolean
  timings: AlertTimings
}

export interface CorrelationGroup {
  kind: string
  alert_ids: string[]
  combined_severity_rank: number | null
  event_description: string
}

export interface CorrelationResult {
  groups: CorrelationGroup[]
  timings?: TimingStage[]
  [key: string]: any
}

export interface AggregateStats {
  total_alerts: number
  n_judgments: number
  consistent_count: number
  rank_counts: number[]
  avg_judge: number | null
}

export interface AlertTableRow {
  alert_id: number | string
  source: string
  alert_message: string
  actual_ranking: any
  sid: any
  majority_rank: number
  ranks: number[]
  consistent: boolean
  avg_judge_score: number | null
}

export interface SingleRunResult {
  kind: 'single'
  model: string
  processed: ProcessedAlert[]
  correlation: CorrelationResult | null
  stats: AggregateStats
  alert_table: AlertTableRow[]
  correlation_table: ReturnType<typeof Object>[]
  report_markdown: string
  timing_text: string
}

export interface ComparisonModelRow {
  majority_rank: number
  ranks: number[]
  avg_judge_score: number | null
}

export interface ComparisonRow {
  alert_id: number | string
  source: string
  alert_message: string
  actual_ranking: any
  sid: any
  models: Record<string, ComparisonModelRow>
}

export interface ComparisonResult {
  kind: 'comparison'
  models: string[]
  per_model: Record<
    string,
    {
      processed: ProcessedAlert[]
      correlation: CorrelationResult | null
      stats: AggregateStats
      alert_table: AlertTableRow[]
      correlation_table: any[]
      report_markdown: string
      timing_text: string
    }
  >
  comparison_rows: ComparisonRow[]
  model_summary: (AggregateStats & { model: string })[]
  timing_text: string
  excel_base64: string
}

export interface JobStatus<T = SingleRunResult | ComparisonResult> {
  id: string
  status: 'running' | 'done' | 'error'
  progress: { message: string; current: number; total: number }
  result: T | null
  error: string | null
}

export interface MetaInfo {
  ollama_host: string
  ollama_is_local: boolean
  doc_names: string[]
  judge_model: string
  severity_meanings: Record<string, string>
}

export interface IntelLocalStatus {
  downloaded: boolean
  status: { bytes: number; modified: string; sources: Record<string, string> } | null
}

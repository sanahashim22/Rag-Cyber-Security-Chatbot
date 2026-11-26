"""
ranking_service.py
-------------------
Orchestration + report-building logic ported from dashboard_app.py, with
every Streamlit call (st.progress/st.empty/st.spinner/pandas DataFrame
styling) replaced by either a jobs.update_progress() call or a plain
list-of-dicts return (JSON-friendly instead of a DataFrame/Styler).

snort_ranking.py itself is reused completely unchanged - this file only
reshapes its output for JSON responses and reports job progress instead of
drawing a Streamlit progress bar.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import rag_core
import snort_ranking

RANKING_MODELS = ["llama3.2", "qwen2.5:1.5b", "phi4-mini", "gemma4:e2b"]

# Alerts beyond this count require the frontend to ask "how many do you want
# to run" instead of silently ranking everything - same threshold as
# dashboard_app.py's AUTO_RUN_LIMIT.
AUTO_RUN_LIMIT = 10


def mode_rank(ranks: List[int]) -> int:
    """Majority vote across the repeated runs; ties broken toward the first run's value."""
    counts = Counter(ranks)
    best = max(counts.values())
    for r in ranks:
        if counts[r] == best:
            return r
    return ranks[0]


def aggregate_stats(processed: List[Dict]) -> Dict:
    total_alerts = len(processed)
    all_runs = [r for item in processed for r in item["runs"]]
    rank_counts = Counter(r["model_severity_rank"] for r in all_runs)
    judge_scores = [r["judge_score"] for r in all_runs if r.get("judge_score") is not None]
    consistent_count = sum(1 for p in processed if p["consistent"])
    return {
        "total_alerts": total_alerts,
        "n_judgments": len(all_runs),
        "consistent_count": consistent_count,
        "rank_counts": [rank_counts.get(i, 0) for i in range(1, 6)],
        "avg_judge": (sum(judge_scores) / len(judge_scores)) if judge_scores else None,
    }


def build_alert_table(processed: List[Dict]) -> List[Dict]:
    rows = []
    for item in processed:
        alert = item["alert_entry"].get("alert", {}) or {}
        judge_scores = [r.get("judge_score") for r in item["runs"] if r.get("judge_score") is not None]
        avg_judge = (sum(judge_scores) / len(judge_scores)) if judge_scores else None
        rows.append(
            {
                "alert_id": item["alert_entry"].get("alert_id", "?"),
                "source": alert.get("engine", "snort").capitalize(),
                "alert_message": alert.get("alert_message", "(no message)"),
                "actual_ranking": alert.get("priority", "?"),
                "sid": alert.get("sid", "?"),
                "majority_rank": mode_rank(item["ranks"]),
                "ranks": item["ranks"],
                "consistent": item["consistent"],
                "avg_judge_score": round(avg_judge, 2) if avg_judge is not None else None,
            }
        )
    return rows


def build_correlation_table(groups: List[Dict]) -> List[Dict]:
    rows = []
    for g in groups:
        rank = g.get("combined_severity_rank")
        rows.append(
            {
                "kind": g.get("kind", "ai_suggested"),
                "alert_ids": [str(a) for a in g.get("alert_ids", [])],
                "combined_rank": rank,
                "event_description": g.get("event_description", ""),
            }
        )
    return rows


def run_ranking(job_id: str, entries: List[Dict], model: str, rag_system,
                 label_prefix: str = "", run_judge: bool = True,
                 run_mismatch: bool = True, repeats: int = 3) -> List[Dict]:
    """Runs rank_alert_three_times over every alert with `model`, reporting
    progress into the job instead of a Streamlit progress bar. `repeats` is
    user-configurable from the frontend (default 1 there) - snort_ranking.py
    itself already supported this per-call, dashboard_app.py just never
    exposed it."""
    from . import jobs

    processed = []
    total = len(entries)
    for i, alert_entry in enumerate(entries, 1):
        msg = alert_entry.get("alert", {}).get("alert_message", "")

        def _progress(run_idx, run_total, _i=i, _msg=msg):
            jobs.update_progress(
                job_id,
                f"{label_prefix}Ranking alert {_i}/{total} (run {run_idx}/{run_total}): {_msg}",
                current=_i, total=total,
            )

        result = snort_ranking.rank_alert_three_times(
            rag_core.ollama_client, model, alert_entry, rag_system, repeats=repeats,
            progress_cb=_progress, run_judge=run_judge, run_mismatch=run_mismatch,
        )
        processed.append(result)
        jobs.update_progress(job_id, f"{label_prefix}Ranked {i}/{total} alert(s)", current=i, total=total)

    return processed


def run_correlation_analysis(entries: List[Dict], processed: List[Dict], model: str,
                              run_correlation: bool, job_id: Optional[str] = None,
                              label_prefix: str = "") -> Optional[Dict]:
    """Same 'only makes sense for more than one alert' guard as
    dashboard_app.py's run_correlation_analysis(). None means 'don't run /
    don't show this section' - as opposed to an empty groups list, which
    means 'it ran and found nothing'.

    Reports progress into the job (same role dashboard_app.py's
    `st.spinner(f"Checking for combined/correlated alert groups with
    {model}...")` played) so the frontend's progress bar/message doesn't
    sit frozen on the last "Ranked N/N alert(s)" text while this - often
    multi-LLM-call - step runs."""
    if not run_correlation or len(entries) <= 1:
        return None
    if job_id is not None:
        from . import jobs

        total = len(entries)
        jobs.update_progress(
            job_id, f"{label_prefix}Checking for combined/correlated alert groups with {model}...",
            current=total, total=total,
        )
    return snort_ranking.analyze_correlated_alerts(rag_core.ollama_client, model, processed)


# ── Timing report helpers (ported from dashboard_app.py, pure text/number
#    builders - no Streamlit involved there to begin with) ─────────────────

STAGE_LABELS = {
    "rag_retrieval": "RAG retrieval (NIST/ISO chunks)",
    "rule_doc_lookup": "Rule doc lookup (sid exact match)",
    "ranking_llm_call": "Ranking LLM call",
    "mismatch_llm_call": "Mismatch justification LLM call",
    "judge_llm_call": "Judge LLM call",
    "correlation_llm_call": "Combined/correlated alert check",
    "correlation_verify_llm_call": "Combined-event self-check (per group)",
}
STAGE_ORDER = [
    "rag_retrieval", "rule_doc_lookup", "ranking_llm_call", "mismatch_llm_call",
    "judge_llm_call", "correlation_llm_call", "correlation_verify_llm_call",
]


def format_duration(seconds: Any) -> str:
    if seconds is None:
        return "-"
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if total < 60:
        return f"{total:.1f} sec"
    minutes = int(total // 60)
    secs = total - minutes * 60
    return f"{minutes} min {secs:.0f} sec"


def timing_stage_totals(processed: List[Dict]) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for item in processed:
        timings = item.get("timings") or {}
        retrieval = timings.get("retrieval")
        if retrieval:
            totals["rag_retrieval"] = totals.get("rag_retrieval", 0.0) + retrieval["duration_sec"]
        for stage_list in timings.get("runs", []):
            for stage in stage_list:
                totals[stage["stage"]] = totals.get(stage["stage"], 0.0) + stage["duration_sec"]
    return totals


def correlation_timing_total(correlation: Optional[Dict]) -> float:
    if not correlation:
        return 0.0
    return sum(t.get("duration_sec", 0.0) for t in (correlation.get("timings") or []))


def merge_correlation_into_totals(totals: Dict[str, float], correlation: Optional[Dict]) -> Dict[str, float]:
    for t in (correlation or {}).get("timings") or []:
        totals[t["stage"]] = totals.get(t["stage"], 0.0) + t["duration_sec"]
    return totals


def _append_alert_timing_text(lines: List[str], item: Dict) -> None:
    timings = item.get("timings") or {}
    alert = item["alert_entry"].get("alert", {}) or {}
    alert_id = item["alert_entry"].get("alert_id", "?")
    msg = alert.get("alert_message", "(no message)")
    lines.append(f"Alert {alert_id} - {msg}")
    lines.append(
        f"  Alert span: {timings.get('alert_start_ts', '?')} -> "
        f"{timings.get('alert_end_ts', '?')}  (total {format_duration(timings.get('alert_duration_sec'))})"
    )
    retrieval = timings.get("retrieval")
    n_runs = len(timings.get("runs", []))
    if retrieval:
        lines.append(
            f"  RAG retrieval: {retrieval['start_ts']} -> {retrieval['end_ts']} "
            f"({format_duration(retrieval['duration_sec'])}) - shared by all {n_runs} run(s) below"
        )
    for run_idx, stage_list in enumerate(timings.get("runs", []), 1):
        if not stage_list:
            continue
        lines.append(f"  Run {run_idx}:")
        for s in stage_list:
            label = STAGE_LABELS.get(s["stage"], s["stage"])
            lines.append(
                f"    {label:<32} {s['start_ts']} -> {s['end_ts']}  ({format_duration(s['duration_sec'])})"
            )
            attempts = s.get("attempts")
            if attempts:
                lines.append(f"      {len(attempts)} attempt(s):")
                for a in attempts:
                    round_part = f"round {a['round']}, " if a.get("round") else ""
                    lines.append(
                        f"        {round_part}attempt {a['attempt']}: {a['start_ts']} -> {a['end_ts']} "
                        f"({format_duration(a['duration_sec'])}) - {a['outcome']}"
                    )
    lines.append("")


def _append_correlation_timing_text(lines: List[str], correlation: Optional[Dict]) -> None:
    timings = (correlation or {}).get("timings") or []
    if not timings:
        return
    lines.append("COMBINED/CORRELATED ALERT CHECK")
    for s in timings:
        label = STAGE_LABELS.get(s["stage"], s["stage"])
        lines.append(f"  {label:<32} {s['start_ts']} -> {s['end_ts']}  ({format_duration(s['duration_sec'])})")
        attempts = s.get("attempts")
        if attempts:
            lines.append(f"    {len(attempts)} attempt(s):")
            for a in attempts:
                round_part = f"round {a['round']}, " if a.get("round") else ""
                lines.append(
                    f"      {round_part}attempt {a['attempt']}: {a['start_ts']} -> {a['end_ts']} "
                    f"({format_duration(a['duration_sec'])}) - {a['outcome']}"
                )
    lines.append("")


def build_timing_report_text(processed: List[Dict], label: str, correlation: Optional[Dict] = None) -> str:
    alert_durations = [item.get("timings", {}).get("alert_duration_sec", 0.0) for item in processed]
    corr_total = correlation_timing_total(correlation)
    total_duration = sum(alert_durations) + corr_total
    avg_duration = (sum(alert_durations) / len(processed)) if processed else 0.0
    totals = merge_correlation_into_totals(timing_stage_totals(processed), correlation)

    lines = [
        f"Timing Report - {label}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}",
        "",
        "SUMMARY",
        f"  Total batch time: {format_duration(total_duration)}",
        f"  Avg time / alert: {format_duration(avg_duration)}",
        f"  Alerts timed: {len(processed)}",
        f"  Combined/correlated check: {format_duration(corr_total)}",
        "",
        "TIME SPENT PER STAGE (summed across all alerts & runs)",
    ]
    for s in STAGE_ORDER:
        if s in totals:
            lines.append(f"  {STAGE_LABELS[s]:<32} {format_duration(totals[s])}")
    lines.append("")
    _append_correlation_timing_text(lines, correlation)
    lines.append("PER-ALERT TIMING DETAIL")
    lines.append("=" * 70)
    for item in processed:
        _append_alert_timing_text(lines, item)
        lines.append("-" * 70)
    return "\n".join(lines)


def build_comparison_timing_report_text(entries: List[Dict], results: Dict[str, List[Dict]],
                                         correlations: Optional[Dict[str, Optional[Dict]]] = None) -> str:
    correlations = correlations or {}
    lines = [
        "Timing Report - 4-Model Comparison",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}",
        "",
        "SUMMARY (all 4 models)",
    ]
    for model in RANKING_MODELS:
        processed = results[model]
        correlation = correlations.get(model)
        corr_total = correlation_timing_total(correlation)
        totals = merge_correlation_into_totals(timing_stage_totals(processed), correlation)
        total_duration = sum(item.get("timings", {}).get("alert_duration_sec", 0.0) for item in processed) + corr_total
        avg_duration = (total_duration / len(processed)) if processed else 0.0
        lines.append(f"  {model}:")
        lines.append(f"    Total time: {format_duration(total_duration)}")
        lines.append(f"    Avg time/alert: {format_duration(avg_duration)}")
        lines.append(f"    RAG retrieval: {format_duration(totals.get('rag_retrieval', 0.0))}")
        lines.append(f"    Ranking LLM: {format_duration(totals.get('ranking_llm_call', 0.0))}")
        lines.append(f"    Mismatch LLM: {format_duration(totals.get('mismatch_llm_call', 0.0))}")
        lines.append(f"    Judge LLM: {format_duration(totals.get('judge_llm_call', 0.0))}")
        lines.append(f"    Combined/correlated check: {format_duration(corr_total)}")
    lines.append("")
    lines.append("PER-ALERT TIMING DETAIL (all 4 models)")
    lines.append("=" * 70)
    for idx in range(len(entries)):
        base_item = results[RANKING_MODELS[0]][idx]
        alert = base_item["alert_entry"].get("alert", {}) or {}
        alert_id = base_item["alert_entry"].get("alert_id", "?")
        msg = alert.get("alert_message", "(no message)")
        lines.append(f"Alert {alert_id} - {msg}")
        lines.append("")
        for model in RANKING_MODELS:
            lines.append(f"  [{model}]")
            sub_lines: List[str] = []
            _append_alert_timing_text(sub_lines, results[model][idx])
            for sl in sub_lines:
                lines.append(f"  {sl}" if sl else "")
        lines.append("-" * 70)
    lines.append("")
    lines.append("COMBINED/CORRELATED ALERT CHECK (per model)")
    lines.append("=" * 70)
    for model in RANKING_MODELS:
        sub_lines: List[str] = []
        _append_correlation_timing_text(sub_lines, correlations.get(model))
        if sub_lines:
            lines.append(f"  [{model}]")
            for sl in sub_lines:
                lines.append(f"  {sl}" if sl else "")
    return "\n".join(lines)


def build_comparison_excel(entries: List[Dict], results: Dict[str, List[Dict]]) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "4-Model Comparison"

    repeats = len(results[RANKING_MODELS[0]][0]["runs"]) if entries else 1

    fixed_headers = ["Alert #", "Alert ID", "Source", "Alert Name", "Actual Ranking"]
    total_cols = len(fixed_headers) + len(RANKING_MODELS) * 2
    last_col = get_column_letter(total_cols)

    thin = Side(style="thin", color="FFB7B7B7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws.merge_cells(f"A1:{last_col}1")
    cell = ws["A1"]
    cell.value = "Snort Alert Severity Ranking: 4-Model Comparison"
    cell.font = Font(name="Arial", size=14, bold=True, color="FFFFFFFF")
    cell.fill = PatternFill("solid", fgColor="FF1F4E78")
    cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 26

    ws.merge_cells(f"A2:{last_col}2")
    cell = ws["A2"]
    cell.value = (
        f"Each alert is ranked 1-5 (1 = critical ... 5 = informational/noise) by each model, {repeats} "
        "time(s) (repeats), using the same retrieved NIST/ISO context and exact sid-matched Snort rule "
        "documentation. Judge score (0.0-1.0, independent gemma2:2b model) measures how well the "
        "model's justification is actually supported by the evidence it was given."
    )
    cell.font = Font(name="Arial", size=9, italic=True, color="FF7F6000")
    cell.fill = PatternFill("solid", fgColor="FFFFF2CC")
    cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
    ws.row_dimensions[2].height = 60

    for i, h in enumerate(fixed_headers, start=1):
        cell = ws.cell(row=3, column=i, value=h)
        cell.font = Font(name="Arial", bold=True, color="FFFFFFFF")
        cell.fill = PatternFill("solid", fgColor="FF404040")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    col = len(fixed_headers) + 1
    for model in RANKING_MODELS:
        ws.merge_cells(start_row=3, start_column=col, end_row=3, end_column=col + 1)
        cell = ws.cell(row=3, column=col, value=model)
        cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFFFF")
        cell.fill = PatternFill("solid", fgColor="FF2E75B6")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        for cc in (col, col + 1):
            ws.cell(row=3, column=cc).border = border
        col += 2
    ws.row_dimensions[3].height = 30

    for i in range(1, len(fixed_headers) + 1):
        ws.cell(row=4, column=i).fill = PatternFill("solid", fgColor="FFD9D9D9")
        ws.cell(row=4, column=i).border = border
    col = len(fixed_headers) + 1
    for _ in RANKING_MODELS:
        for label in [f"Ranks ({repeats} run(s))", "Judge Score (avg)"]:
            cell = ws.cell(row=4, column=col, value=label)
            cell.font = Font(name="Arial", size=10, bold=True)
            cell.fill = PatternFill("solid", fgColor="FFD9D9D9")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
            col += 1
    ws.row_dimensions[4].height = 26

    row = 5
    for idx in range(len(entries)):
        base_item = results[RANKING_MODELS[0]][idx]
        alert = base_item["alert_entry"].get("alert", {}) or {}

        ws.cell(row=row, column=1, value=idx + 1)
        ws.cell(row=row, column=2, value=base_item["alert_entry"].get("alert_id", "?"))
        ws.cell(row=row, column=3, value=alert.get("engine", "snort").capitalize())
        ws.cell(row=row, column=4, value=alert.get("alert_message", "(no message)"))
        ws.cell(row=row, column=5, value=alert.get("priority", "?"))
        for c in range(1, 6):
            cell = ws.cell(row=row, column=c)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(
                horizontal="left" if c == 4 else "center", vertical="top", wrap_text=(c == 4)
            )
            cell.border = border

        col = 6
        for model in RANKING_MODELS:
            item = results[model][idx]
            judge_scores = [r.get("judge_score") for r in item["runs"] if r.get("judge_score") is not None]
            avg_judge = (sum(judge_scores) / len(judge_scores)) if judge_scores else None
            ws.cell(row=row, column=col, value=", ".join(str(r) for r in item["ranks"]))
            ws.cell(row=row, column=col + 1, value=round(avg_judge, 3) if avg_judge is not None else "-")
            for cc in (col, col + 1):
                cell = ws.cell(row=row, column=cc)
                cell.font = Font(name="Arial", size=10)
                cell.alignment = Alignment(horizontal="center", vertical="top")
                cell.border = border
            col += 2
        row += 1

    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 45
    ws.column_dimensions["E"].width = 14
    for i in range(6, total_cols + 1):
        ws.column_dimensions[get_column_letter(i)].width = 16

    summary_title_row = row + 1
    ws.merge_cells(start_row=summary_title_row, start_column=1, end_row=summary_title_row, end_column=6)
    cell = ws.cell(
        row=summary_title_row, column=1,
        value="MODEL SUMMARY (consistency, judge score)",
    )
    cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFFFF")
    cell.fill = PatternFill("solid", fgColor="FF1F4E78")
    cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[summary_title_row].height = 22

    header_row = summary_title_row + 1
    headers2 = [
        "Model", "Consistent (3/3)", "Consistency %", "Avg Judge Score",
        "Rank 1-5 distribution", "",
    ]
    for i, h in enumerate(headers2, start=1):
        cell = ws.cell(row=header_row, column=i, value=h)
        cell.font = Font(name="Arial", size=10, bold=True)
        cell.fill = PatternFill("solid", fgColor="FFD9D9D9")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    ws.row_dimensions[header_row].height = 24

    best_judge = -1.0
    best_row = None
    data_start = header_row + 1
    for mi, model in enumerate(RANKING_MODELS):
        stats = aggregate_stats(results[model])
        r = data_start + mi
        consistency_pct = 100.0 * stats["consistent_count"] / stats["total_alerts"] if stats["total_alerts"] else 0.0
        values = [
            model,
            f'{stats["consistent_count"]}/{stats["total_alerts"]}',
            f"{consistency_pct:.1f}%",
            f'{stats["avg_judge"]:.3f}' if stats["avg_judge"] is not None else "-",
            "/".join(str(x) for x in stats["rank_counts"]),
            "",
        ]
        for ci, v in enumerate(values, start=1):
            cell = ws.cell(row=r, column=ci, value=v)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border
        if stats["avg_judge"] is not None and stats["avg_judge"] > best_judge:
            best_judge = stats["avg_judge"]
            best_row = r

    if best_row is not None:
        ws.cell(row=best_row, column=6, value="<-- Highest judge score")
        for ci in range(1, 7):
            cell = ws.cell(row=best_row, column=ci)
            cell.fill = PatternFill("solid", fgColor="FFC6E0B4")
            cell.font = Font(name="Arial", size=10, bold=True)

    ws.freeze_panes = "E5"
    ws.sheet_view.showGridLines = False

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()

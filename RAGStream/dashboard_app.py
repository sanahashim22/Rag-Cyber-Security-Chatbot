"""
dashboard_app.py
----------------
Dashboard frontend for Snort/Suricata alert severity ranking. Replaces the
chat-based RAGStream.py interface with a dashboard (metric cards, tabs,
styled tables) suitable for a client demo.

Backend logic (ranking pipeline, prompts, judge model, sid-exact rule-doc
lookup, NIST/ISO retrieval) is UNCHANGED - this file only reuses:
  - ConvoRAG / document loading / chunking from RAGStream.py
  - everything in snort_ranking.py (rank_alert_three_times, build_report_markdown, ...)

RAGStream.py itself is kept only as the backend module providing ConvoRAG -
run `streamlit run dashboard_app.py`, not RAGStream.py, going forward.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from io import BytesIO
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from load_documents import load_all_from_docs_folder, list_loaded_documents
import snort_ranking
from RAGStream import ADVANCED_MODE, ConvoRAG, DEFAULT_CYBERSECURITY_INFO, chunk_text_with_overlap, ollama_client

RANKING_MODELS = ["llama3.2", "qwen2.5:1.5b", "phi4-mini", "gemma4:e2b"]

# Alerts beyond this count trigger a "how many do you want to run" prompt
# instead of silently ranking everything - 3 runs/alert (and, in the
# comparison tab, x4 models) is slow on local models, same threshold the old
# chat interface used.
AUTO_RUN_LIMIT = 10

DASHBOARD_VERSION = "v1.1"

# Severity scale colors (1 = critical/red ... 5 = informational/green), used
# consistently for metric accents, table cells, and badges throughout.
SEVERITY_COLORS = {1: "#ff4b4b", 2: "#f59e0b", 3: "#d2a106", 4: "#58a6ff", 5: "#2ea043"}
SEVERITY_ICONS = {1: "\U0001f534", 2: "\U0001f7e0", 3: "\U0001f7e1", 4: "\U0001f535", 5: "\U0001f7e2"}

STAGE_LABELS = {
    "rag_retrieval": "RAG retrieval (NIST/ISO chunks)",
    "rule_doc_lookup": "Rule doc lookup (sid exact match)",
    "ranking_llm_call": "Ranking LLM call",
    "mismatch_llm_call": "Mismatch justification LLM call",
    "judge_llm_call": "Judge LLM call",
}
STAGE_ORDER = ["rag_retrieval", "rule_doc_lookup", "ranking_llm_call", "mismatch_llm_call", "judge_llm_call"]

st.set_page_config(
    page_title="Snort Alert Severity Dashboard",
    page_icon="\U0001f6e1️",
    layout="wide",
)

st.markdown(
    """
<style>
.main { background-color: #0b0e14; }
section[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }

h1, h2, h3, h4 { color: #f0f6fc; font-weight: 800; }
p, label, span { color: #c9d1d9; }
hr { border-color: #21262d; }

.hero-card {
    background: linear-gradient(135deg, #111b2e 0%, #0d1117 100%);
    border: 1px solid #27354a;
    border-radius: 18px;
    padding: 22px 26px;
    margin-bottom: 18px;
    box-shadow: 0 14px 34px rgba(0,0,0,0.30);
}
.hero-title { font-size: 28px; font-weight: 850; color: #f0f6fc; margin: 0; }
.hero-caption { color: #8b949e; font-size: 14px; margin-top: 6px; }

.version-pill {
    display: inline-block; background: linear-gradient(135deg, #238636, #2ea043);
    color: #ffffff; padding: 5px 13px; border-radius: 999px; font-size: 12.5px;
    font-weight: 750; letter-spacing: 0.3px; border: 1px solid rgba(255,255,255,0.14);
    vertical-align: middle;
}

.legend-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.legend-chip {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,0.04); border: 1px solid #27354a;
    border-radius: 999px; padding: 6px 12px; font-size: 12.5px; font-weight: 650; color: #c9d1d9;
}
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

.metric-card {
    background: linear-gradient(150deg, #121a28, #0c1016);
    padding: 18px 18px 16px 18px;
    border-radius: 16px;
    border: 1px solid #27354a;
    min-height: 104px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.22);
    transition: transform 0.12s ease, border-color 0.12s ease;
}
.metric-card:hover { transform: translateY(-2px); border-color: #3a5a86; }
.metric-label {
    color: #8b949e; font-size: 12px; font-weight: 800; letter-spacing: 0.4px;
    text-transform: uppercase; margin-bottom: 10px;
}
.metric-value { color: #ffffff; font-size: 28px; font-weight: 850; line-height: 1; letter-spacing: -0.4px; }
.metric-accent { width: 32px; height: 4px; border-radius: 999px; margin-top: 12px; background: #58a6ff; }

.section-label {
    color: #8b949e; font-size: 13px; font-weight: 800; letter-spacing: 0.6px;
    text-transform: uppercase; margin: 22px 0 10px 2px;
}

.info-card {
    background-color: #111826; border: 1px solid #27354a; border-radius: 14px;
    padding: 15px 17px; margin: 0 0 14px 0;
}
.info-card-title { color: #f0f6fc; font-size: 13.5px; font-weight: 800; margin-bottom: 6px; }
.info-card p, .info-card li { color: #9aa7b8; font-size: 13px; }

/* Native st.container(border=True) - used wherever real widgets (inputs,
   buttons) need to sit inside a styled card, since raw HTML divs can't
   wrap widgets split across separate st.* calls (each becomes its own
   sibling in the DOM, leaving an empty, unstyled "pipe" where the div
   was opened with nothing ever rendered inside it). */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #111826; border: 1px solid #27354a !important; border-radius: 14px;
}

.rule-doc-card {
    background: #10151f; border: 1px solid #27354a; border-radius: 14px;
    padding: 15px 18px; margin: 10px 0; box-shadow: 0 8px 20px rgba(0,0,0,0.16);
}
.rule-doc-card.accent-blue { border-left: 4px solid #58a6ff; }
.rule-doc-card.accent-green { border-left: 4px solid #2ea043; }
.rule-doc-card.accent-orange { border-left: 4px solid #f59e0b; }
.rule-doc-title {
    color: #f0f6fc; font-size: 12.5px; font-weight: 800; letter-spacing: 0.3px;
    text-transform: uppercase; margin-bottom: 8px;
}
.rule-doc-body { color: #c9d1d9; font-size: 14px; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
.small-muted { color: #8b949e; font-size: 13px; }

.run-card {
    background: #0f141d; border: 1px solid #222c3b; border-radius: 12px;
    padding: 12px 15px; margin-bottom: 10px;
}
.run-card-head { font-size: 13.5px; font-weight: 800; color: #f0f6fc; margin-bottom: 4px; }

div[data-testid="stDataFrame"] { border: 1px solid #27354a; border-radius: 12px; overflow: hidden; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 700;
    background-color: #111826; border: 1px solid #27354a; border-bottom: none;
}
.stTabs [aria-selected="true"] { background-color: #16233a !important; color: #ffffff !important; }

div.stButton > button {
    border-radius: 10px; font-weight: 700; border: 1px solid #2ea04355;
}
div.stDownloadButton > button {
    border-radius: 10px; font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Shared RAG system (NIST/ISO knowledge base), built once per process ─────

@st.cache_resource(show_spinner="Loading NIST/ISO knowledge base (first run only)...")
def load_rag_system():
    docs_folder_text = load_all_from_docs_folder()
    doc_names = list_loaded_documents()
    if docs_folder_text.strip():
        document_text = docs_folder_text
    else:
        document_text = DEFAULT_CYBERSECURITY_INFO
        doc_names = []

    documents = chunk_text_with_overlap(text=document_text, chunk_size=200, overlap_size=40)
    search_mode = "hybrid" if ADVANCED_MODE else "vector"
    rag_system = ConvoRAG(
        documents,
        embedding_model="nomic-embed-text",
        llm_model=RANKING_MODELS[0],
        search_mode=search_mode,
    )
    return rag_system, doc_names


# ── Small render helpers (dashboard styling) ────────────────────────────────

ACCENT_COLORS = {"blue": "#58a6ff", "orange": "#f59e0b", "green": "#2ea043", "purple": "#a371f7", "red": "#ff4b4b"}


def metric_card(label: str, value: Any, accent: str = "blue") -> None:
    color = ACCENT_COLORS.get(accent, accent)
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-accent" style="background:{color};"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rule_card(title: str, body: str, accent: str = "blue") -> None:
    if not body:
        return
    st.markdown(
        f'<div class="rule-doc-card accent-{html.escape(accent)}">'
        f'<div class="rule-doc-title">{html.escape(title)}</div>'
        f'<div class="rule-doc-body">{html.escape(body)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def mode_rank(ranks: List[int]) -> int:
    """Majority vote across the 3 runs; ties broken toward the first run's value."""
    counts = Counter(ranks)
    best = max(counts.values())
    for r in ranks:
        if counts[r] == best:
            return r
    return ranks[0]


def severity_label(rank: int) -> str:
    return f"{SEVERITY_ICONS.get(rank, '')} {rank} — {snort_ranking.SEVERITY_MEANINGS.get(rank, '-')}"


def severity_legend() -> None:
    chips = "".join(
        f'<span class="legend-chip"><span class="legend-dot" style="background:{SEVERITY_COLORS[r]};"></span>'
        f"{r} · {snort_ranking.SEVERITY_MEANINGS[r]}</span>"
        for r in range(1, 6)
    )
    st.markdown(f'<div class="legend-row">{chips}</div>', unsafe_allow_html=True)


def style_majority_rank(df: pd.DataFrame, columns: List[str]):
    """Returns a pandas Styler coloring the given 'Majority Rank' int columns
    by severity (red=critical ... green=informational)."""

    def _color(val):
        try:
            color = SEVERITY_COLORS.get(int(val))
        except (TypeError, ValueError):
            color = None
        if not color:
            return ""
        return f"background-color: {color}33; color: #f0f6fc; font-weight: 700;"

    return df.style.map(_color, subset=columns)


# ── Stats shared by both the single-model and comparison tabs ──────────────

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


def run_ranking(entries: List[Dict], model: str, rag_system, label_prefix: str = "") -> List[Dict]:
    """Runs rank_alert_three_times over every alert with `model`, showing a
    progress bar. ConvoRAG.search() emits its own st.write() debug lines per
    alert - tucked inside a collapsed expander so they don't clutter the
    dashboard, same as the old chat interface did."""
    processed = []
    total = len(entries)
    progress_bar = st.progress(0.0)
    status = st.empty()

    with st.expander(f"\U0001f50d Debug info: {label_prefix or model} retrieval", expanded=False):
        for i, alert_entry in enumerate(entries, 1):
            msg = alert_entry.get("alert", {}).get("alert_message", "")

            def _progress(run_idx, run_total, _i=i, _msg=msg):
                status.markdown(f"{label_prefix}Ranking alert {_i}/{total} (run {run_idx}/{run_total}): {_msg}")

            result = snort_ranking.rank_alert_three_times(
                ollama_client, model, alert_entry, rag_system, repeats=3, progress_cb=_progress,
            )
            processed.append(result)
            progress_bar.progress(i / total)

    progress_bar.empty()
    status.empty()
    return processed


def build_alert_table(processed: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in processed:
        alert = item["alert_entry"].get("alert", {}) or {}
        judge_scores = [r.get("judge_score") for r in item["runs"] if r.get("judge_score") is not None]
        avg_judge = (sum(judge_scores) / len(judge_scores)) if judge_scores else None
        rows.append(
            {
                "Alert ID": item["alert_entry"].get("alert_id", "?"),
                "Alert Message": alert.get("alert_message", "(no message)"),
                "Snort Priority": alert.get("priority", "?"),
                "SID": alert.get("sid", "?"),
                "Majority Rank": mode_rank(item["ranks"]),
                "Ranks (3 runs)": ", ".join(str(r) for r in item["ranks"]),
                "Consistent": "✅ Yes" if item["consistent"] else "❌ No",
                "Avg Judge Score": f"{avg_judge:.2f}" if avg_judge is not None else "-",
            }
        )
    return pd.DataFrame(rows)


# ── Tab 1 sub-views: Ranking / Reasoning / Report & Material ────────────────

def render_ranking_overview(processed: List[Dict], model: str) -> None:
    stats = aggregate_stats(processed)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Alerts Ranked", stats["total_alerts"], "blue")
    with c2:
        metric_card("Consistent (3/3)", f'{stats["consistent_count"]}/{stats["total_alerts"]}', "green")
    with c3:
        metric_card("Avg Judge Score", f'{stats["avg_judge"]:.2f}' if stats["avg_judge"] is not None else "-", "purple")
    with c4:
        metric_card("Judge Model", snort_ranking.JUDGE_MODEL, "red")

    st.markdown('<div class="section-label">Severity scale</div>', unsafe_allow_html=True)
    severity_legend()

    st.markdown('<div class="section-label">Severity rank distribution (all runs)</div>', unsafe_allow_html=True)
    dist_df = pd.DataFrame(
        {
            "Rank": [f"{i}" for i in range(1, 6)],
            "# judgments": stats["rank_counts"],
        }
    ).set_index("Rank")
    st.bar_chart(dist_df, color=SEVERITY_COLORS[1])

    st.markdown('<div class="section-label">Per-alert ranking</div>', unsafe_allow_html=True)
    table = build_alert_table(processed)
    st.dataframe(style_majority_rank(table, ["Majority Rank"]), width='stretch', hide_index=True)


def render_run_detail(i: int, r: Dict) -> None:
    """Renders ONE run's full detail: rank, judge score, the model's own
    justification, mismatch-vs-Snort justification (if any), and the judge's
    reasoning (if any) - the actual content the dashboard's "reasoning" is
    supposed to show, for that specific run."""
    rank = r["model_severity_rank"]
    judge = r.get("judge_score")
    judge_str = f"{judge:.2f}" if judge is not None else "-"
    st.markdown(
        f"""
        <div class="run-card">
            <div class="run-card-head">{SEVERITY_ICONS.get(rank, '')} Run {i} — {html.escape(severity_label(rank))} · judge score {judge_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(r.get("model_justification", "") or "_(no justification returned)_")
    if r.get("judge_reasoning"):
        st.caption(f"\U0001f9d1⚖️ Judge reasoning: {r.get('judge_reasoning')}")


def render_shared_alert_context(runs: List[Dict]) -> None:
    """Rule doc + retrieved NIST/ISO context: deterministic per ALERT (exact
    sid lookup + fixed retrieval query), identical across every run and -
    in the comparison tab - every model, so it's shown once rather than
    repeated underneath each run/model."""
    rule_doc_block = runs[0].get("rule_doc_block") or ""
    if rule_doc_block:
        rule_card("Rule documentation used (exact sid lookup, not RAG-retrieved)", rule_doc_block, "blue")
    else:
        st.caption("No rule documentation entry found for this alert's sid - model relied on alert data + retrieved context only.")

    retrieved = runs[0].get("retrieved_context") or ""
    chunks = [c.strip() for c in retrieved[:4000].split("<<<CHUNK_BREAK>>>") if c.strip()]
    if chunks:
        chunk_body = "\n\n".join(f"({n}) {chunk}" for n, chunk in enumerate(chunks, 1))
        rule_card(f"NIST/ISO context retrieved ({len(chunks)} chunks, same for all runs)", chunk_body, "green")


def render_reasoning_tab(processed: List[Dict]) -> None:
    for item in processed:
        alert = item["alert_entry"].get("alert", {}) or {}
        alert_id = item["alert_entry"].get("alert_id", "?")
        msg = alert.get("alert_message", "(no message)")
        runs = item["runs"]
        majority = mode_rank(item["ranks"])
        consistency_tag = "consistent" if item["consistent"] else "varied across runs"

        label = (
            f"{SEVERITY_ICONS.get(majority, '')} Alert {alert_id} — {msg}  "
            f"· ranks {item['ranks']} ({consistency_tag})"
        )
        with st.expander(label, expanded=False):
            render_shared_alert_context(runs)

            st.markdown('<div class="section-label">Run-by-run detail</div>', unsafe_allow_html=True)
            for i, r in enumerate(runs, 1):
                render_run_detail(i, r)


def render_report_tab(processed: List[Dict], model: str) -> None:
    report_text = snort_ranking.build_report_markdown(processed, model)
    st.download_button(
        "⬇ Download full report (.md)",
        data=report_text,
        file_name=f"snort_ranking_report_{model.replace(':', '_')}.md",
        mime="text/markdown",
    )
    st.markdown(report_text)


def format_duration(seconds: Any) -> str:
    """Human-readable duration - '4.2 sec' under a minute, '3 min 5 sec'
    at/above a minute, instead of a raw second count that's hard to parse
    at a glance for longer batch runs."""
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
    """Sums each stage's duration_sec across every alert and every one of its
    3 runs, so it's visible which stage (retrieval, ranking call, mismatch
    call, judge call) is actually eating the time, not just the alert's
    overall duration."""
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


def render_alert_timing_detail(item: Dict) -> None:
    """Start/end/duration for one alert's retrieval + every run's stages -
    the per-alert drill-down behind the Timing tab's summary numbers."""
    timings = item.get("timings") or {}
    st.caption(
        f"⏱️ Alert span: {timings.get('alert_start_ts', '?')} → "
        f"{timings.get('alert_end_ts', '?')}  (total {format_duration(timings.get('alert_duration_sec'))})"
    )
    retrieval = timings.get("retrieval")
    if retrieval:
        st.caption(
            f"\U0001f50d RAG retrieval: {retrieval['start_ts']} → {retrieval['end_ts']} "
            f"({format_duration(retrieval['duration_sec'])}) - shared by all 3 runs below"
        )
    for run_idx, stage_list in enumerate(timings.get("runs", []), 1):
        if not stage_list:
            continue
        rows = [
            {
                "Stage": STAGE_LABELS.get(s["stage"], s["stage"]),
                "Start": s["start_ts"],
                "End": s["end_ts"],
                "Duration": format_duration(s["duration_sec"]),
            }
            for s in stage_list
        ]
        st.markdown(f"**Run {run_idx}**")
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        # Not every stage's call_ollama_json succeeds on its first try - a
        # bad/looping response or a transient error triggers a retry (see
        # call_ollama_json's attempt_log). Surface every one of those
        # attempts here, per stage, so a long "ranking_llm_call" duration is
        # explained by actual retries instead of looking like one slow call.
        for s in stage_list:
            attempts = s.get("attempts")
            if not attempts:
                continue
            st.caption(f"↳ {STAGE_LABELS.get(s['stage'], s['stage'])} — {len(attempts)} attempt(s)")
            attempt_rows = [
                {
                    "Round": a.get("round", "-"),
                    "Attempt": a["attempt"],
                    "Start": a["start_ts"],
                    "End": a["end_ts"],
                    "Duration": format_duration(a["duration_sec"]),
                    "Model load": format_duration(a["model_load_sec"]) if a.get("model_load_sec") else "-",
                    "Prompt tokens": a.get("prompt_tokens", "-"),
                    "Reading prompt (prefill)": format_duration(a["prompt_eval_sec"]) if a.get("prompt_eval_sec") is not None else "-",
                    "Generating answer (decode)": format_duration(a["response_eval_sec"]) if a.get("response_eval_sec") is not None else "-",
                    "Outcome": a["outcome"],
                }
                for a in attempts
            ]
            attempt_df = pd.DataFrame(attempt_rows)
            if all(r["Round"] == "-" for r in attempt_rows):
                attempt_df = attempt_df.drop(columns=["Round"])
            st.dataframe(attempt_df, width='stretch', hide_index=True)
            st.caption(
                "\"Reading prompt\" = time Ollama spent processing the input (prefill); "
                "\"Generating answer\" = time spent producing the response (decode) - straight "
                "from Ollama's own response metadata, not estimated. \"Model load\" only shows up "
                "when Ollama had to load the model fresh (usually just the very first call)."
            )


def render_timing_tab(processed: List[Dict]) -> None:
    alert_durations = [item.get("timings", {}).get("alert_duration_sec", 0.0) for item in processed]
    total_duration = sum(alert_durations)
    avg_duration = (total_duration / len(processed)) if processed else 0.0
    totals = timing_stage_totals(processed)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Total batch time", format_duration(total_duration), "blue")
    with c2:
        metric_card("Avg time / alert", format_duration(avg_duration), "purple")
    with c3:
        metric_card("Alerts timed", len(processed), "green")

    st.markdown('<div class="section-label">Time spent per stage (summed across all alerts & runs)</div>', unsafe_allow_html=True)
    stage_names = [STAGE_LABELS[s] for s in STAGE_ORDER if s in totals]
    stage_seconds = [round(totals[s], 2) for s in STAGE_ORDER if s in totals]
    chart_df = pd.DataFrame({"Stage": stage_names, "Total seconds": stage_seconds}).set_index("Stage")
    st.bar_chart(chart_df, horizontal=True)
    st.dataframe(
        pd.DataFrame({"Stage": stage_names, "Total time": [format_duration(s) for s in stage_seconds]}),
        width='stretch', hide_index=True,
    )
    st.caption(
        "Ranking LLM call usually dominates - mismatch calls only run for alerts whose rank fell "
        "outside Snort's expected range, so that bar can be 0 or small depending on the alert mix."
    )

    st.markdown('<div class="section-label">Per-alert timing detail</div>', unsafe_allow_html=True)
    for item in processed:
        alert = item["alert_entry"].get("alert", {}) or {}
        alert_id = item["alert_entry"].get("alert_id", "?")
        msg = alert.get("alert_message", "(no message)")
        dur = item.get("timings", {}).get("alert_duration_sec")
        with st.expander(f"Alert {alert_id} — {msg}  ⏱️ {format_duration(dur)}", expanded=False):
            render_alert_timing_detail(item)


def render_comparison_timing_tab(entries: List[Dict], results: Dict[str, List[Dict]]) -> None:
    summary_rows = []
    for model in RANKING_MODELS:
        processed = results[model]
        totals = timing_stage_totals(processed)
        total_duration = sum(item.get("timings", {}).get("alert_duration_sec", 0.0) for item in processed)
        avg_duration = (total_duration / len(processed)) if processed else 0.0
        summary_rows.append(
            {
                "Model": model,
                "total": total_duration,
                "avg": avg_duration,
                "retrieval": totals.get("rag_retrieval", 0.0),
                "ranking": totals.get("ranking_llm_call", 0.0),
                "mismatch": totals.get("mismatch_llm_call", 0.0),
                "judge": totals.get("judge_llm_call", 0.0),
            }
        )
    raw_df = pd.DataFrame(summary_rows)

    st.markdown('<div class="section-label">Timing summary (all 4 models)</div>', unsafe_allow_html=True)
    display_df = pd.DataFrame(
        {
            "Model": raw_df["Model"],
            "Total time": raw_df["total"].apply(format_duration),
            "Avg time/alert": raw_df["avg"].apply(format_duration),
            "RAG retrieval": raw_df["retrieval"].apply(format_duration),
            "Ranking LLM": raw_df["ranking"].apply(format_duration),
            "Mismatch LLM": raw_df["mismatch"].apply(format_duration),
            "Judge LLM": raw_df["judge"].apply(format_duration),
        }
    )
    st.dataframe(display_df, width='stretch', hide_index=True)

    chart_df = raw_df.set_index("Model")[["retrieval", "ranking", "mismatch", "judge"]].rename(
        columns={
            "retrieval": "RAG retrieval",
            "ranking": "Ranking LLM",
            "mismatch": "Mismatch LLM",
            "judge": "Judge LLM",
        }
    )
    st.bar_chart(chart_df, horizontal=True)
    st.caption("Stacked by stage per model - the longest bar is whichever model/stage combo is the current bottleneck.")

    st.markdown('<div class="section-label">Per-alert timing detail (all 4 models)</div>', unsafe_allow_html=True)
    for idx in range(len(entries)):
        base_item = results[RANKING_MODELS[0]][idx]
        alert = base_item["alert_entry"].get("alert", {}) or {}
        alert_id = base_item["alert_entry"].get("alert_id", "?")
        msg = alert.get("alert_message", "(no message)")

        with st.expander(f"Alert {alert_id} — {msg}", expanded=False):
            model_tabs = st.tabs([f"{model}" for model in RANKING_MODELS])
            for tab, model in zip(model_tabs, RANKING_MODELS):
                with tab:
                    render_alert_timing_detail(results[model][idx])


def render_single_model_tab(entries: List[Dict], rag_system, fhash: str) -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        model = st.selectbox("Ranking model", RANKING_MODELS, key="single_model_select")
    with col2:
        st.write("")
        run_clicked = st.button("▶ Run ranking", key="run_single_btn", width='stretch')

    state_key = f"single::{fhash}::{model}"
    if run_clicked:
        with st.spinner(f"Ranking {len(entries)} alert(s) with {model} (3 runs each)..."):
            st.session_state[state_key] = run_ranking(entries, model, rag_system)

    processed = st.session_state.get(state_key)
    if processed is None:
        st.info("Select a model and click **Run ranking** to rank these alerts.")
        return

    sub_rank, sub_reason, sub_report, sub_timing = st.tabs(
        ["\U0001f4cb Ranking", "\U0001f9e0 Reasoning", "\U0001f4c4 Report & Material", "⏱️ Timing"]
    )
    with sub_rank:
        render_ranking_overview(processed, model)
    with sub_reason:
        render_reasoning_tab(processed)
    with sub_report:
        render_report_tab(processed, model)
    with sub_timing:
        render_timing_tab(processed)


# ── Tab 2: 4-model side-by-side comparison ──────────────────────────────────

def build_comparison_excel(entries: List[Dict], results: Dict[str, List[Dict]]) -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "4-Model Comparison"

    fixed_headers = ["Alert #", "Alert ID", "Alert Name", "Snort Priority"]
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
        "Each alert is ranked 1-5 (1 = critical ... 5 = informational/noise) by each model, 3 times "
        "(repeats), using the same retrieved NIST/ISO context and exact sid-matched Snort rule "
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
        for label in ["Ranks (3 runs)", "Judge Score (avg)"]:
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
        ws.cell(row=row, column=3, value=alert.get("alert_message", "(no message)"))
        ws.cell(row=row, column=4, value=alert.get("priority", "?"))
        for c in range(1, 5):
            cell = ws.cell(row=row, column=c)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(
                horizontal="left" if c == 3 else "center", vertical="top", wrap_text=(c == 3)
            )
            cell.border = border

        col = 5
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
    ws.column_dimensions["C"].width = 45
    ws.column_dimensions["D"].width = 12
    for i in range(5, total_cols + 1):
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


def render_comparison_tab(entries: List[Dict], rag_system, fhash: str) -> None:
    run_clicked = st.button("▶ Run all 4 models", key="run_compare_btn")
    state_key = f"compare::{fhash}"

    if run_clicked:
        results: Dict[str, List[Dict]] = {}
        overall_status = st.empty()
        for mi, model in enumerate(RANKING_MODELS, 1):
            overall_status.markdown(f"**Running model {mi}/{len(RANKING_MODELS)}: `{model}`**")
            with st.spinner(f"Ranking {len(entries)} alert(s) with {model} (3 runs each)..."):
                results[model] = run_ranking(entries, model, rag_system, label_prefix=f"[{model}] ")
        overall_status.empty()
        st.session_state[state_key] = results

    results = st.session_state.get(state_key)
    if results is None:
        st.info(
            "Click **Run all 4 models** to rank every alert with all 4 models "
            "(3 runs each) and compare them side by side. This runs 12 full passes "
            "per alert, so it takes roughly 4x longer than the single-model tab."
        )
        return

    st.markdown('<div class="section-label">Severity scale</div>', unsafe_allow_html=True)
    severity_legend()

    st.markdown('<div class="section-label">Model summary</div>', unsafe_allow_html=True)
    summary_rows = []
    for model in RANKING_MODELS:
        stats = aggregate_stats(results[model])
        summary_rows.append(
            {
                "Model": model,
                "Consistency (3/3)": f'{stats["consistent_count"]}/{stats["total_alerts"]}',
                "Avg Judge Score": f'{stats["avg_judge"]:.3f}' if stats["avg_judge"] is not None else "-",
                "Rank 1": stats["rank_counts"][0],
                "Rank 2": stats["rank_counts"][1],
                "Rank 3": stats["rank_counts"][2],
                "Rank 4": stats["rank_counts"][3],
                "Rank 5": stats["rank_counts"][4],
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), width='stretch', hide_index=True)
    st.caption(
        "These are pipeline-measured stats only (consistency across 3 runs, independent judge "
        "score) - there's no external \"correct answer\" baked in, so no model is labeled "
        "best/worst here."
    )

    st.markdown('<div class="section-label">Per-alert comparison</div>', unsafe_allow_html=True)
    rows = []
    rank_cols = [f"{model} — rank" for model in RANKING_MODELS]
    for idx in range(len(entries)):
        base_item = results[RANKING_MODELS[0]][idx]
        alert = base_item["alert_entry"].get("alert", {}) or {}
        row = {
            "Alert ID": base_item["alert_entry"].get("alert_id", "?"),
            "Alert Message": alert.get("alert_message", "(no message)"),
            "Snort Priority": alert.get("priority", "?"),
            "SID": alert.get("sid", "?"),
        }
        for model in RANKING_MODELS:
            item = results[model][idx]
            judge_scores = [r.get("judge_score") for r in item["runs"] if r.get("judge_score") is not None]
            avg_judge = (sum(judge_scores) / len(judge_scores)) if judge_scores else None
            rank_col = f"{model} — rank"
            row[rank_col] = mode_rank(item["ranks"])
            row[f"{model} — runs"] = ", ".join(str(r) for r in item["ranks"])
            row[f"{model} — judge"] = f"{avg_judge:.2f}" if avg_judge is not None else "-"
        rows.append(row)
    compare_df = pd.DataFrame(rows)
    st.dataframe(style_majority_rank(compare_df, rank_cols), width='stretch', hide_index=True)
    st.caption("Each model's colored column is its majority rank across 3 runs; \"runs\" shows the raw 3 values and \"judge\" the average judge score.")

    excel_bytes = build_comparison_excel(entries, results)
    st.download_button(
        "⬇ Download comparison workbook (.xlsx)",
        data=excel_bytes,
        file_name="model_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown('<div class="section-label">Per-alert reasoning (all 3 runs, all 4 models)</div>', unsafe_allow_html=True)
    st.caption(
        "Rule documentation and retrieved NIST/ISO context are identical for every model on a given "
        "alert (sid-exact lookup + a fixed retrieval query), so they're shown once per alert; each "
        "model's tab below has its own 3 runs with that model's own justification and judge reasoning."
    )
    for idx in range(len(entries)):
        base_item = results[RANKING_MODELS[0]][idx]
        alert = base_item["alert_entry"].get("alert", {}) or {}
        alert_id = base_item["alert_entry"].get("alert_id", "?")
        msg = alert.get("alert_message", "(no message)")

        with st.expander(f"Alert {alert_id} — {msg}", expanded=False):
            render_shared_alert_context(base_item["runs"])

            model_tabs = st.tabs([f"{model}" for model in RANKING_MODELS])
            for tab, model in zip(model_tabs, RANKING_MODELS):
                with tab:
                    item = results[model][idx]
                    consistency_tag = "consistent" if item["consistent"] else "varied across runs"
                    st.caption(f"Ranks across 3 runs: {item['ranks']} ({consistency_tag})")
                    for i, r in enumerate(item["runs"], 1):
                        render_run_detail(i, r)

    st.markdown('<div class="section-label">⏱️ Timing (all 4 models)</div>', unsafe_allow_html=True)
    render_comparison_timing_tab(entries, results)


# ── Page ─────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="hero-card">
        <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
            <div class="hero-title">\U0001f6e1️ Snort Alert Severity Ranking Dashboard</div>
            <span class="version-pill">{DASHBOARD_VERSION}</span>
        </div>
        <div class="hero-caption">
            AI-assisted Snort/Suricata alert severity ranking, grounded in NIST CSF 2.0, ISO/IEC 27001:2022,
            and each alert's own rule documentation.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ConvoRAG's setup (document chunking, embedding, ChromaDB/BM25 indexing)
# writes a lot of raw st.write()/progress-bar debug lines - st.cache_resource
# means that only actually happens on the very first load (cached afterwards),
# but when it does, tuck it inside one tidy collapsible status box instead of
# letting it spill out as plain loose text at the top of the page.
with st.status("Loading NIST/ISO knowledge base...", expanded=False) as kb_status:
    rag_system, doc_names = load_rag_system()
    kb_status.update(label="✅ NIST/ISO knowledge base ready", state="complete", expanded=False)

with st.sidebar:
    doc_list_html = "".join(f"<li>{html.escape(name)}</li>" for name in doc_names) or "<li>Built-in NIST CSF 2.0 summary</li>"
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">\U0001f4da Loaded standards</div>
            <ul style="margin:0; padding-left:18px;">{doc_list_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">ℹ️ How this works</div>
            <p style="margin:0;">Each alert is ranked 1-5 for severity, 3 times per model, citing its own
            Snort rule documentation (exact sid lookup) plus retrieved NIST/ISO context. An independent
            judge model (<code>{html.escape(snort_ranking.JUDGE_MODEL)}</code>) scores how well each
            justification is actually supported by the evidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown('<div class="info-card-title">\U0001f3af Severity scale</div>', unsafe_allow_html=True)
        severity_legend()

uploaded_file = st.file_uploader("Upload Snort/Suricata alerts (JSON)", type=["json"])

if uploaded_file is None:
    st.info("Upload a JSON alerts file to begin - a single alert object, a list of alerts, or an `{\"alerts\": [...]}` export are all accepted.")
    st.stop()

raw_bytes = uploaded_file.getvalue()
fhash = hashlib.md5(raw_bytes).hexdigest()

try:
    data = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
except Exception as e:
    st.error(f"Could not parse `{uploaded_file.name}` as JSON: {e}")
    st.stop()

if not snort_ranking.is_snort_alerts_payload(data):
    st.error(
        f"`{uploaded_file.name}` doesn't look like a Snort alerts file (expected fields like "
        "`sid`, `priority`, or an `alerts` list). Please upload a Snort/Suricata alerts JSON export."
    )
    st.stop()

entries = snort_ranking.extract_alert_entries(data)
if not entries:
    st.error(f"`{uploaded_file.name}` was recognized as a Snort alerts file but contained no alerts.")
    st.stop()

total_parsed = len(entries)
st.success(f"Parsed {total_parsed} alert(s) from `{uploaded_file.name}`.")

if total_parsed > AUTO_RUN_LIMIT:
    confirm_key = f"confirmed_count::{fhash}"
    confirmed_n = st.session_state.get(confirm_key)
    if confirmed_n is None:
        with st.container(border=True):
            st.markdown(f'<div class="info-card-title">⚠️ {total_parsed} alerts found</div>', unsafe_allow_html=True)
            st.write(
                f"This file has **{total_parsed} alerts**. Ranking all of them (3 runs per alert, "
                "x4 models in the 4-Model Comparison tab) can take a while on local models. "
                "Choose how many to process below."
            )
            # Wrapped in a form so pressing Enter after typing a number submits
            # immediately (acts like clicking "Process entered number") instead
            # of just committing the number_input's value with no action taken.
            with st.form(key=f"confirm_form::{fhash}"):
                n = st.number_input(
                    "Number of alerts to process",
                    min_value=1, max_value=total_parsed,
                    value=min(AUTO_RUN_LIMIT, total_parsed), step=1,
                    key=f"confirm_n_input::{fhash}",
                )
                st.caption(f"Press Enter, or click below, to process the first **{int(n)}** of {total_parsed} alert(s).")
                cc1, cc2 = st.columns(2)
                with cc1:
                    process_n = st.form_submit_button("▶ Process entered number", width='stretch')
                with cc2:
                    process_all = st.form_submit_button(f"Process all {total_parsed}", width='stretch')

            if process_n:
                st.session_state[confirm_key] = int(n)
                st.rerun()
            if process_all:
                st.session_state[confirm_key] = total_parsed
                st.rerun()
        st.stop()

    entries = entries[:confirmed_n]
    if confirmed_n < total_parsed:
        st.caption(f"Processing the first {confirmed_n} of {total_parsed} alerts (as selected above).")

tab_single, tab_compare = st.tabs(["\U0001f50e Single-Model Ranking", "\U0001f4ca 4-Model Comparison"])
with tab_single:
    render_single_model_tab(entries, rag_system, fhash)
with tab_compare:
    render_comparison_tab(entries, rag_system, fhash)

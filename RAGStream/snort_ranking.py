"""
snort_ranking.py
-----------------
Snort alert severity ranking, ported from
3-runs-of-snort-alerts/rank_alerts_rag.py, but using the app's own local
Ollama models (via the existing ConvoRAG instance for retrieval and the
ollama.Client already used for chat) instead of Groq's API.

For each alert:
  1. Retrieve relevant chunks from the RAG knowledge base (NIST CSF, ISO
     27001) via the app's ConvoRAG.search().
  2. Look up the alert's own rule directly by sid in
     rule_docs_preprocessed_by_sid.json (exact match, not through RAG
     retrieval - see get_rule_doc_block).
  3. Ask the ranking LLM for a severity_rank (1=critical ... 5=noise) with a
     justification citing the rule documentation / retrieved evidence.
  4. If the rank disagrees with Snort's own priority-implied range, ask the
     SAME ranking model for a dedicated mismatch_justification.
  5. Ask a separate, fixed judge model to score (0.0-1.0) how well the
     justification is actually supported by the evidence.

Each alert is run through this pipeline 3 times (see rank_alert_three_times)
so the caller can show whether the model is consistent across repeats.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from load_documents import _rule_obj_to_text


# ── Timing instrumentation (see render_timing_tab in dashboard_app.py) ──────
# Every stage of the pipeline (RAG retrieval, the ranking LLM call, the
# mismatch-justification call, the judge call) gets wall-clock start/end
# timestamps + duration recorded, so it's visible from the dashboard exactly
# where the time for a run actually goes - without this, "it's slow" gives no
# way to tell whether retrieval, a specific model, or the judge is the
# bottleneck.

def _now() -> datetime:
    return datetime.now()


def _fmt_ts(dt: datetime) -> str:
    """12-hour clock with AM/PM, e.g. '02:15:30.123 PM'."""
    return dt.strftime("%I:%M:%S.%f")[:-3] + dt.strftime(" %p")


def _ollama_breakdown(response) -> Dict:
    """Ollama's own chat() response already reports exactly how its total
    time split between loading the model, reading/prefilling the prompt, and
    generating the answer (all in nanoseconds) - pull those out instead of
    only knowing the call's overall wall-clock duration."""
    def _ns_to_sec(ns):
        return round(ns / 1e9, 3) if ns else None

    return {
        "model_load_sec": _ns_to_sec(response.get("load_duration")),
        "prompt_tokens": response.get("prompt_eval_count"),
        "prompt_eval_sec": _ns_to_sec(response.get("prompt_eval_duration")),
        "response_tokens": response.get("eval_count"),
        "response_eval_sec": _ns_to_sec(response.get("eval_duration")),
    }


def _stage_timing(stage: str, start: datetime, end: datetime,
                   attempts: List[Dict] | None = None) -> Dict:
    d = {
        "stage": stage,
        "start_ts": _fmt_ts(start),
        "end_ts": _fmt_ts(end),
        "duration_sec": round((end - start).total_seconds(), 3),
    }
    if attempts:
        d["attempts"] = attempts
    return d

# Kept OUTSIDE docs/ on purpose (see RAGStream chat history) - this file is
# never chunked/embedded/vector-searched. It's used purely as a plain
# sid -> rule dict lookup table, so a given alert's OWN rule always gets
# matched exactly (no semantic-similarity guessing, no competing against
# NIST/ISO in embedding space the way it used to when it lived in docs/).
RULE_DOCS_TABLE_PATH = Path(__file__).parent / "rule_docs_preprocessed_by_sid.json"

_rule_docs_table_cache: Dict[str, Dict] | None = None


def _load_rule_docs_table() -> Dict[str, Dict]:
    """Loads rule_docs_preprocessed_by_sid.json once and caches it in memory
    as a sid(str) -> rule dict. Missing/unreadable file just yields an empty
    table (no error) - callers then simply find no match for any sid, which
    is the same "skip it, don't error" behavior as a single unmatched sid."""
    global _rule_docs_table_cache
    if _rule_docs_table_cache is None:
        try:
            with open(RULE_DOCS_TABLE_PATH, "r", encoding="utf-8") as f:
                _rule_docs_table_cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            _rule_docs_table_cache = {}
    return _rule_docs_table_cache


def resolve_ground_truth_sid(alert_entry: Dict) -> Any:
    """The alert's own sid, read from wherever it lives (alert.sid, or
    rule_documentation.sid for an export shaped that way)."""
    alert = alert_entry.get("alert", {}) or {}
    rule_doc = alert_entry.get("rule_documentation", {}) or {}
    return alert.get("sid") if alert.get("sid") is not None else rule_doc.get("sid")


def get_rule_doc_block(sid: Any) -> str:
    """Exact-match lookup of `sid` in rule_docs_preprocessed_by_sid.json.
    Returns the rule formatted as readable text (same formatter used when
    this file WAS embedded, for consistent wording), or "" if this sid isn't
    in the table - a future/unknown sid is simply skipped, never an error."""
    if sid is None:
        return ""
    table = _load_rule_docs_table()
    rule = table.get(str(sid))
    if not rule:
        return ""
    return _rule_obj_to_text(rule)

# Independent judge model, always used regardless of the model selected in
# the sidebar for chat/ranking, so it acts as a check rather than the same
# model grading its own homework. qwen2.5:1.5b was tried first but tends to
# default to a flat 0.5 regardless of input (too small to differentiate
# fine-grained justification quality) - gemma2:2b gives more varied scores.
JUDGE_MODEL = "gemma2:2b"

# Temperature for the ranking model's calls (severity rank + mismatch
# justification) when running the 3 repeats. Set to 0 (fully deterministic/
# greedy) to match project 2's (rank_alerts_rag.py) behavior exactly - it
# also hardcoded "temperature": 0 for every Groq call. At 0, repeats on the
# SAME alert will almost always agree here, since local Ollama inference
# doesn't have the cloud-batching nondeterminism that made Groq's temp-0
# calls vary run to run despite the same setting. Raise this (e.g. 0.5) if
# genuine run-to-run disagreement is wanted to make the consistency check
# meaningful again. The judge call stays at temperature 0 regardless (see
# call_ollama_json's default) since its job is to grade a given
# justification consistently, not to be re-sampled.
RANKING_TEMPERATURE = 0

# Snort priority (1-3) -> allowed new-scale ranks (1=critical ... 5=noise)
EXPECTED_RANGE = {
    1: (1, 2),
    2: (3, 3),
    3: (4, 5),
}

SEVERITY_MEANINGS = {
    1: "Critical / clearly dangerous - must be reviewed immediately",
    2: "Likely a real attack or vulnerability - worth review",
    3: "Uncertain / moderate - could go either way",
    4: "Low severity - probably not worth analyst time",
    5: "Clearly informational / noise - not a real threat",
}

SYSTEM_PROMPT = """You are a SOC (Security Operations Center) analyst reviewing ONE \
single Snort/Suricata IDS alert occurrence at a time.

You will be given:
  - The full raw data for exactly this one alert.
  - That alert's own Snort rule documentation, looked up directly by sid \
(an exact match, not a retrieval guess) - given as its own labeled section.
  - Relevant excerpts retrieved from cybersecurity standards documents (NIST \
CSF 2.0, ISO/IEC 27001, and any other reference documents placed in docs/).

Judge THIS ONE alert using ALL of the above and your own security expertise.

IMPORTANT - how to weigh the two kinds of evidence:
  - The retrieved NIST CSF / ISO 27001 excerpts are generic organizational \
policy/control framework text. They describe broad control categories (access \
control, monitoring, vulnerability management, etc.) and will almost NEVER \
name a specific exploit, CVE, or attack technique. The fact that a retrieved \
excerpt does not explicitly mention this alert's specific vulnerability is \
NORMAL and is NOT evidence that the alert is low severity - do not downgrade \
severity just because the standards text doesn't literally describe this \
exact scenario.
  - The alert's own rule documentation (classtype, CVE/CVSS data, MITRE \
ATT&CK mapping, rule_vulnerability, Snort's own priority) is the primary, \
specific technical evidence for how dangerous THIS alert actually is. Weigh \
that evidence first. Use the NIST/ISO excerpts only to name which control \
area/policy this alert relates to, not to validate or invalidate its \
technical severity.
  - Snort's own priority reflects real detection-engineering judgement. \
Treat it as a meaningful starting point. Only deviate from the normal \
mapping (priority 1 -> rank 1-2, priority 2 -> rank 3, priority 3 -> rank \
4-5) when the alert's OWN technical evidence (not the mere absence of a \
NIST/ISO match) clearly supports a different rank.

Rank it on a scale of 1 to 5:
  1 = Critical / clearly dangerous - must be reviewed immediately
  2 = Likely a real attack or vulnerability - worth review
  3 = Uncertain / moderate - could go either way
  4 = Low severity - probably not worth analyst time
  5 = Clearly informational / noise - not a real threat

(Note: this is the OPPOSITE direction from a 1=noise/5=critical scale - here \
1 is the MOST severe, 5 is the LEAST severe.)

Use your own judgement, but your justification MUST explicitly say which \
specific piece of the retrieved document context (name the document and the \
relevant control/clause if you can) supports your decision. If the \
retrieved context doesn't say anything relevant, say so plainly and explain \
your reasoning from the alert data alone (classtype, message, CVE \
references, destination port, etc.).

You must also report which exact Snort rule sid you judged this alert \
against. Read it from the ALERT DATA given to you (alert.sid / \
rule_documentation.sid) - do NOT take a sid from some other, unrelated rule \
that happens to appear in the retrieved document context.

Do NOT default to rank 3 just because you're unsure which direction to \
lean. Rank 3 means the evidence is genuinely mixed or absent, not "I didn't \
fully analyze this." If the rule documentation gives you a specific CVE, \
CVSS score, or MITRE ATT&CK technique, that alone is enough concrete \
evidence to commit to a 1, 2, 4, or 5 - use rank 3 only when the rule \
documentation is generic (e.g. no CVE/CVSS/MITRE mapping at all) AND you \
have no other basis to lean toward either end.

Respond with ONLY a single JSON object, nothing else, with exactly these \
keys, IN THIS ORDER (write your reasoning out first, THEN commit to a rank \
based on what you just wrote - do not decide the rank before writing the \
justification):
{"justification": "<2-4 sentences citing the document evidence and/or alert \
data that led to this rank>", "metrics_used": ["<specific fields/evidence \
you used>"], "reported_sid": <integer - the sid of THIS alert's own Snort \
rule, from the ALERT DATA>, "severity_rank": <integer 1-5, decided AFTER \
writing the justification above>}
"""

JUDGE_SYSTEM_PROMPT = """You are an independent SOC quality-assurance reviewer. \
You did NOT write the ranking below - your only job is to grade how well it \
is actually supported by the evidence, not to re-rank the alert yourself.

You will be given: the original alert data, the retrieved document context, \
and another analyst's severity_rank + justification for this alert.

Score the JUSTIFICATION (not the numeric rank itself) on a continuous scale \
from 0.0 to 1.0, where:
  1.0 = justification is fully and specifically supported by the cited \
document context and/or alert data (correct document/clause names, \
accurate claims, nothing invented)
  0.5 = partially supported - some reasonable reasoning but vague, generic, \
or only loosely tied to the actual evidence provided
  0.0 = not supported at all - justification is generic filler, contradicts \
the evidence, or cites a document/control that isn't actually relevant \
or doesn't say what the justification claims it says

Use fine-grained values (e.g. 0.16, 0.42, 0.83), not just 0/0.5/1.

Respond with ONLY a JSON object: {"judge_score": <float between 0.0 and 1.0>, \
"judge_reasoning": "<1-3 sentences explaining the score>"}
"""

MISMATCH_SYSTEM_PROMPT = """You are the same SOC analyst who just ranked this \
alert. Your new-scale severity_rank did not fall into the range normally \
expected for this alert's original Snort priority.

Snort priority is on its own 1-3 scale (1=high severity, 3=low severity), \
your scale is 1-5 (1=critical, 5=noise). The normal expected mapping is:
  Snort 1 -> new rank 1 or 2
  Snort 2 -> new rank 3
  Snort 3 -> new rank 4 or 5

Your rank fell outside that expected range for this alert. Explain, using \
SOLID evidence from the retrieved document context and/or the alert data \
(name the document/control/clause), why you still believe your rank is \
correct despite disagreeing with Snort's own priority. Do not just restate \
your original justification - explain the disagreement specifically.

Respond with ONLY a JSON object: {"mismatch_justification": "<2-4 sentences>"}
"""


# ── Detection & loading ──────────────────────────────────────────────────────

def is_snort_alerts_payload(data: Any) -> bool:
    """Detect a Snort alerts JSON without requiring any special prompt text."""
    if isinstance(data, dict) and isinstance(data.get("alerts"), list) and data["alerts"]:
        sample = data["alerts"][0]
        if isinstance(sample, dict):
            alert = sample.get("alert", sample)
            if isinstance(alert, dict) and ("sid" in alert or "priority" in alert):
                return True
        return "sid" in str(sample) or "priority" in str(sample)

    if isinstance(data, list) and data:
        sample = data[0]
        if isinstance(sample, dict):
            alert = sample.get("alert", sample)
            if isinstance(alert, dict) and ("sid" in alert or "priority" in alert):
                return True

    if isinstance(data, dict):
        alert = data.get("alert", data)
        if isinstance(alert, dict) and ("sid" in alert or "priority" in alert):
            return True

    return False


def _normalize_entry(entry: Any) -> Dict:
    """Wrap a flat alert dict (no 'alert' key) as {"alert": entry} so
    downstream code's alert_entry.get("alert", {}) always finds the fields -
    matches the same flat-shape fallback is_snort_alerts_payload() already
    detects, otherwise a flat upload would be "recognized" but then
    processed with entirely empty extracted data."""
    if isinstance(entry, dict) and "alert" not in entry:
        return {"alert": entry}
    return entry


def extract_alert_entries(data: Any) -> List[Dict]:
    """Normalize any of the accepted shapes into a flat list of alert entries."""
    if isinstance(data, dict) and isinstance(data.get("alerts"), list):
        return [_normalize_entry(e) for e in data["alerts"]]
    if isinstance(data, list):
        return [_normalize_entry(e) for e in data]
    if isinstance(data, dict):
        return [_normalize_entry(data)]
    return []


def build_retrieval_query(alert_entry: Dict) -> str:
    alert = alert_entry.get("alert", {}) or {}
    rule_doc = alert_entry.get("rule_documentation", {}) or {}
    parts = [
        alert.get("alert_message", ""),
        alert.get("classification", ""),
        rule_doc.get("classtype", ""),
        rule_doc.get("rule_category", ""),
        f"protocol {alert.get('protocol', '')} port {alert.get('destination_port', '')}",
    ]
    return " ".join(p for p in parts if p)


# ── JSON parsing helpers ─────────────────────────────────────────────────────

def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def extract_json_object(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start:end + 1]


def parse_json_response(raw: str, required_keys: List[str]):
    cleaned = strip_json_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = extract_json_object(cleaned)
        parsed = None
        if extracted:
            try:
                parsed = json.loads(extracted)
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, dict) and all(k in parsed for k in required_keys):
        return parsed
    return None


# ── Ollama call wrapper ──────────────────────────────────────────────────────

def call_ollama_json(ollama_client, model: str, system_prompt: str, user_prompt: str,
                      max_retries: int = 3, temperature: float = 0.0,
                      validate: Callable[[str], bool] | None = None,
                      attempt_log: List[Dict] | None = None) -> str:
    """Calls the given Ollama model in JSON mode and returns the raw content
    string. Retries on transient errors (Ollama not warmed up yet, etc.).

    Uses the given `temperature` on every attempt except the last, which
    retries with at least a little randomness (temperature 0, greedy
    decoding, occasionally locks a model into a degenerate repetition loop
    for a given prompt - "prediction aborted, token repeat limit reached" -
    so the final attempt guarantees an escape route even for temperature-0
    callers).

    `validate`: optional callback that receives the raw response text and
    returns True if it's usable. A small model can get stuck repeating the
    same sentence verbatim for the entire num_predict budget - Ollama treats
    that as a normal completion (no exception), so without this check a bad
    repetitive response would be returned as "successful" on attempt 1 and
    the temperature-escalation safety net above would never even run. When
    `validate` is given, a response that fails it is treated the same as a
    transient error (counted against max_retries, eligible for the same
    final-attempt temperature bump) instead of being returned as-is.

    `attempt_log`: optional list this appends one timing entry to per
    attempt (start/end/duration + outcome) - lets the dashboard's Timing tab
    show exactly how many retries happened and why, instead of just one
    collapsed duration for the whole call."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = None
    for attempt in range(max_retries):
        attempt_temperature = temperature if attempt < max_retries - 1 else max(temperature, 0.4)
        _a0 = _now()
        response = None
        try:
            response = ollama_client.chat(
                model=model,
                messages=messages,
                stream=False,
                format="json",
                # Reasoning models (e.g. gemma4:e2b) generate a hidden
                # "thinking" chain-of-thought that shares the SAME
                # num_predict token budget as the actual answer - on a long
                # prompt like ours, thinking alone can consume the whole
                # budget, truncating the visible JSON mid-string and making
                # it unparseable. think=False skips that entirely and is a
                # harmless no-op on models that don't support it.
                think=False,
                # num_predict caps how many tokens the model is allowed to
                # GENERATE (the answer, not the prompt). 800 turned out too
                # tight: observed raw outputs get cut off mid-string (e.g.
                # `"justification": "This alert is classified as a web applic`
                # with no closing quote/brace) before the model finishes the
                # full JSON object - a genuinely truncated response, not a
                # malformed one, so parse_json_response() correctly fails to
                # parse it and the fail-safe rank-3 default kicks in. Raised
                # so a full severity_rank + justification + metrics_used +
                # reported_sid always has room to complete.
                # num_ctx: Ollama defaults to 2048 tokens of INPUT context if
                # this isn't set. A rich rule_doc_block (full MITRE/CVE
                # paragraphs) plus a 4000-char NIST/ISO excerpt can push a
                # single prompt past that default, silently truncating the
                # OLDEST tokens - including the system prompt's JSON-format
                # instructions. Set generously high so nothing gets truncated.
                # repeat_penalty: tried raising this above Ollama's default
                # (1.3, then 1.15) to stop a small model (qwen2.5:1.5b) from
                # looping on the exact same sentence dozens of times for a
                # rich rule_doc_block. It did stop the loop, but at BOTH
                # values it also collapsed nearly every alert's severity_rank
                # to the same "3" regardless of the alert's actual content
                # (27/30, then 29/30 judgments landed on rank 3 across two
                # tests). Root cause: severity_rank is the FIRST key the
                # model generates, before any justification text, so the
                # penalty's lookback window at that point is just the tail
                # of the PROMPT itself (the retrieved NIST/ISO chunk, full of
                # section numbers like 8.9, 8.10, PR.IR-02, DE.CM-01). Any
                # repeat_penalty > 1.0 downweights whichever digits happen to
                # be common in that boilerplate, which isn't "3" - so the
                # model gets nudged toward "3" independent of the alert.
                # Disabled here (1.0 = no penalty). The `validate` callback
                # above (retry + final-attempt temperature bump on a bad/
                # looping response) is the real defense against the
                # repetition loop and doesn't have this side effect.
                options={"temperature": attempt_temperature, "num_predict": 1536,
                         "num_ctx": 8192, "repeat_penalty": 1.0},
            )
            content = response["message"]["content"]
            _a1 = _now()
            if validate is not None and not validate(content):
                last_error = RuntimeError(
                    f"response failed validation (e.g. repetition loop or bad JSON) - "
                    f"last content seen: {content[:300]!r}"
                )
                if attempt_log is not None:
                    attempt_log.append({
                        "attempt": attempt + 1,
                        "start_ts": _fmt_ts(_a0),
                        "end_ts": _fmt_ts(_a1),
                        "duration_sec": round((_a1 - _a0).total_seconds(), 3),
                        "temperature": attempt_temperature,
                        "outcome": "failed validation (bad JSON / repetition loop) - retrying",
                        **_ollama_breakdown(response),
                    })
                time.sleep(1.5)
                continue
            if attempt_log is not None:
                attempt_log.append({
                    "attempt": attempt + 1,
                    "start_ts": _fmt_ts(_a0),
                    "end_ts": _fmt_ts(_a1),
                    "duration_sec": round((_a1 - _a0).total_seconds(), 3),
                    "temperature": attempt_temperature,
                    "outcome": "success",
                    **_ollama_breakdown(response),
                })
            return content
        except Exception as e:
            _a1 = _now()
            if attempt_log is not None:
                entry = {
                    "attempt": attempt + 1,
                    "start_ts": _fmt_ts(_a0),
                    "end_ts": _fmt_ts(_a1),
                    "duration_sec": round((_a1 - _a0).total_seconds(), 3),
                    "temperature": attempt_temperature,
                    "outcome": f"error: {e}",
                }
                if response is not None:
                    entry.update(_ollama_breakdown(response))
                attempt_log.append(entry)
            last_error = e
            time.sleep(1.5)
    raise RuntimeError(f"Ollama call to '{model}' failed after {max_retries} attempts: {last_error}")


# ── Core ranking ─────────────────────────────────────────────────────────────

def judge_justification(ollama_client, payload_record: Dict,
                         context: str, rank: int, justification: str,
                         rule_doc_block: str = "",
                         attempt_log: List[Dict] | None = None) -> Tuple[Any, Any]:
    rule_doc_section = (
        f"RULE DOCUMENTATION FOR THIS ALERT (exact sid lookup, not RAG-retrieved):\n"
        f"{rule_doc_block}\n\n"
        if rule_doc_block else ""
    )
    judge_prompt = (
        f"ALERT DATA:\n{json.dumps(payload_record, indent=2)}\n\n"
        f"{rule_doc_section}"
        f"RETRIEVED DOCUMENT CONTEXT:\n{context[:4000]}\n\n"
        f"ANALYST'S severity_rank: {rank}\n"
        f"ANALYST'S justification: {justification}\n\n"
        "Score this justification per your instructions."
    )
    try:
        raw = call_ollama_json(ollama_client, JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, judge_prompt,
                                attempt_log=attempt_log)
        parsed = parse_json_response(raw, ["judge_score"])
        if not parsed:
            return None, None
        try:
            score = float(parsed["judge_score"])
        except (TypeError, ValueError):
            return None, None
        score = max(0.0, min(1.0, score))
        return score, parsed.get("judge_reasoning")
    except Exception as e:
        return None, f"(judge call failed: {e})"


def rank_one_alert(ollama_client, ranking_model: str, alert_entry: Dict,
                    rag_system,
                    retrieved: Tuple[str, float] | None = None) -> Dict:
    """Runs the full single-pass ranking pipeline for one alert: RAG
    retrieval + exact rule-doc lookup + ranking call + mismatch check +
    judge scoring. Returns a result dict (never raises - failures degrade to
    a fail-safe rank of 3).

    `retrieved`: an optional pre-computed (context, relevance_score) pair.
    Retrieval is a deterministic function of the alert's own data (same
    query -> same embedding -> same search every time), so when ranking the
    SAME alert multiple times (see rank_alert_three_times), it's wasteful to
    re-run the embedding + search call on every repeat - pass the result in
    once instead. If omitted, this falls back to fetching it itself (so
    direct/standalone calls still work unchanged)."""
    if retrieved is not None:
        context, relevance_score = retrieved
    elif rag_system is not None:
        query = build_retrieval_query(alert_entry)
        context, relevance_score = rag_system.search(query, top_k=4)
    else:
        context, relevance_score = "No RAG system available.", 0.0

    timings: List[Dict] = []

    # Exact sid lookup against rule_docs_preprocessed_by_sid.json - NOT part
    # of the RAG/vector retrieval above. If this alert's sid isn't a key in
    # that table (e.g. a future alert for a rule not in the file yet),
    # rule_doc_block is just "" and that section is skipped - no error.
    _t0 = _now()
    ground_truth_sid = resolve_ground_truth_sid(alert_entry)
    rule_doc_block = get_rule_doc_block(ground_truth_sid)
    timings.append(_stage_timing("rule_doc_lookup", _t0, _now()))

    payload_record = {
        "alert_id": alert_entry.get("alert_id"),
        "alert": alert_entry.get("alert"),
    }

    rule_doc_section = (
        f"RULE DOCUMENTATION FOR THIS ALERT (sid {ground_truth_sid}, exact lookup from "
        f"rule_docs_preprocessed_by_sid.json - given in full, NOT part of the RAG "
        f"retrieval below):\n{rule_doc_block}\n\n"
        if rule_doc_block else ""
    )

    user_prompt = (
        "Rank this ONE alert 1-5 (1=critical, 5=noise) using the alert data, "
        "its rule documentation, and the retrieved document context below. "
        "Respond with ONLY a JSON object with exactly the keys justification, "
        "metrics_used, reported_sid, severity_rank, IN THAT ORDER (write the "
        "justification first, decide severity_rank last, based on it):\n\n"
        f"ALERT DATA:\n{json.dumps(payload_record, indent=2)}\n\n"
        f"{rule_doc_section}"
        f"RETRIEVED DOCUMENT CONTEXT (relevance {relevance_score:.3f}):\n{context[:4000]}\n"
    )

    def _valid_rank_json(raw: str) -> bool:
        """Used as call_ollama_json's `validate` - catches a degenerate
        repetition loop (model "succeeds" with no exception but the content
        never becomes valid JSON with a usable rank), so that case gets the
        SAME retry + final-attempt temperature escalation as a transient
        connection error, instead of being returned as-is on attempt 1."""
        p = parse_json_response(raw, ["severity_rank", "justification"])
        if not p:
            return False
        try:
            r = int(p["severity_rank"])
        except (TypeError, ValueError):
            return False
        return r in (1, 2, 3, 4, 5)

    _t0 = _now()
    parsed = None
    last_raw_debug = None
    ranking_attempts: List[Dict] = []
    for round_idx in range(2):
        round_log: List[Dict] = []
        try:
            raw = call_ollama_json(ollama_client, ranking_model, SYSTEM_PROMPT, user_prompt,
                                    temperature=RANKING_TEMPERATURE, validate=_valid_rank_json,
                                    attempt_log=round_log)
        except Exception as e:
            last_raw_debug = f"(ollama call raised: {e})"
            for entry in round_log:
                entry["round"] = round_idx + 1
            ranking_attempts.extend(round_log)
            continue
        for entry in round_log:
            entry["round"] = round_idx + 1
        ranking_attempts.extend(round_log)
        last_raw_debug = raw
        parsed = parse_json_response(raw, ["severity_rank", "justification"])
        if parsed:
            try:
                rank = int(parsed["severity_rank"])
            except (TypeError, ValueError):
                rank = None
            if rank in (1, 2, 3, 4, 5):
                break
        parsed = None
    timings.append(_stage_timing("ranking_llm_call", _t0, _now(), attempts=ranking_attempts))

    snort_priority = alert_entry.get("alert", {}).get("priority")
    try:
        # EXPECTED_RANGE is keyed by int - coerce so a priority serialized
        # as a numeric string ("1" instead of 1) still matches.
        snort_priority = int(snort_priority) if snort_priority is not None else None
    except (TypeError, ValueError):
        pass

    if not parsed:
        return {
            "model_severity_rank": 3,
            "model_justification": "Could not get a valid model response - defaulted to rank 3 (fail-safe).",
            "model_metrics_used": [],
            "model_defaulted": True,
            "raw_model_output_debug": last_raw_debug,
            "retrieved_context": context,
            "rag_relevance_score": relevance_score,
            "rule_doc_sid": ground_truth_sid,
            "rule_doc_block": rule_doc_block,
            "snort_priority": snort_priority,
            "expected_new_rank_range": None,
            "mismatch_with_snort": None,
            "mismatch_justification": None,
            "reported_sid": None,
            "sid_match": None,
            "judge_score": None,
            "judge_reasoning": None,
            "judge_model": JUDGE_MODEL,
            "timings": timings,
        }

    rank = int(parsed["severity_rank"])
    justification = parsed.get("justification", "")
    metrics = parsed.get("metrics_used") or []
    if not isinstance(metrics, list):
        metrics = []

    # ground_truth_sid was already resolved above (used for the rule_docs
    # lookup) - reused here for the SID mismatch check.
    reported_sid = parsed.get("reported_sid")
    try:
        reported_sid = int(reported_sid) if reported_sid is not None else None
    except (TypeError, ValueError):
        reported_sid = None
    sid_match = None
    if reported_sid is not None and ground_truth_sid is not None:
        sid_match = (reported_sid == ground_truth_sid)

    expected = EXPECTED_RANGE.get(snort_priority)
    mismatch = None
    mismatch_justification = None
    if expected is not None:
        mismatch = not (expected[0] <= rank <= expected[1])
        if mismatch:
            mismatch_prompt = (
                f"Original alert + your ranking:\n{json.dumps(payload_record, indent=2)}\n\n"
                f"Your severity_rank: {rank}\nSnort priority: {snort_priority} "
                f"(expected new-rank range: {expected[0]}-{expected[1]})\n\n"
                f"{rule_doc_section}"
                f"Retrieved document context:\n{context[:3000]}\n"
            )
            _t0 = _now()
            mismatch_attempts: List[Dict] = []
            try:
                mismatch_raw = call_ollama_json(ollama_client, ranking_model,
                                                 MISMATCH_SYSTEM_PROMPT, mismatch_prompt,
                                                 temperature=RANKING_TEMPERATURE,
                                                 attempt_log=mismatch_attempts)
                mismatch_parsed = parse_json_response(mismatch_raw, ["mismatch_justification"])
                if mismatch_parsed:
                    mismatch_justification = mismatch_parsed["mismatch_justification"]
            except Exception as e:
                mismatch_justification = f"(mismatch justification call failed: {e})"
            timings.append(_stage_timing("mismatch_llm_call", _t0, _now(), attempts=mismatch_attempts))

    _t0 = _now()
    judge_attempts: List[Dict] = []
    judge_score, judge_reasoning = judge_justification(
        ollama_client, payload_record, context, rank, justification,
        rule_doc_block=rule_doc_block, attempt_log=judge_attempts,
    )
    timings.append(_stage_timing("judge_llm_call", _t0, _now(), attempts=judge_attempts))

    return {
        "model_severity_rank": rank,
        "model_justification": justification,
        "model_metrics_used": metrics,
        "model_defaulted": False,
        "retrieved_context": context,
        "rag_relevance_score": relevance_score,
        "rule_doc_sid": ground_truth_sid,
        "rule_doc_block": rule_doc_block,
        "snort_priority": snort_priority,
        "expected_new_rank_range": list(expected) if expected else None,
        "mismatch_with_snort": mismatch,
        "mismatch_justification": mismatch_justification,
        "reported_sid": reported_sid,
        "sid_match": sid_match,
        "judge_score": judge_score,
        "judge_reasoning": judge_reasoning,
        "judge_model": JUDGE_MODEL,
        "timings": timings,
    }


def rank_alert_three_times(ollama_client, ranking_model: str, alert_entry: Dict,
                            rag_system, repeats: int = 3,
                            progress_cb=None) -> Dict:
    """Runs rank_one_alert `repeats` times for the SAME alert and reports
    whether the severity_rank was consistent across all runs."""
    alert_start = _now()

    # Retrieve ONCE and reuse for every repeat - the query is built purely
    # from the alert's own fixed data, so re-running the embedding + search
    # call on every repeat would just redo the exact same work for an
    # identical result.
    _t0 = _now()
    if rag_system is not None:
        query = build_retrieval_query(alert_entry)
        retrieved = rag_system.search(query, top_k=4)
    else:
        retrieved = ("No RAG system available.", 0.0)
    retrieval_timing = _stage_timing("rag_retrieval", _t0, _now())

    runs = []
    for i in range(repeats):
        if progress_cb:
            progress_cb(i + 1, repeats)
        runs.append(rank_one_alert(ollama_client, ranking_model, alert_entry, rag_system,
                                    retrieved=retrieved))

    ranks = [r["model_severity_rank"] for r in runs]
    consistent = len(set(ranks)) == 1
    alert_end = _now()

    return {
        "alert_entry": alert_entry,
        "runs": runs,
        "ranks": ranks,
        "consistent": consistent,
        "timings": {
            "alert_start_ts": _fmt_ts(alert_start),
            "alert_end_ts": _fmt_ts(alert_end),
            "alert_duration_sec": round((alert_end - alert_start).total_seconds(), 3),
            "retrieval": retrieval_timing,
            "runs": [r.get("timings", []) for r in runs],
        },
    }


# ── Report formatting (used for both the chat message and the .txt download) ─

def _format_bool(value) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Unknown"


def _build_summary_section(processed: List[Dict]) -> List[str]:
    """Batch-level aggregate statistics across ALL processed alerts (and
    their 3 runs each), mirroring the rank-distribution / rule-usage /
    SID-check / judge-average / Snort-disagreement sections that
    build_report_rag.py used to write to report_<model>.md."""
    total_alerts = len(processed)
    all_runs = [r for item in processed for r in item["runs"]]
    n_judgments = len(all_runs)

    lines = ["## Summary statistics", ""]
    lines.append(f"Aggregated across **{total_alerts} alerts × 3 runs = {n_judgments} judgments**.")
    lines.append("")

    # --- Rank distribution (every individual run counted) ---
    rank_counts = Counter(r["model_severity_rank"] for r in all_runs)
    lines.append("### Rank distribution")
    lines.append("")
    lines.append("| Rank | Meaning | # judgments |")
    lines.append("|---|---|---|")
    for rank in range(1, 6):
        lines.append(f"| {rank} | {SEVERITY_MEANINGS[rank]} | {rank_counts.get(rank, 0)} |")
    lines.append("")

    # --- SID mismatch check (every individual run counted) ---
    sid_matched = sum(1 for r in all_runs if r.get("sid_match") is True)
    sid_mismatched = sum(1 for r in all_runs if r.get("sid_match") is False)
    sid_unknown = sum(1 for r in all_runs if r.get("sid_match") is None)
    lines.append("### SID mismatch check")
    lines.append("")
    lines.append(
        "Checks whether the model reported the SAME Snort rule sid the alert "
        "actually belongs to, rather than one pulled in by mistake from the "
        "RAG-retrieved context (`Unknown` = the model's response didn't "
        "include a parseable reported_sid for that run)."
    )
    lines.append("")
    lines.append(f"- Matched: {sid_matched}/{n_judgments}")
    lines.append(f"- **Mismatched: {sid_mismatched}/{n_judgments}**")
    lines.append(f"- Unknown: {sid_unknown}/{n_judgments}")
    lines.append("")

    # --- LLM-as-judge average (every individual run counted) ---
    judge_scores = [r["judge_score"] for r in all_runs if r.get("judge_score") is not None]
    lines.append(f"### LLM-as-judge (justification quality, 0.0-1.0, judge model: `{JUDGE_MODEL}`)")
    lines.append("")
    if judge_scores:
        avg_judge = sum(judge_scores) / len(judge_scores)
        lines.append(f"- Scored: {len(judge_scores)}/{n_judgments}")
        lines.append(f"- **Average score: {avg_judge:.3f}** (0.0 = unsupported, 1.0 = fully supported)")
    else:
        lines.append("No judge scores were returned.")
    lines.append("")

    lines.append("---")

    return lines


def build_report_markdown(processed: List[Dict], ranking_model: str) -> str:
    """processed = list of dicts returned by rank_alert_three_times."""
    total = len(processed)
    consistent_count = sum(1 for p in processed if p["consistent"])

    lines = [f"# Snort Alert Severity Ranking Report", ""]
    lines.append(f"- **Alerts processed:** {total}")
    lines.append(f"- **Ranking model:** `{ranking_model}` (Ollama, local)")
    lines.append(f"- **Judge model:** `{JUDGE_MODEL}` (Ollama, local, independent)")
    lines.append(f"- **Repeats per alert:** 3")
    lines.append(f"- **Consistent across all 3 runs:** {consistent_count}/{total}")
    lines.append("")
    lines.append("Severity scale: 1 = Critical, 2 = Likely real attack, 3 = Uncertain, "
                  "4 = Low severity, 5 = Informational/noise.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.extend(_build_summary_section(processed))

    for item in processed:
        alert_entry = item["alert_entry"]
        alert = alert_entry.get("alert", {}) or {}
        alert_id = alert_entry.get("alert_id", "?")
        msg = alert.get("alert_message", "(no message)")
        sid = alert.get("sid", "?")
        priority = alert.get("priority", "?")
        runs = item["runs"]
        ranks = item["ranks"]
        consistent = item["consistent"]

        lines.append("")
        lines.append(f"## Alert {alert_id}: {msg}")
        lines.append("")
        lines.append(f"- **Snort priority:** {priority} (sid {sid})")
        lines.append(f"- **AI severity ranks (3 runs):** {', '.join(str(r) for r in ranks)}")
        lines.append(f"- **Consistent across 3 runs:** {'✅ Yes' if consistent else '❌ No'}")
        lines.append("")

        # Rule-docs lookup is also deterministic per alert (exact sid match,
        # not retrieval) - show it ONCE here, same reasoning as the chunks
        # block below.
        rule_doc_sid = runs[0].get("rule_doc_sid")
        rule_doc_block = runs[0].get("rule_doc_block") or ""
        if rule_doc_block:
            lines.append(f"**Rule documentation for this alert (sid {rule_doc_sid}, exact lookup "
                          f"from `rule_docs_preprocessed_by_sid.json` - not RAG-retrieved, given to "
                          f"the model in full):**")
            lines.append("")
            lines.append(rule_doc_block)
            lines.append("")
        else:
            lines.append(f"**Rule documentation:** no entry found for sid {rule_doc_sid} in "
                          f"`rule_docs_preprocessed_by_sid.json` - skipped, model relied on alert "
                          f"data + retrieved context only.")
            lines.append("")

        # Retrieval is deterministic (same alert -> same query -> same
        # chunks every time), so all 3 runs get identical chunks - show them
        # ONCE here instead of repeating the same text 3 times below.
        retrieved = runs[0].get("retrieved_context") or ""
        total_chunks = [c.strip() for c in retrieved.split("<<<CHUNK_BREAK>>>") if c.strip()]
        sent_to_model = retrieved[:4000]
        chunks = [c.strip() for c in sent_to_model.split("<<<CHUNK_BREAK>>>") if c.strip()]
        label = f"**Retrieved chunks for this alert ({len(chunks)} shown, same for all 3 runs"
        if len(chunks) < len(total_chunks):
            label += f" - {len(total_chunks) - len(chunks)} more were retrieved but cut off by the 4000-char limit actually sent to the model"
        label += "):**"
        lines.append(label)
        lines.append("")
        if chunks:
            for n, chunk in enumerate(chunks, 1):
                lines.append(f"{n}: {chunk}")
                lines.append("")
        else:
            lines.append("(no context retrieved)")
            lines.append("")

        lines.append("| Run | Rank | Meaning | SID match | Judge score |")
        lines.append("|---|---|---|---|---|")
        for i, r in enumerate(runs, 1):
            rank = r["model_severity_rank"]
            meaning = SEVERITY_MEANINGS.get(rank, "-")
            sid_match = _format_bool(r.get("sid_match"))
            judge = r.get("judge_score")
            judge_str = f"{judge:.2f}" if judge is not None else "-"
            flag = " ⚠ defaulted" if r.get("model_defaulted") else ""
            lines.append(f"| {i} | {rank}{flag} | {meaning} | {sid_match} | {judge_str} |")

        lines.append("")
        for i, r in enumerate(runs, 1):
            lines.append(f"**Run {i} justification:** {r.get('model_justification', '')}")
            lines.append("")
            if r.get("model_defaulted") and r.get("raw_model_output_debug"):
                lines.append(
                    f"**Run {i} raw model output (debug - this is what the model actually "
                    f"sent back that failed to parse as valid JSON):**\n```\n{r.get('raw_model_output_debug')}\n```"
                )
                lines.append("")
            if r.get("judge_reasoning"):
                lines.append(f"**Run {i} judge reasoning:** {r.get('judge_reasoning')}")
                lines.append("")

        lines.append("---")

    return "\n".join(lines)

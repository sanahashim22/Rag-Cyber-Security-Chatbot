"""
snort_ranking.py
-----------------
Snort alert severity ranking, ported from
3-runs-of-snort-alerts/rank_alerts_rag.py, but using the app's own local
Ollama models (via the existing ConvoRAG instance for retrieval and the
ollama.Client already used for chat) instead of Groq's API.

For each alert:
  1. Retrieve relevant chunks from the RAG knowledge base (NIST CSF, ISO
     27001, and the embedded Snort rule docs) via the app's ConvoRAG.search().
  2. Match the alert against predefined_rules.json (loaded directly, not
     through RAG retrieval).
  3. Ask the ranking LLM for a severity_rank (1=critical ... 5=noise) with a
     justification citing the matched rules / retrieved evidence.
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
from pathlib import Path
from typing import Any, Dict, List, Tuple

RULES_PATH = Path(__file__).parent / "predefined_rules.json"

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
  - The full raw data for exactly this one alert (and its rule documentation).
  - Relevant excerpts retrieved from cybersecurity standards documents (NIST \
CSF 2.0, ISO/IEC 27001, and any other reference documents provided).
  - Any predefined rules that matched this alert's characteristics.

Judge THIS ONE alert using ALL of the above. Rank it on a scale of 1 to 5:
  1 = Critical / clearly dangerous - must be reviewed immediately
  2 = Likely a real attack or vulnerability - worth review
  3 = Uncertain / moderate - could go either way
  4 = Low severity - probably not worth analyst time
  5 = Clearly informational / noise - not a real threat

(Note: this is the OPPOSITE direction from a 1=noise/5=critical scale - here \
1 is the MOST severe, 5 is the LEAST severe.)

Use your own judgement, but your justification MUST explicitly say:
  - which matched predefined rule(s), if any, influenced your decision, and
  - which specific piece of the retrieved document context (name the \
document and the relevant control/clause if you can) supports your decision.
If neither the predefined rules nor the retrieved context say anything \
relevant, say so plainly and explain your reasoning from the alert data alone.

You must also report which exact Snort rule sid you judged this alert \
against. Read it from the ALERT DATA given to you (alert.sid / \
rule_documentation.sid) - do NOT take a sid from some other, unrelated rule \
that happens to appear in the retrieved document context.

Respond with ONLY a single JSON object, nothing else, with exactly these keys:
{"severity_rank": <integer 1-5>, "justification": "<2-4 sentences citing the \
predefined rule(s) and/or document evidence that led to this rank>", \
"metrics_used": ["<specific fields/evidence you used>"], "reported_sid": \
<integer - the sid of THIS alert's own Snort rule, from the ALERT DATA>}
"""

JUDGE_SYSTEM_PROMPT = """You are an independent SOC quality-assurance reviewer. \
You did NOT write the ranking below - your only job is to grade how well it \
is actually supported by the evidence, not to re-rank the alert yourself.

You will be given: the original alert data, the matched predefined rules, \
the retrieved document context, and another analyst's severity_rank + \
justification for this alert.

Score the JUSTIFICATION (not the numeric rank itself) on a continuous scale \
from 0.0 to 1.0, where:
  1.0 = justification is fully and specifically supported by the cited \
predefined rule(s) and/or document context (correct document/clause names, \
accurate claims, nothing invented)
  0.5 = partially supported - some reasonable reasoning but vague, generic, \
or only loosely tied to the actual evidence provided
  0.0 = not supported at all - justification is generic filler, contradicts \
the evidence, or cites a document/rule/control that isn't actually relevant \
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
SOLID evidence from the predefined rules and/or the retrieved document \
context (name the document/control/clause), why you still believe your \
rank is correct despite disagreeing with Snort's own priority. Do not just \
restate your original justification - explain the disagreement specifically.

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


def load_predefined_rules() -> List[Dict]:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["rules"]


def match_predefined_rules(alert_entry: Dict, rules: List[Dict]) -> List[Dict]:
    alert = alert_entry.get("alert", {}) or {}
    rule_doc = alert_entry.get("rule_documentation", {}) or {}

    message = (alert.get("alert_message") or "").lower()
    classtype = (rule_doc.get("classtype") or "").lower()
    dest_port = alert.get("destination_port")
    docs_blob = " ".join(
        str(v).lower() for v in [
            alert.get("classification"), rule_doc.get("rule_category"),
            rule_doc.get("rule_text"), rule_doc.get("metadata"),
        ] if v
    )

    matched = []
    for rule in rules:
        cond = rule.get("match", {})
        hit = False

        if "classtype_any" in cond and classtype:
            hit = hit or any(c.lower() in classtype for c in cond["classtype_any"])
        if "message_contains_any" in cond:
            hit = hit or any(m.lower() in message for m in cond["message_contains_any"])
        if "dest_port_in" in cond and dest_port is not None:
            try:
                hit = hit or int(dest_port) in cond["dest_port_in"]
            except (TypeError, ValueError):
                pass  # non-numeric port (e.g. ICMP/protocol-only alerts) - just skip this condition
        if "message_or_docs_contains_any" in cond:
            haystack = message + " " + docs_blob
            hit = hit or any(m.lower() in haystack for m in cond["message_or_docs_contains_any"])

        if hit:
            matched.append(rule)
    return matched


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
                      max_retries: int = 3, temperature: float = 0.0) -> str:
    """Calls the given Ollama model in JSON mode and returns the raw content
    string. Retries on transient errors (Ollama not warmed up yet, etc.).

    Uses the given `temperature` on every attempt except the last, which
    retries with at least a little randomness (temperature 0, greedy
    decoding, occasionally locks a model into a degenerate repetition loop
    for a given prompt - "prediction aborted, token repeat limit reached" -
    so the final attempt guarantees an escape route even for temperature-0
    callers)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = None
    for attempt in range(max_retries):
        attempt_temperature = temperature if attempt < max_retries - 1 else max(temperature, 0.4)
        try:
            response = ollama_client.chat(
                model=model,
                messages=messages,
                stream=False,
                format="json",
                # num_predict caps how many tokens the model can generate -
                # our JSON responses never need more than a few hundred, so
                # this bounds worst-case generation time on top of the
                # client's request timeout.
                options={"temperature": attempt_temperature, "num_predict": 800},
            )
            return response["message"]["content"]
        except Exception as e:
            last_error = e
            time.sleep(1.5)
    raise RuntimeError(f"Ollama call to '{model}' failed after {max_retries} attempts: {last_error}")


# ── Core ranking ─────────────────────────────────────────────────────────────

def judge_justification(ollama_client, payload_record: Dict, matched_rules_text: str,
                         context: str, rank: int, justification: str) -> Tuple[Any, Any]:
    judge_prompt = (
        f"ALERT DATA:\n{json.dumps(payload_record, indent=2)}\n\n"
        f"MATCHED PREDEFINED RULES:\n{matched_rules_text}\n\n"
        f"RETRIEVED DOCUMENT CONTEXT:\n{context[:4000]}\n\n"
        f"ANALYST'S severity_rank: {rank}\n"
        f"ANALYST'S justification: {justification}\n\n"
        "Score this justification per your instructions."
    )
    try:
        raw = call_ollama_json(ollama_client, JUDGE_MODEL, JUDGE_SYSTEM_PROMPT, judge_prompt)
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
                    rag_system, rules: List[Dict]) -> Dict:
    """Runs the full single-pass ranking pipeline for one alert: RAG
    retrieval + predefined-rule matching + ranking call + mismatch check +
    judge scoring. Returns a result dict (never raises - failures degrade to
    a fail-safe rank of 3)."""
    matched_rules = match_predefined_rules(alert_entry, rules)
    query = build_retrieval_query(alert_entry)

    if rag_system is not None:
        context, relevance_score = rag_system.search(query, top_k=4)
    else:
        context, relevance_score = "No RAG system available.", 0.0

    payload_record = {
        "alert_id": alert_entry.get("alert_id"),
        "alert": alert_entry.get("alert"),
        "rule_documentation": alert_entry.get("rule_documentation"),
    }
    matched_rules_text = (
        "\n".join(
            f"- {r['id']}: {r['description']} "
            f"(hint: {r.get('severity_hint', r.get('severity_escalation'))}) - {r['rationale']}"
            for r in matched_rules
        )
        or "(no predefined rule matched this alert)"
    )

    user_prompt = (
        "Rank this ONE alert 1-5 (1=critical, 5=noise) using the alert data, "
        "the matched predefined rules, and the retrieved document context below. "
        "Respond with ONLY a JSON object with exactly the keys severity_rank, "
        "justification, metrics_used, reported_sid:\n\n"
        f"ALERT DATA:\n{json.dumps(payload_record, indent=2)}\n\n"
        f"MATCHED PREDEFINED RULES:\n{matched_rules_text}\n\n"
        f"RETRIEVED DOCUMENT CONTEXT (relevance {relevance_score:.3f}):\n{context[:4000]}\n"
    )

    parsed = None
    for _ in range(2):
        try:
            raw = call_ollama_json(ollama_client, ranking_model, SYSTEM_PROMPT, user_prompt,
                                    temperature=RANKING_TEMPERATURE)
        except Exception:
            continue
        parsed = parse_json_response(raw, ["severity_rank", "justification"])
        if parsed:
            try:
                rank = int(parsed["severity_rank"])
            except (TypeError, ValueError):
                rank = None
            if rank in (1, 2, 3, 4, 5):
                break
        parsed = None

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
            "matched_predefined_rules": [r["id"] for r in matched_rules],
            "rag_relevance_score": relevance_score,
            "snort_priority": snort_priority,
            "expected_new_rank_range": None,
            "mismatch_with_snort": None,
            "mismatch_justification": None,
            "reported_sid": None,
            "sid_match": None,
            "judge_score": None,
            "judge_reasoning": None,
            "judge_model": JUDGE_MODEL,
        }

    rank = int(parsed["severity_rank"])
    justification = parsed.get("justification", "")
    metrics = parsed.get("metrics_used") or []
    if not isinstance(metrics, list):
        metrics = []

    # SYSTEM_PROMPT tells the model the sid may live under alert.sid OR
    # rule_documentation.sid - check ground truth the same way, otherwise an
    # export that only nests sid under rule_documentation would make
    # sid_match permanently "Unknown" even when the model answers correctly.
    ground_truth_sid = (
        alert_entry.get("alert", {}).get("sid")
        or alert_entry.get("rule_documentation", {}).get("sid")
    )
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
                f"Matched predefined rules:\n{matched_rules_text}\n\n"
                f"Retrieved document context:\n{context[:3000]}\n"
            )
            try:
                mismatch_raw = call_ollama_json(ollama_client, ranking_model,
                                                 MISMATCH_SYSTEM_PROMPT, mismatch_prompt,
                                                 temperature=RANKING_TEMPERATURE)
                mismatch_parsed = parse_json_response(mismatch_raw, ["mismatch_justification"])
                if mismatch_parsed:
                    mismatch_justification = mismatch_parsed["mismatch_justification"]
            except Exception as e:
                mismatch_justification = f"(mismatch justification call failed: {e})"

    judge_score, judge_reasoning = judge_justification(
        ollama_client, payload_record, matched_rules_text, context, rank, justification,
    )

    return {
        "model_severity_rank": rank,
        "model_justification": justification,
        "model_metrics_used": metrics,
        "model_defaulted": False,
        "matched_predefined_rules": [r["id"] for r in matched_rules],
        "rag_relevance_score": relevance_score,
        "snort_priority": snort_priority,
        "expected_new_rank_range": list(expected) if expected else None,
        "mismatch_with_snort": mismatch,
        "mismatch_justification": mismatch_justification,
        "reported_sid": reported_sid,
        "sid_match": sid_match,
        "judge_score": judge_score,
        "judge_reasoning": judge_reasoning,
        "judge_model": JUDGE_MODEL,
    }


def rank_alert_three_times(ollama_client, ranking_model: str, alert_entry: Dict,
                            rag_system, rules: List[Dict], repeats: int = 3,
                            progress_cb=None) -> Dict:
    """Runs rank_one_alert `repeats` times for the SAME alert and reports
    whether the severity_rank was consistent across all runs."""
    runs = []
    for i in range(repeats):
        if progress_cb:
            progress_cb(i + 1, repeats)
        runs.append(rank_one_alert(ollama_client, ranking_model, alert_entry, rag_system, rules))

    ranks = [r["model_severity_rank"] for r in runs]
    consistent = len(set(ranks)) == 1

    return {
        "alert_entry": alert_entry,
        "runs": runs,
        "ranks": ranks,
        "consistent": consistent,
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

    # --- Predefined rules usage (one count per alert, since rule matching
    # is deterministic and identical across an alert's 3 runs) ---
    rule_tally = Counter()
    for item in processed:
        for rid in item["runs"][0].get("matched_predefined_rules", []):
            rule_tally[rid] += 1
    lines.append("### Predefined rules usage (per alert)")
    lines.append("")
    lines.append("| Rule ID | # alerts matched |")
    lines.append("|---|---:|")
    if rule_tally:
        for rid, count in rule_tally.most_common():
            lines.append(f"| {rid} | {count} |")
    else:
        lines.append("| (none matched) | 0 |")
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

    # --- Alerts that disagreed with Snort's own priority ---
    mismatch_rows = []
    for item in processed:
        alert = item["alert_entry"].get("alert", {}) or {}
        n_mismatch = sum(1 for r in item["runs"] if r.get("mismatch_with_snort"))
        if n_mismatch:
            mismatch_rows.append((
                item["alert_entry"].get("alert_id", "?"),
                alert.get("alert_message", ""),
                alert.get("priority", "?"),
                n_mismatch,
            ))
    lines.append(f"### Alerts that disagreed with Snort's own priority ({len(mismatch_rows)}/{total_alerts})")
    lines.append("")
    if mismatch_rows:
        lines.append("| alert_id | msg | snort priority | mismatched in (of 3 runs) |")
        lines.append("|---|---|---|---|")
        for alert_id, msg, priority, n_mismatch in mismatch_rows:
            msg = msg.replace("|", "\\|")
            lines.append(f"| {alert_id} | {msg} | {priority} | {n_mismatch}/3 |")
    else:
        lines.append("None - every alert's rank fell within the expected range for its Snort priority "
                      "in all 3 runs.")
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

        matched_rules = runs[0].get("matched_predefined_rules") or []
        lines.append(f"- **Matched predefined rules:** {', '.join(matched_rules) if matched_rules else '(none)'}")
        lines.append("")

        lines.append("| Run | Rank | Meaning | SID match | Mismatch vs Snort | Judge score |")
        lines.append("|---|---|---|---|---|---|")
        for i, r in enumerate(runs, 1):
            rank = r["model_severity_rank"]
            meaning = SEVERITY_MEANINGS.get(rank, "-")
            sid_match = _format_bool(r.get("sid_match"))
            mismatch = _format_bool(r.get("mismatch_with_snort"))
            judge = r.get("judge_score")
            judge_str = f"{judge:.2f}" if judge is not None else "-"
            flag = " ⚠ defaulted" if r.get("model_defaulted") else ""
            lines.append(f"| {i} | {rank}{flag} | {meaning} | {sid_match} | {mismatch} | {judge_str} |")

        lines.append("")
        for i, r in enumerate(runs, 1):
            lines.append(f"**Run {i} justification:** {r.get('model_justification', '')}")
            if r.get("mismatch_with_snort"):
                lines.append(f"**Run {i} mismatch justification:** {r.get('mismatch_justification') or '(none returned)'}")
            if r.get("judge_reasoning"):
                lines.append(f"**Run {i} judge reasoning:** {r.get('judge_reasoning')}")
            lines.append("")

        lines.append("---")

    return "\n".join(lines)

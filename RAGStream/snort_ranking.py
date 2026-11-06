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
data that led to this rank>", "reported_sid": <integer - the sid of THIS \
alert's own Snort rule, from the ALERT DATA>, "severity_rank": <integer 1-5, \
decided AFTER writing the justification above>}
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

CORRELATION_SYSTEM_PROMPT = """You are a SOC (Security Operations Center) analyst who has \
already ranked a batch of Snort/Suricata alerts independently, one at a time. Now look at \
the WHOLE batch together, the way an analyst reviews a shift's worth of alerts side by side \
instead of in isolation.

Your job here is different: find alerts that, read TOGETHER as one sequence of related \
events, tell a more dangerous story than any single alert suggests on its own.

CRITICAL - what "related" means here: relatedness must come from the alerts' actual CONTENT \
(the same rule/alert type firing more than once, or one alert's action plausibly causing or \
enabling what the next alert shows), never from coincidence alone. Two alerts that merely \
share a timestamp window, or a source/destination IP, or sit next to each other in the list, \
are NOT "related" if their alert_message/classification describe unrelated, unconnected \
activity - e.g. one "OS-MOBILE Android User-Agent detected" alert and one unrelated web-server \
alert happening around the same time are NOT a group, no matter how close in time they are, \
because nothing about a mobile user-agent plausibly leads to or explains the other alert. \
Timing and shared IP/port are only ever a SUPPORTING detail for a grouping that content \
relevance already justifies - never the reason for a grouping by themselves.

SEQUENCE REQUIREMENT - read carefully: the alerts below are listed in CHRONOLOGICAL order and \
each one carries a sequence_position number (1 = earliest). A valid group's members must stay \
CLOSE TOGETHER in this order - at most ONE unrelated alert in between two consecutive members is \
fine (e.g. positions 2, 3, 5 is OK, one unrelated alert sits at position 4), but do NOT reach \
far across the batch to link alerts separated by two or more unrelated alerts in between (e.g. \
position 2 and position 40 in a 40-alert batch, or even position 2 and position 6, is NOT a \
sequence, no matter how similar they look) - a real sequence happens within a short span of \
time/activity, not scattered across the whole shift.

There are two genuinely valid kinds of grouping:
  1. The SAME alert type (same alert_message / same rule) firing more than once, especially \
consecutively or repeatedly - e.g. several "OS-MOBILE Android User-Agent detected" alerts in a \
row. Individually each occurrence might be low severity, but the repetition/pattern itself can \
mean something more significant (sustained probing, a persistent policy violation, an \
escalating pattern) than any single occurrence suggests - evaluate THAT pattern, not just one \
instance of it.
  2. A genuinely plausible step-by-step attack chain between DIFFERENT alert types, where one \
alert's action is a believable precondition or cause of the next (e.g. a sensitive action \
being triggered, shortly followed by that same kind of sensitive data being accessed or sent \
out; a reconnaissance/scanning alert followed by an exploitation attempt against the same \
specific host it scanned). Use your own general security knowledge of how real attacks unfold \
(recon, initial access, credential theft, privilege escalation, lateral movement, data access, \
exfiltration, and how often each stage is followed by the next) to judge whether such a chain \
is actually plausible here, not just superficially adjacent in time.

BEFORE grouping two DIFFERENT alert types together, apply this test: explain the connection \
using ONLY what each alert's own alert_message/classification actually describes happening - \
if your explanation would still make sense with the timestamps and IPs deleted, it's a real \
content connection; if the only thing holding your explanation together is "they happened \
around the same time" or "they share an IP", it is NOT a real connection, and you must NOT \
group them, no matter how tempting it is to invent a story. For instance, a client's HTTP \
User-Agent string being logged tells you nothing about, and has no causal relationship with, \
a completely different exploit attempt against a server - one is passive metadata about a \
request, the other is an active attack attempt; sharing an IP/timestamp does not change that, \
and the two must be left ungrouped. Do not describe one alert as "attempting" or "leading to" \
the other unless that specific alert_message genuinely describes an action capable of causing \
or enabling what the next one describes.

Do NOT force a grouping - only report a group when the alerts genuinely look related by \
content (one of the two kinds above) and meaningfully change the picture together. An alert \
that doesn't connect to anything else in this batch by actual relevance - not just proximity - \
should simply not appear in any group; it is NOT an error to return no groups at all if \
nothing in this batch is actually related. Returning no groups at all is the CORRECT answer \
far more often than finding one - most alerts in a real batch are unrelated noise.

DO NOT SWALLOW MOST OF THE BATCH INTO ONE GROUP just because many alerts happened in the same \
rough window - a genuine sequence is a tight run of specific, causally-connected actions, not \
"everything that happened around the same time". There is no fixed size limit on a group (a real \
sequence can be long if every member genuinely belongs), but every single member must individually \
pass the relevance test above - if you find yourself including an alert only because it's nearby, \
not because its own content connects to the rest, take it back out, or report no group at all. For \
example, several "OS-MOBILE ... User-Agent detected" alerts sitting near a directory-traversal \
attack in time do NOT belong in that attack's group - passive User-Agent metadata is never part \
of an exploit \
sequence, no matter how close in time it occurred.

MULTIPLE SEPARATE GROUPS ARE EXPECTED, not unusual: a batch can easily contain more than one \
distinct real pattern at once - e.g. one repeated/escalating attack against ONE specific target \
host, and a SEPARATE repeated pattern against a DIFFERENT target host, elsewhere in the same \
batch. When that happens, report them as separate entries in "groups" (each with its own \
alert_ids, event_description, and combined_severity_rank) rather than merging everything into one \
broad group just because they're all in the same batch. Only combine alerts aimed at different \
targets into a SINGLE group when there's a genuine content-based reason they're part of the SAME \
overarching story (e.g. reconnaissance against host A plausibly leading to and explaining a later \
exploit attempt against host B) - never merely because they share a source IP, occurred in the \
same batch, or are both "concerning". Both shapes are valid outcomes depending on what the \
content actually supports: several tight, separately-ranked groups; one broader group when a real \
cross-target story connects them; or no group at all. When unsure whether to report one broad \
group or several tighter ones, prefer several tighter ones - precision beats one sprawling \
narrative that quietly includes alerts that don't really belong.

For EACH group you find, decide:
  - combined_severity_rank: 1-5 on the SAME scale as the independent ranks (1=critical, \
5=noise), reflecting how severe the COMBINED event is - this can be more severe than any one \
alert's own independent rank, since the combination is the whole point.
  - event_description: 1-3 plain-language sentences describing what actually seems to have \
happened across these alerts together - what action occurred, in what order, and why that \
specific combination is concerning.

Your combined_severity_rank MUST be consistent with your own event_description - decide the \
rank based on the same reasoning you just wrote, never separately from it. If your description \
says this looks like a sustained attack, brute-force attempt, or escalating sequence, the rank \
must reflect that (low number); if your description says this looks like routine/coincidental \
activity, the rank must reflect that (high number). A rank that contradicts your own \
description is wrong.

Respond with ONLY a JSON object, nothing else: {"groups": [{"alert_ids": [<the alert_id \
values exactly as given in the input, at least 2 per group>], "event_description": "<text>", \
"combined_severity_rank": <integer 1-5>}, ...]}. If nothing in this batch looks related, \
respond with {"groups": []}.
"""

CORRELATION_VERIFY_SYSTEM_PROMPT = """You are a second, more skeptical SOC analyst performing a quality check on a \
colleague's work. Your colleague just proposed that a specific set of alerts, taken together, form ONE combined/\
correlated security event. Your ONLY job is to critically re-examine that specific proposed group and decide \
whether every member actually belongs - you are not looking for new groups, and you are not re-ranking any alert \
on its own.

Models reviewing a whole batch at once tend to over-include alerts just because they happened close together in \
time, even when an alert's own content has no real connection to the rest of the group. Your job is to catch and \
correct that: for EACH alert_id in the proposed group, ask "does this alert's own content - what it actually \
describes happening - genuinely connect to the others, or is it only here because it's nearby?" Passive/\
informational alerts (e.g. a User-Agent string being logged) almost never belong in an active attack sequence just \
because they occurred nearby in time - remove them unless there's a specific, stated reason they're actually part \
of the same action chain.

Be willing to shrink the group significantly - down to removing most members, or even down to an empty result - \
if you don't believe these alerts truly tell one connected story. Do not be deferential to the original proposal; \
your value here is in catching over-grouping, not rubber-stamping it. At the same time, don't remove an alert that \
genuinely does belong just to make the group smaller - judge each one on its own merits.

Respond with ONLY a JSON object, nothing else: {"keep_alert_ids": [<subset of the given alert_ids that genuinely \
belong together - may be the full list, a shorter list, or empty>], "event_description": "<1-3 sentences \
describing what the KEPT alerts together actually show - ignored if keep_alert_ids has fewer than 2 members>", \
"combined_severity_rank": <integer 1-5, same scale as before (1=critical, 5=noise), reflecting the KEPT group - \
ignored if keep_alert_ids has fewer than 2 members>}
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
                # so a full severity_rank + justification + reported_sid
                # always has room to complete.
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
                    retrieved: Tuple[str, float] | None = None,
                    run_judge: bool = True,
                    run_mismatch: bool = True) -> Dict:
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
    direct/standalone calls still work unchanged).

    `run_judge` / `run_mismatch`: let the caller skip the judge LLM call
    and/or the mismatch-justification LLM call entirely (each is a separate
    model call and a meaningful chunk of total ranking time). The mismatch
    BOOLEAN itself (whether the rank disagrees with Snort's priority) is
    always computed either way - it's free, no LLM call involved - only the
    justification call for it is skipped when `run_mismatch` is False."""
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
        "reported_sid, severity_rank, IN THAT ORDER (write the "
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
        if mismatch and run_mismatch:
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

    judge_score, judge_reasoning = None, None
    if run_judge:
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
        "judge_model": JUDGE_MODEL if run_judge else None,
        "timings": timings,
    }


def rank_alert_three_times(ollama_client, ranking_model: str, alert_entry: Dict,
                            rag_system, repeats: int = 3,
                            progress_cb=None,
                            run_judge: bool = True,
                            run_mismatch: bool = True) -> Dict:
    """Runs rank_one_alert `repeats` times for the SAME alert and reports
    whether the severity_rank was consistent across all runs.

    `run_judge` / `run_mismatch` are forwarded to every rank_one_alert call
    unchanged - see its docstring."""
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
                                    retrieved=retrieved,
                                    run_judge=run_judge, run_mismatch=run_mismatch))

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


# ── Combined / correlated alert analysis ─────────────────────────────────────

def _mode_rank(ranks: List[int]) -> int:
    """Same majority-vote-with-first-run-tiebreak logic as dashboard_app.py's
    mode_rank() - duplicated locally (rather than imported) to avoid a
    circular import, since dashboard_app.py imports this module."""
    counts = Counter(ranks)
    best = max(counts.values())
    for r in ranks:
        if counts[r] == best:
            return r
    return ranks[0]


def _valid_correlation_json(raw: str) -> bool:
    p = parse_json_response(raw, ["groups"])
    return isinstance(p, dict) and isinstance(p.get("groups"), list)


def _valid_verify_json(raw: str) -> bool:
    p = parse_json_response(raw, ["keep_alert_ids"])
    return isinstance(p, dict) and isinstance(p.get("keep_alert_ids"), list)


def _verify_correlation_group(ollama_client, model: str, group: Dict,
                               alert_summaries_by_id: Dict[str, Dict]) -> Tuple[Dict | None, List[Dict]]:
    """Second LLM pass over ONE already-proposed group: a code-level gap check \
    can't catch a model claiming "most of the batch is one group", since \
    alert_ids that are all close together in sequence_position look identical \
    - in POSITION terms - to a real long sequence. Only re-examining the \
    actual CONTENT (this call, with a deliberately skeptical prompt and \
    nothing else in context to anchor on) can catch that kind of \
    over-grouping. Returns (parsed_result_or_None, attempt_log) - a None \
    result means the call/parse itself failed, and the caller falls back to \
    keeping the original unverified group rather than silently discarding a \
    real finding over a technical hiccup (fail-soft, same pattern as the rest \
    of this pipeline)."""
    members = [
        alert_summaries_by_id[str(aid)]
        for aid in group["alert_ids"]
        if str(aid) in alert_summaries_by_id
    ]
    user_prompt = (
        "A colleague proposed that the following alerts form ONE combined/correlated event:\n\n"
        f"Proposed event_description: {group.get('event_description', '')}\n"
        f"Proposed combined_severity_rank: {group.get('combined_severity_rank')}\n\n"
        f"The proposed alerts (in chronological order):\n{json.dumps(members, indent=2)}\n\n"
        "Critically re-examine this group per your instructions."
    )
    attempts: List[Dict] = []
    try:
        raw = call_ollama_json(ollama_client, model, CORRELATION_VERIFY_SYSTEM_PROMPT, user_prompt,
                                temperature=RANKING_TEMPERATURE, validate=_valid_verify_json,
                                attempt_log=attempts)
        parsed = parse_json_response(raw, ["keep_alert_ids"])
    except Exception:
        parsed = None
    return parsed, attempts


# Max allowed gap between two consecutive members' sequence_position values
# within an accepted group (e.g. 2 means positions 2 and 4 are still OK - up
# to 1 unrelated alert interleaved in between - but 2 and 5 is not). Real
# alert streams interleave related events with unrelated noise, so requiring
# a PERFECTLY contiguous run (gap of 1) rejected almost every genuine
# repeated-rule pattern in testing; this tolerates a LITTLE interleaving (at
# most one unrelated alert between two members) while still refusing to link
# alerts separated by a larger, unrelated span.
_MAX_SEQUENCE_GAP = 2


def analyze_correlated_alerts(ollama_client, model: str, processed: List[Dict]) -> Dict:
    """Takes the already-independently-ranked `processed` list (one entry per
    alert, as returned by rank_alert_three_times) for a batch run and asks
    CORRELATION_SYSTEM_PROMPT to look at the WHOLE batch together and suggest
    groups of alert_ids that collectively tell a more dangerous story than
    any one of them alone - e.g. alerts 4, 5, 6 together look like a
    sustained/escalating sequence worth reviewing as one event. This is
    entirely a model judgment call based on the alerts' own content (not an
    exact-match/code-confirmed check), so every group returned carries
    kind="ai_suggested".

    Which alerts are picked and how severe the combination is are the
    model's judgment call - but WHICH SHAPE a valid group can take is
    enforced in code, not just prompted: alerts are sorted chronologically
    and stamped with a sequence_position, and any model-suggested group
    whose members are spread further apart than _MAX_SEQUENCE_GAP positions
    is dropped outright - a tight sequence like 2,3,5 (one unrelated alert
    skipped) is kept, but a wide scattered set like 2,20,38 in a 40-alert
    batch is never returned, regardless of what the model asked for. A
    group's SIZE is not capped - it can be any number of alerts, large or
    small, as long as its members stay close together in sequence.

    That gap check alone can't catch over-grouping where the model claims
    nearly the WHOLE batch is one group - if those alert_ids are mostly
    adjacent in sequence, the gaps are all small and the check passes even
    though the grouping is wrong. So every surviving group also goes through
    a second, independent LLM pass (_verify_correlation_group) that is shown
    ONLY that one proposed group - deliberately out of context of the rest of
    the batch - and asked to skeptically re-examine and, if needed, shrink
    it. This is still a model judgment call (not a code-level size cap), just
    a second one dedicated to catching the first call's over-inclusion
    mistakes instead of trusting its first answer outright.

    Only meaningful with more than one alert - callers should skip calling
    this entirely for a single-alert batch rather than relying on this
    function to notice.

    Returns {"groups": [...], "timings": [<stage timing dicts>]}. Never
    raises - a call/parse failure just degrades to "nothing found", same
    fail-soft pattern as the rest of this pipeline."""
    alert_summaries = []
    for item in processed:
        alert = item["alert_entry"].get("alert", {}) or {}
        majority = _mode_rank(item["ranks"])
        # The justification from whichever run actually landed on the
        # majority rank (falls back to the first run) - representative of
        # why this alert was ranked the way it was, without dragging all 3
        # runs' full text into an already multi-alert prompt.
        rep_run = next(
            (r for r in item["runs"] if r.get("model_severity_rank") == majority),
            item["runs"][0],
        )
        alert_summaries.append({
            "alert_id": item["alert_entry"].get("alert_id"),
            "alert_message": alert.get("alert_message"),
            "classification": alert.get("classification"),
            "timestamp": alert.get("timestamp"),
            "source_ip": alert.get("source_ip"),
            "source_port": alert.get("source_port"),
            "destination_ip": alert.get("destination_ip"),
            "destination_port": alert.get("destination_port"),
            "protocol": alert.get("protocol"),
            "snort_priority": alert.get("priority"),
            "independent_severity_rank": majority,
            "independent_justification": rep_run.get("model_justification", ""),
        })

    # Sort chronologically and stamp an explicit sequence_position (1 =
    # earliest) - Snort's native "MM/DD-HH:MM:SS.ffffff" timestamp format is
    # fixed-width/zero-padded, so a plain string sort is already a correct
    # chronological sort. This position number is what both the prompt's
    # "contiguous run" instruction and the code-level contiguity check below
    # key off of, instead of trusting list order alone.
    alert_summaries.sort(key=lambda s: s.get("timestamp") or "")
    for i, s in enumerate(alert_summaries, start=1):
        s["sequence_position"] = i
    position_by_id = {str(s["alert_id"]): s["sequence_position"] for s in alert_summaries}

    user_prompt = (
        f"Here are {len(alert_summaries)} alerts from the same batch, already ranked "
        "independently (one at a time), listed in CHRONOLOGICAL order with a sequence_position "
        "field (1 = earliest). Find any that are related to each other and describe the combined "
        "event(s), per your instructions:\n\n"
        f"{json.dumps(alert_summaries, indent=2)}\n"
    )

    _t0 = _now()
    attempts: List[Dict] = []
    candidate_groups: List[Dict] = []
    try:
        raw = call_ollama_json(ollama_client, model, CORRELATION_SYSTEM_PROMPT, user_prompt,
                                temperature=RANKING_TEMPERATURE, validate=_valid_correlation_json,
                                attempt_log=attempts)
        parsed = parse_json_response(raw, ["groups"])
        if parsed and isinstance(parsed.get("groups"), list):
            for g in parsed["groups"]:
                if not isinstance(g, dict):
                    continue
                ids = g.get("alert_ids")
                if not isinstance(ids, list) or len(ids) < 2:
                    continue
                try:
                    rank = int(g.get("combined_severity_rank"))
                except (TypeError, ValueError):
                    rank = None
                candidate_groups.append({
                    "alert_ids": ids,
                    "event_description": g.get("event_description", ""),
                    "combined_severity_rank": rank if rank in (1, 2, 3, 4, 5) else None,
                })
    except Exception:
        candidate_groups = []
    timings = [_stage_timing("correlation_llm_call", _t0, _now(), attempts=attempts)]

    # Reliability guarantee, enforced in CODE rather than trusted to the
    # model's compliance: a group is only kept if its alert_ids map to
    # sequence_position values that stay CLOSE TOGETHER in chronological
    # order - consecutive members may skip up to _MAX_SEQUENCE_GAP - 1
    # unrelated alerts in between (real alert streams interleave related
    # events with unrelated noise - e.g. the same attack rule firing 4 times
    # with a couple of unrelated alerts from other traffic in between is
    # still one real pattern), but a group may NOT reach far across the
    # batch to link alerts separated by many unrelated alerts. Any id the
    # model hallucinated (not in this batch), any duplicate, or any group
    # with too wide a gap is dropped outright rather than repaired, same
    # fail-soft pattern as the rest of this pipeline. Kept groups have their
    # alert_ids reordered into chronological order for display.
    close_groups = []
    for g in candidate_groups:
        positions = [position_by_id.get(str(aid)) for aid in g["alert_ids"]]
        if any(p is None for p in positions) or len(set(positions)) != len(positions):
            continue
        ordered_positions = sorted(positions)
        gaps_ok = all(
            b - a <= _MAX_SEQUENCE_GAP
            for a, b in zip(ordered_positions, ordered_positions[1:])
        )
        if not gaps_ok:
            continue
        ordered = [aid for _, aid in sorted(zip(positions, g["alert_ids"]))]
        close_groups.append({**g, "alert_ids": ordered})
    candidate_groups = close_groups

    # Second LLM pass per surviving group: a dedicated, skeptical re-check
    # that only sees this one proposed group (not the whole batch), asked to
    # shrink it if any member doesn't genuinely belong. Catches the
    # "swallowed most of the batch" failure mode the gap filter above can't -
    # see analyze_correlated_alerts' docstring.
    alert_summaries_by_id = {str(s["alert_id"]): s for s in alert_summaries}
    verified_groups: List[Dict] = []
    for g in candidate_groups:
        _tv0 = _now()
        verify_result, verify_attempts = _verify_correlation_group(
            ollama_client, model, g, alert_summaries_by_id
        )
        timings.append(_stage_timing("correlation_verify_llm_call", _tv0, _now(), attempts=verify_attempts))

        if not isinstance(verify_result, dict) or not isinstance(verify_result.get("keep_alert_ids"), list):
            # Verify call/parse failed - fall back to the original,
            # unverified group rather than discarding a real finding over a
            # technical hiccup.
            verified_groups.append(g)
            continue

        keep_ids_set = {str(k) for k in verify_result["keep_alert_ids"]}
        new_ids = [aid for aid in g["alert_ids"] if str(aid) in keep_ids_set]
        if len(new_ids) < 2:
            continue  # the second pass decided these alerts don't really belong together

        new_positions = sorted(position_by_id.get(str(aid)) for aid in new_ids)
        if any(p is None for p in new_positions) or not all(
            b - a <= _MAX_SEQUENCE_GAP for a, b in zip(new_positions, new_positions[1:])
        ):
            continue  # shrinking left a scattered remainder - don't report it

        try:
            rank = int(verify_result.get("combined_severity_rank"))
        except (TypeError, ValueError):
            rank = None
        verified_groups.append({
            "alert_ids": new_ids,
            "event_description": verify_result.get("event_description") or g["event_description"],
            "combined_severity_rank": rank if rank in (1, 2, 3, 4, 5) else g["combined_severity_rank"],
        })
    candidate_groups = verified_groups

    final_groups = [
        {
            "alert_ids": g["alert_ids"],
            "event_description": g["event_description"],
            "combined_severity_rank": g["combined_severity_rank"],
            "kind": "ai_suggested",
        }
        for g in candidate_groups
    ]

    return {"groups": final_groups, "timings": timings}


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
            if r.get("mismatch_with_snort") and r.get("mismatch_justification"):
                expected = r.get("expected_new_rank_range")
                expected_str = f"{expected[0]}-{expected[1]}" if expected else "?"
                lines.append(
                    f"**Run {i} mismatch justification** (disagrees with Snort's priority, which "
                    f"implies rank {expected_str}): {r.get('mismatch_justification')}"
                )
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

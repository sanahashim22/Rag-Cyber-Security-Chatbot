#!/usr/bin/env python3
"""The threat-intelligence database on the LLM PC: fetch it from the IDS
dashboard, and write what it says of a combined JSON file's rules, CVEs and
ATT&CK techniques as text, for the LLM's prompt.

The combined JSON has no enrichment. Its rules are named
``snort_signature:GID:SID:REV`` / ``suricata_signature:GID:SID:REV``; each
rule's text names its CVEs (``reference:cve,2014-6271``), Suricata's alert
metadata names CVEs (``CVE_2014_6271``) and ATT&CK techniques
(``mitre_technique_id``), and a rule's message can name a CVE. This script
finds those and reads what the database holds on them:

- a Snort rule: Talos's documentation of it (category, explanation, what to
  look for, false positives, known usage, vulnerability type), or for a
  built-in rule (GID 2 and up) Snort's own description of it and the part
  of Snort that raises it (its packet decoder or an inspector);
- a Suricata rule: the rule's own documentation, since ET Open has no
  pages like Talos's (its ruleset and category, what its class means,
  confidence, severity, when it was written and updated, references, CVEs
  and ATT&CK IDs);
- a CVE: the NVD summary, publication date, CVSS, CWEs, EPSS and, when it is
  in CISA's Known Exploited Vulnerabilities catalog, its KEV entry;
- a technique: its name, tactics, platforms and description.

Copy this one file to the LLM PC (Windows or Linux; Python 3.8 or newer, no
packages) and run it there:

    python intel_lookup.py download http://172.16.20.142:8501/
    python intel_lookup.py enrich snort_suricata_combined_20261117_1651_last20.json
    python intel_lookup.py lookup snort_signature:1:31978:5 CVE-2014-6271 T1190
    python intel_lookup.py lookup 2009208 116:414

A rule can be named in full (``suricata_signature:1:2009208:4``) or by
GID:SID:REV, GID:SID or SID alone: each rule of either engine with that
SID is given.

``download`` saves the dashboard's intel.db (about 1.4 GB; the IDS machine
refreshes it) next to this script, or at --db; it is written under a
temporary name and renamed when complete. ``enrich`` and ``lookup`` print
plain text to put in the prompt (--json for JSON). A program on the LLM PC
can import it instead: ``prompt_text(db, combined)`` gives the same text for
a combined file's JSON (text or parsed). With the dashboard's API token on,
give it with --token or IDS_API_TOKEN.

NOTE: this file is copied verbatim from the IDS-Dashboard repo's
scripts/intel_lookup.py, per that project's own instructions for the "LLM
PC" side of the integration - see dashboard_app.py / snort_ranking.py for
how this project calls into it (lookup() + as_text(), by gid/sid/rev, not
the CLI). Re-copy it from there if their schema ever changes (this copy was
refreshed on 2026-11-23 to add suricata_rule_docs/snort_builtin_rules
support - sids like Suricata's ET rules were returning no documentation at
all under the previous copy, even though intel.db already had it).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_DB = Path(__file__).resolve().with_name("intel.db")
# CVE-2014-6271, CVE_2014_6271 (Suricata's metadata) and reference:cve,2014-6271.
_CVE_RE = re.compile(r"\bCVE[-_](\d{4})[-_](\d{4,7})\b", re.IGNORECASE)
_CVE_REFERENCE_RE = re.compile(r"reference\s*:\s*cve\s*,\s*(\d{4})-(\d{4,7})", re.IGNORECASE)
_TECHNIQUE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
_RULE_RE = re.compile(r"\b(snort|suricata)_signature:(\d+):(\d+):(\d+)\b")
# 2009208, 116:414, 1:2009208:4.
_PLAIN_RULE_RE = re.compile(r"(?:(\d+):)?(\d+)(?::(\d+))?")
# Talos's explanation: a description, then these sections.
_TALOS_SECTION_RE = re.compile(r"(?:^|\n)[ \t]*(Impact|Details|Ease of Attack):[ \t]*")
# ATT&CK's citations and Markdown links.
_CITATION_RE = re.compile(r"\(Citation:[^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def _json(text: Any, default: Any) -> Any:
    try:
        return json.loads(text) if text else default
    except ValueError:
        return default


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def named(text: str) -> dict[str, list[str]]:
    """The rules, CVE IDs and ATT&CK technique IDs a text names, each once."""
    cves = {f"CVE-{year}-{number}" for year, number in _CVE_RE.findall(text)}
    cves |= {f"CVE-{year}-{number}" for year, number in _CVE_REFERENCE_RE.findall(text)}
    rules = {match.group(0) for match in _RULE_RE.finditer(text)}
    return {
        "rules": sorted(rules, key=lambda rule: [int(part) if part.isdigit() else part for part in rule.split(":")]),
        "cves": sorted(cves),
        "attack_techniques": sorted(set(_TECHNIQUE_RE.findall(text))),
    }


def _without_empty(value: Any) -> Any:
    if isinstance(value, dict):
        kept = {key: _without_empty(item) for key, item in value.items()}
        return {key: item for key, item in kept.items() if item not in (None, "", [], {})}
    return value


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone() is not None


def _row(connection: sqlite3.Connection, table: str, gid: int, sid: int) -> sqlite3.Row | None:
    if not _has_table(connection, table):
        return None
    return connection.execute(f"SELECT * FROM {table} WHERE gid = ? AND sid = ?", (gid, sid)).fetchone()


def _words(value: Any) -> str:
    """A metadata value as words: Client_Endpoint is Client Endpoint."""
    return _one_line(str(value or "").replace("_", " "))


def _raised_by(module: str, kind: str, about: str) -> str:
    """Which part of Snort raises a built-in rule, and what that means."""
    about = f" ({about})" if about else ""
    if kind == "codec" or module == "decode":
        return (
            f"Snort's packet decoder, {module}{about}, while reading the packet's headers: the packet "
            "is unusual or malformed for that protocol. This is a check on the packet's form, not a "
            "signature of a known attack, and it names no CVE."
        )
    if kind.startswith("inspector"):
        return (
            f"Snort's {module} inspector{about}: something that inspector watches for in the protocol "
            "or the traffic's behaviour, not a match of a written attack signature. It names no CVE."
        )
    return f"Snort's {module} module{about}."


def _builtin_details(connection: sqlite3.Connection, gid: int, sid: int) -> dict[str, Any] | None:
    """Snort's own description of a built-in rule and the module that raises it."""
    described = _row(connection, "snort_builtin_docs", gid, sid)
    listed = _row(connection, "snort_builtin_rules", gid, sid)
    if described is None and listed is None:
        return None
    details: dict[str, Any] = {"description": described["description"] if described else None}
    if listed is not None:
        details.update({
            "message": listed["msg"],
            "module": listed["module"],
            "module_help": listed["module_help"],
            "raised_by": _raised_by(
                str(listed["module"] or ""), str(listed["module_type"] or ""), _one_line(listed["module_help"]),
            ),
        })
    return {"snort_documentation": _without_empty(details)}


def _suricata_details(connection: sqlite3.Connection, gid: int, sid: int, rev: int | None) -> dict[str, Any] | None:
    """The documentation a Suricata rule carries itself."""
    row = _row(connection, "suricata_rule_docs", gid, sid)
    if row is None:
        return None
    metadata = _json(row["metadata_json"], {})
    references = []
    for reference in _json(row["references_json"], []):
        kind, _, value = str(reference).partition(",")
        if kind.lower() != "cve":  # the CVEs are given below
            references.append(value.strip() if kind.lower() == "url" else f"{kind} {value.strip()}")
    return _without_empty({"suricata_rule": {
        "ruleset": row["ruleset"],
        "message": row["msg"],
        "category": row["category"],
        "class": {
            "name": row["classtype"], "description": row["class_description"], "priority": row["priority"],
        } if row["classtype"] else None,
        "confidence": _words(row["confidence"]),
        "signature_severity": _words(row["signature_severity"]),
        "created": row["created_at"],
        "updated": row["updated_at"],
        "rev_in_database": row["rev"] if rev is not None and row["rev"] is not None and row["rev"] != rev else None,
        **{
            key: ", ".join(_words(value) for value in metadata.get(key, []))
            for key in ("attack_target", "deployment", "affected_product", "malware_family", "tag", "former_category")
        },
        "references": references,
        "cves": [cve.strip() for cve in str(row["cve_ids"] or "").split(",") if cve.strip()],
        "attack_tactics": [
            f"{tactic_id} {_words(name)}".strip()
            for tactic_id, name in zip(
                metadata.get("mitre_tactic_id", []),
                metadata.get("mitre_tactic_name", []) + [""] * len(metadata.get("mitre_tactic_id", [])),
            )
        ],
        "attack_techniques": [
            technique.strip() for technique in str(row["mitre_technique_id"] or "").split(",") if technique.strip()
        ],
    }})


def rule_details(connection: sqlite3.Connection, rule: str) -> dict[str, Any] | None:
    """Talos's documentation of a Snort rule, or Snort's own description of
    a built-in rule Talos doesn't document, or a Suricata rule's own
    documentation; None for a rule the database doesn't hold."""
    match = _RULE_RE.fullmatch(rule)
    if not match:
        return None
    gid, sid, rev = int(match.group(2)), int(match.group(3)), int(match.group(4))
    if match.group(1) == "suricata":
        return _suricata_details(connection, gid, sid, rev)
    row = _row(connection, "talos_rule_docs", gid, sid)
    if row is None or not _one_line(row["explanation"]) or gid > 3:
        builtin = _builtin_details(connection, gid, sid)
        if builtin is not None:
            return builtin
    if row is None:
        return None
    pieces = _TALOS_SECTION_RE.split(str(row["explanation"] or ""))
    sections = {"description": _one_line(pieces[0])}
    for label, body in zip(pieces[1::2], pieces[2::2]):
        sections[label.lower().replace(" ", "_")] = _one_line(body)
    if sections.get("details") and sections["details"] in (
        sections.get("description"), f"{sections.get('description')} {sections.get('impact')}",
    ):
        del sections["details"]
    if sections.get("impact", "").startswith("CVSS "):
        # NVD's scores, which the CVE gives.
        del sections["impact"]
    look_for = _one_line(row["what_to_look_for"])
    kind, _, about = str(row["vulnerability"] or "").strip().partition("\n")
    return _without_empty({
        "talos_documentation": {
            "category": _one_line(row["category"]),
            **sections,
            # Often the description again, after the category's name.
            "what_to_look_for": "" if look_for.endswith(sections.get("description") or "\0") else look_for,
            "false_positives": _one_line(row["false_positives"]),
            "known_usage": _one_line(row["known_usage"]),
            "vulnerability": (
                {"type": _one_line(kind), "description": _one_line(about)}
                if _one_line(kind) and _one_line(kind).upper() != "N/A" else None
            ),
            "cves": [cve.strip() for cve in str(row["cve_ids"] or "").split(",") if cve.strip()],
            "attack_techniques": [
                technique.strip() for technique in str(row["mitre_technique_id"] or "").split(",")
                if technique.strip()
            ],
            "written_from_nvd": "national vulnerability database" in str(row["contributors"] or "").lower(),
        },
    })


def cve_details(connection: sqlite3.Connection, cve: str) -> dict[str, Any] | None:
    nvd = connection.execute(
        "SELECT description, published, cvss_score, cvss_version, cvss_severity, cvss_vector,"
        " cwe_json FROM cves WHERE cve_id = ?", (cve,),
    ).fetchone()
    epss = connection.execute(
        "SELECT score, percentile, score_date FROM epss WHERE cve_id = ?", (cve,),
    ).fetchone()
    kev = connection.execute(
        "SELECT vendor_project, product, vulnerability_name, date_added, short_description,"
        " required_action, due_date, known_ransomware_campaign_use FROM kev WHERE cve_id = ?",
        (cve,),
    ).fetchone()
    if not (nvd or epss or kev):
        return None
    details: dict[str, Any] = {}
    if nvd:
        details.update({
            "summary": _one_line(nvd["description"]),
            "published": (nvd["published"] or "")[:10],
            "cvss": {
                "score": nvd["cvss_score"], "version": nvd["cvss_version"],
                "severity": nvd["cvss_severity"], "vector": nvd["cvss_vector"],
            } if nvd["cvss_score"] is not None else None,
            "cwe": _json(nvd["cwe_json"], []),
        })
    if epss:
        details["epss"] = {
            "score": epss["score"], "percentile": epss["percentile"], "date": epss["score_date"],
        }
    if kev:
        details["cisa_kev"] = dict(kev)
    return _without_empty(details)


def technique_details(connection: sqlite3.Connection, technique: str) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT name, tactics_json, platforms_json, description FROM attack_techniques"
        " WHERE technique_id = ?", (technique,),
    ).fetchone()
    if row is None:
        return None
    description = _LINK_RE.sub(r"\1", _CITATION_RE.sub("", str(row["description"] or "")))
    return _without_empty({
        "name": row["name"],
        "tactics": _json(row["tactics_json"], []),
        "platforms": _json(row["platforms_json"], []),
        # Its first paragraph: what the technique is.
        "description": _one_line(description.strip().split("\n\n")[0]),
    })


def rules_named(connection: sqlite3.Connection, given: str) -> list[str]:
    """The full names of the rules a plain ID (SID, GID:SID or GID:SID:REV)
    names, of either engine, that the database holds."""
    match = _PLAIN_RULE_RE.fullmatch(given)
    if not match:
        return []
    gid = int(match.group(1)) if match.group(1) else None
    sid = int(match.group(2))
    rev = int(match.group(3)) if match.group(3) else None
    names: dict[str, None] = {}
    for engine, table in (
        ("snort", "talos_rule_docs"), ("snort", "snort_builtin_rules"), ("snort", "snort_builtin_docs"),
        ("suricata", "suricata_rule_docs"),
    ):
        if not _has_table(connection, table):
            continue
        has_rev = table in ("talos_rule_docs", "suricata_rule_docs")
        query = f"SELECT gid, sid{', rev' if has_rev else ''} FROM {table} WHERE sid = ?"
        values: list[Any] = [sid]
        if gid is not None:
            query += " AND gid = ?"
            values.append(gid)
        for row in connection.execute(query + " ORDER BY gid", values):
            found_rev = rev or (row["rev"] if has_rev and row["rev"] else 1)
            names[f"{engine}_signature:{row['gid']}:{row['sid']}:{found_rev}"] = None
    return list(names)


def lookup(db: Path, ids: Iterable[str]) -> dict[str, Any]:
    """What the database holds on each rule name, CVE and technique ID. The
    CVEs and techniques a rule's Talos documentation names are looked up too."""
    if not Path(db).is_file():
        raise SystemExit(f"No database at {db}: run `python {Path(__file__).name} download <dashboard URL>` first")
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    found: dict[str, Any] = {"rules": {}, "cves": {}, "attack_techniques": {}, "unknown": []}
    try:
        wanted = [given.strip() for given in ids if given.strip()]
        cves: list[str] = []
        techniques: list[str] = []
        for given in wanted:
            plain = _PLAIN_RULE_RE.fullmatch(given) is not None
            if plain or _RULE_RE.fullmatch(given):
                names = rules_named(connection, given) if plain else [given]
                if plain and not names:
                    found["unknown"].append(given)
                for name in names:
                    details = rule_details(connection, name)
                    if details is None:
                        continue
                    found["rules"][name] = details
                    documented = details.get("talos_documentation") or details.get("suricata_rule") or {}
                    cves += documented.get("cves", [])
                    techniques += documented.get("attack_techniques", [])
            elif given.upper().replace("_", "-").startswith("CVE-"):
                cves.append(given.upper().replace("_", "-"))
            else:
                techniques.append(given.upper())
        for cve in dict.fromkeys(cves):
            details = cve_details(connection, cve)
            if details is None:
                found["unknown"].append(cve)
            else:
                found["cves"][cve] = details
        for technique in dict.fromkeys(techniques):
            details = technique_details(connection, technique)
            if details is None:
                found["unknown"].append(technique)
            else:
                found["attack_techniques"][technique] = details
        found["sources"] = {
            row["source"]: (row["data_as_of"] or row["imported_at"] or "")[:10]
            for row in connection.execute("SELECT source, data_as_of, imported_at FROM source_metadata")
        }
    finally:
        connection.close()
    return _without_empty(found)


def enrich(db: Path, combined: str | Mapping[str, Any]) -> dict[str, Any]:
    """What the database holds on each rule, CVE and technique a combined
    file (its JSON text, or the parsed document) names."""
    text = combined if isinstance(combined, str) else json.dumps(combined)
    ids = named(text)
    return lookup(db, [*ids["rules"], *ids["cves"], *ids["attack_techniques"]])


_SOURCE_NAMES = {
    "nvd": "NVD", "epss": "EPSS", "cisa_kev": "CISA KEV", "mitre_attack": "MITRE ATT&CK",
    "talos_rule_docs": "Talos rule documentation",
    "snort_builtin_docs": "Snort built-in rule descriptions",
    "snort_builtin_rules": "Snort built-in rules",
    "suricata_rule_docs": "Suricata rules",
}


def as_text(found: Mapping[str, Any]) -> str:
    """The lookup as plain text for a prompt: one block per rule, CVE and
    technique."""
    lines: list[str] = []
    sources = found.get("sources", {})
    if sources:
        lines.append("Threat intelligence (" + ", ".join(
            f"{_SOURCE_NAMES.get(source, source)} {date}".strip() for source, date in sources.items()
        ) + ")")
    summaries = {details.get("summary") for details in found.get("cves", {}).values()}
    categories: set[str] = set()
    for rule, details in found.get("rules", {}).items():
        if "snort_documentation" in details:
            snort = details["snort_documentation"]
            lines += ["", f"Rule {rule}: Snort's description of this built-in rule"]
            if snort.get("message") and _one_line(snort["message"]).lower() != _one_line(snort.get("description")).lower():
                lines.append(f"  Message: {snort['message']}")
            if snort.get("description"):
                lines.append(f"  Description: {snort['description']}")
            if snort.get("raised_by"):
                lines.append(f"  Raised by: {snort['raised_by']}")
            continue
        if "suricata_rule" in details:
            rule_docs = details["suricata_rule"]
            lines += ["", f"Rule {rule}: the rule's own documentation ({rule_docs.get('ruleset', 'Suricata')})"]
            for key, label in (("message", "Message"), ("category", "Category")):
                if rule_docs.get(key):
                    lines.append(f"  {label}: {rule_docs[key]}")
            if rule_docs.get("class"):
                kind = rule_docs["class"]
                lines.append(
                    f"  Class: {kind.get('name')}"
                    + (f", {kind['description']}" if kind.get("description") else "")
                    + (f" (priority {kind['priority']})" if kind.get("priority") else "")
                )
            judged = [
                f"{label} {rule_docs[key]}" for key, label in (
                    ("confidence", "confidence"), ("signature_severity", "signature severity"),
                ) if rule_docs.get(key)
            ]
            if judged:
                lines.append(f"  The rule's writers give it: {', '.join(judged)}")
            if rule_docs.get("created") or rule_docs.get("updated"):
                lines.append(
                    "  " + ", ".join(part for part in (
                        f"written {rule_docs['created']}" if rule_docs.get("created") else "",
                        f"last updated {rule_docs['updated']}" if rule_docs.get("updated") else "",
                    ) if part).capitalize()
                )
            if rule_docs.get("rev_in_database"):
                lines.append(f"  Revision in the database: {rule_docs['rev_in_database']}")
            for key, label in (
                ("attack_target", "Attack target"), ("deployment", "Deployment"),
                ("affected_product", "Affected product"), ("malware_family", "Malware family"),
                ("tag", "Tags"), ("former_category", "Former category"),
            ):
                if rule_docs.get(key):
                    lines.append(f"  {label}: {rule_docs[key]}")
            if rule_docs.get("references"):
                lines.append(f"  References: {', '.join(rule_docs['references'])}")
            if rule_docs.get("cves"):
                lines.append(f"  CVEs: {', '.join(rule_docs['cves'])}")
            if rule_docs.get("attack_tactics"):
                lines.append(f"  ATT&CK tactic: {', '.join(rule_docs['attack_tactics'])}")
            if rule_docs.get("attack_techniques"):
                lines.append(f"  ATT&CK technique: {', '.join(rule_docs['attack_techniques'])}")
            continue
        talos = dict(details["talos_documentation"])
        lines += ["", f"Rule {rule}: Talos documentation"]
        # A category's description once, at its first rule.
        name = talos.get("category", "").split(" -- ")[0]
        if name in categories:
            talos["category"] = f"{name} (described above)"
        categories.add(name)
        labels = (
            ("category", "Category"), ("description", "Description"), ("impact", "Impact"),
            ("details", "Details"), ("ease_of_attack", "Ease of attack"),
            ("what_to_look_for", "What to look for"), ("false_positives", "False positives"),
            ("known_usage", "Known usage"),
        )
        for key, label in labels:
            value = talos.get(key)
            # A description Talos copied from a CVE's NVD summary is given there.
            if value and not (key == "description" and value in summaries):
                lines.append(f"  {label}: {value}")
        if talos.get("vulnerability"):
            kind = talos["vulnerability"]
            lines.append(f"  Vulnerability type: {kind.get('type')}"
                         + (f": {kind['description']}" if kind.get("description") else ""))
        if talos.get("cves"):
            lines.append(f"  CVEs: {', '.join(talos['cves'])}")
        if talos.get("attack_techniques"):
            lines.append(f"  ATT&CK technique: {', '.join(talos['attack_techniques'])}")
    for cve, details in found.get("cves", {}).items():
        heading = cve + (f" (published {details['published']})" if details.get("published") else "")
        lines += ["", heading]
        if details.get("summary"):
            lines.append(f"  Summary: {details['summary']}")
        if details.get("cvss"):
            cvss = details["cvss"]
            score = " ".join(str(part) for part in (cvss.get("score"), cvss.get("severity")) if part)
            vector = f" ({cvss['vector']})" if cvss.get("vector") else ""
            lines.append(f"  CVSS {cvss.get('version')}: {score}{vector}")
        if details.get("cwe"):
            lines.append(f"  Weakness: {', '.join(details['cwe'])}")
        if details.get("epss"):
            epss = details["epss"]
            lines.append(f"  EPSS: {epss.get('score')} (percentile {epss.get('percentile')}) on {epss.get('date')}")
        if details.get("cisa_kev"):
            kev = details["cisa_kev"]
            lines.append(
                f"  CISA Known Exploited Vulnerability since {kev.get('date_added')}: "
                f"{kev.get('vulnerability_name', '')}. {kev.get('short_description', '')} "
                f"Required action: {kev.get('required_action', '')} "
                f"Known ransomware use: {kev.get('known_ransomware_campaign_use', 'Unknown')}."
            )
    for technique, details in found.get("attack_techniques", {}).items():
        lines += ["", f"ATT&CK {technique}: {details.get('name', '')}"]
        if details.get("tactics"):
            lines.append(f"  Tactics: {', '.join(details['tactics'])}")
        if details.get("platforms"):
            lines.append(f"  Platforms: {', '.join(details['platforms'])}")
        if details.get("description"):
            lines.append(f"  Description: {details['description']}")
    if found.get("unknown"):
        lines += ["", f"Not in the database: {', '.join(found['unknown'])}"]
    if not (found.get("rules") or found.get("cves") or found.get("attack_techniques")):
        lines += ["", "The database holds nothing on this file's rules: no rule documentation, CVEs or ATT&CK techniques."]
    return "\n".join(lines).strip() + "\n"


def prompt_text(db: Path | str, combined: str | Mapping[str, Any]) -> str:
    """The text to put in the LLM's prompt with a combined file."""
    return as_text(enrich(Path(db), combined))


def download(base_url: str, db: Path, token: str | None) -> Path:
    """The dashboard's intel.db, saved whole at db."""
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", "intel/intel.db")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    part = db.with_name(db.name + ".part")
    db.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as answer:
        total = int(answer.headers.get("Content-Length") or 0)
        done = 0
        with open(part, "wb") as out:
            while True:
                chunk = answer.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done >> 20} / {total >> 20} MB", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if total and done != total:
        os.remove(part)
        raise SystemExit(f"The download stopped at {done} of {total} bytes; nothing was replaced.")
    shutil.move(str(part), str(db))
    return db


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"the database (default: {DEFAULT_DB})")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="fetch intel.db from the dashboard")
    fetch.add_argument("url", help="the dashboard, such as http://172.16.20.142:8501/")
    fetch.add_argument("--token", default=os.environ.get("IDS_API_TOKEN") or None)
    file_command = commands.add_parser("enrich", help="what the database says of a combined JSON file")
    file_command.add_argument("file", type=Path)
    file_command.add_argument("--json", action="store_true", help="print JSON instead of text")
    ids_command = commands.add_parser("lookup", help="what the database says of rule names, CVEs and techniques")
    ids_command.add_argument("ids", nargs="+")
    ids_command.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    if args.command == "download":
        saved = download(args.url, args.db, args.token)
        print(f"Saved {saved} ({saved.stat().st_size >> 20} MB)")
        return
    if args.command == "enrich":
        found = enrich(args.db, args.file.read_text(encoding="utf-8"))
    else:
        found = lookup(args.db, args.ids)
    print(json.dumps(found, ensure_ascii=False, indent=2) if args.json else as_text(found), end="" if not args.json else "\n")


if __name__ == "__main__":
    main()

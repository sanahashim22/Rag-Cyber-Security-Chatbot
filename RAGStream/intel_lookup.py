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
  built-in rule (GID 2 and up) Snort's own description of it;
- a CVE: the NVD summary, publication date, CVSS, CWEs, EPSS and, when it is
  in CISA's Known Exploited Vulnerabilities catalog, its KEV entry;
- a technique: its name, tactics, platforms and description.

Copy this one file to the LLM PC (Windows or Linux; Python 3.8 or newer, no
packages) and run it there:

    python intel_lookup.py download http://172.16.20.142:8501/
    python intel_lookup.py enrich snort_suricata_combined_20261117_1651_last20.json
    python intel_lookup.py lookup snort_signature:1:31978:5 CVE-2014-6271 T1190

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
the CLI). Re-copy it from there if their schema ever changes.
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


def rule_details(connection: sqlite3.Connection, rule: str) -> dict[str, Any] | None:
    """Talos's documentation of a Snort rule, or Snort's own description of
    a built-in rule Talos doesn't document; None for a Suricata rule (ET
    Open has none) or a rule neither documents."""
    match = _RULE_RE.fullmatch(rule)
    if not match or match.group(1) != "snort":
        return None
    gid, sid = int(match.group(2)), int(match.group(3))
    row = connection.execute(
        "SELECT * FROM talos_rule_docs WHERE gid = ? AND sid = ?", (gid, sid),
    ).fetchone() if _has_table(connection, "talos_rule_docs") else None
    if row is None or not _one_line(row["explanation"]) or gid > 3:
        builtin = connection.execute(
            "SELECT description FROM snort_builtin_docs WHERE gid = ? AND sid = ?", (gid, sid),
        ).fetchone() if _has_table(connection, "snort_builtin_docs") else None
        if builtin is not None:
            return {"snort_documentation": {"description": builtin["description"]}}
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
            if _RULE_RE.fullmatch(given):
                details = rule_details(connection, given)
                if details is None:
                    continue
                found["rules"][given] = details
                talos = details.get("talos_documentation", {})
                cves += talos.get("cves", [])
                techniques += talos.get("attack_techniques", [])
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
            lines += ["", f"Rule {rule}: Snort's description of this built-in rule",
                      f"  Description: {details['snort_documentation']['description']}"]
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

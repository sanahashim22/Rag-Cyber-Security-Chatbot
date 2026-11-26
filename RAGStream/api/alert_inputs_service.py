"""
alert_inputs_service.py
------------------------
Alert-folder scanning, IDS-Dashboard pull, and intel.db status/download -
ported from dashboard_app.py's own helpers (_scan_alert_inputs,
_pull_from_ids_dashboard, _get_remote_intel_info, _get_local_intel_status,
_download_intel_db), with the one real behavior change: _download_intel_db
reports progress into a job instead of an st.progress bar, since it's
called from a background job here rather than inline in a Streamlit script.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

import snort_ranking

ALERT_INPUTS_DIR = Path(__file__).resolve().parent.parent / "alert_inputs"
ALERT_INPUTS_DIR.mkdir(exist_ok=True)

# The exact strings IDS-Dashboard's GET /export accepts for its `length`
# query param - from its own src/ids/combined_detections.py WINDOW_LENGTHS.
IDS_DASHBOARD_WINDOW_LENGTHS = [
    "5 minutes", "10 minutes", "15 minutes", "30 minutes",
    "1 hour", "3 hours", "6 hours", "12 hours", "1 day", "7 days",
]


def _read_alert_files() -> tuple[Dict[str, List[Dict]], List[str], str]:
    """Shared scan pass: parses every *.json under ALERT_INPUTS_DIR. Returns
    (entries_by_file, warnings, combined_fhash)."""
    alert_files = sorted(ALERT_INPUTS_DIR.glob("*.json"))
    entries_by_file: Dict[str, List[Dict]] = {}
    warnings: List[str] = []
    combined_hasher = hashlib.md5()

    for path in alert_files:
        raw_bytes = path.read_bytes()
        combined_hasher.update(path.name.encode("utf-8"))
        combined_hasher.update(raw_bytes)

        try:
            data = json.loads(raw_bytes.decode("utf-8", errors="ignore"))
        except Exception as e:
            warnings.append(f"Skipping `{path.name}` - could not parse as JSON: {e}")
            continue

        if not snort_ranking.is_snort_alerts_payload(data):
            warnings.append(
                f"Skipping `{path.name}` - doesn't look like a Snort alerts file (expected fields like "
                "`sid`, `priority`, or an `alerts` list)."
            )
            continue

        file_entries = snort_ranking.extract_alert_entries(data)
        if not file_entries:
            warnings.append(f"Skipping `{path.name}` - recognized as a Snort alerts file but contained no alerts.")
            continue

        entries_by_file[path.name] = file_entries

    return entries_by_file, warnings, combined_hasher.hexdigest()


def scan_alert_inputs() -> Dict[str, Any]:
    """Lightweight status only (file names/counts, no alert bodies) - safe
    to poll every few seconds from the frontend, same role as
    dashboard_app.py's st.fragment(run_every=4) status banner. Fetch the
    actual alert bodies separately via get_alert_entries(), only once (when
    a file is picked / "Run ranking" is clicked), not on every poll."""
    entries_by_file, warnings, fhash = _read_alert_files()
    total_parsed = sum(len(v) for v in entries_by_file.values())
    return {
        "file_summaries": [{"name": n, "count": len(v)} for n, v in entries_by_file.items()],
        "warnings": warnings,
        "fhash": fhash,
        "total_parsed": total_parsed,
        "alert_inputs_dir": str(ALERT_INPUTS_DIR),
    }


def get_alert_entries(file_name: str | None = None) -> Dict[str, Any]:
    """The actual alert entries - for a specific file, or all files
    combined when file_name is None/"All files (combined)". Called once
    when the user picks a file or clicks "Run ranking", not on every poll."""
    entries_by_file, warnings, fhash = _read_alert_files()
    if file_name and file_name in entries_by_file:
        entries = entries_by_file[file_name]
    else:
        entries = [e for file_entries in entries_by_file.values() for e in file_entries]
    return {"entries": entries, "warnings": warnings, "fhash": f"{fhash}::{file_name or 'all'}"}


def _safe_alert_filename(raw_name: str) -> str:
    """Basename only (blocks path traversal), non-filename-safe characters
    replaced, guaranteed .json extension - same policy as
    alert_receiver.py's _safe_filename."""
    name = Path(raw_name).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "received"
    if not name.lower().endswith(".json"):
        name += ".json"
    return name


def _unique_alert_inputs_path(name: str) -> Path:
    candidate = ALERT_INPUTS_DIR / name
    if not candidate.exists():
        return candidate
    stem, ext = name.rsplit(".", 1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ALERT_INPUTS_DIR / f"{stem}_{stamp}.{ext}"


def pull_from_ids_dashboard(base_url: str, token: str, *, length: str | None = None,
                             last_n: int | None = None) -> tuple[bool, str]:
    """Calls IDS-Dashboard's GET /export (time window) or GET
    /export/selection/file (count-based). Exactly one of length/last_n
    should be given. Saves the response into ALERT_INPUTS_DIR. Never
    raises - returns (success, message)."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    label = f"last {last_n} alert(s)" if last_n is not None else f"the last {length}"

    if last_n is not None:
        url = base_url.rstrip("/") + "/export/selection/file"
        params = {"last": last_n}
    else:
        url = base_url.rstrip("/") + "/export"
        params = {"length": length}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=120)
    except requests.RequestException as e:
        return False, f"Could not reach {url}: {e}"

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text[:300])
        except Exception:
            detail = resp.text[:300]
        return False, f"IDS-Dashboard returned {resp.status_code}: {detail}"

    content_disp = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename="?([^";]+)"?', content_disp)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"last{last_n}" if last_n is not None else length.replace(" ", "_")
    fallback_name = f"ids_dashboard_{tag}_{stamp}.json"
    dest = _unique_alert_inputs_path(_safe_alert_filename(m.group(1) if m else fallback_name))
    dest.write_bytes(resp.content)

    n_snort = resp.headers.get("X-Alerts-Snort", "?")
    n_suricata = resp.headers.get("X-Alerts-Suricata", "?")
    return True, (
        f"Saved `{dest.name}` ({len(resp.content)} bytes) - "
        f"{n_snort} Snort + {n_suricata} Suricata alert(s) for {label}."
    )


def get_remote_intel_info(base_url: str, token: str) -> tuple[dict | None, str]:
    """GET /intel - IDS-Dashboard's intel.db size/modified date/source
    dates, without downloading the file itself."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = requests.get(base_url.rstrip("/") + "/intel", headers=headers, timeout=15)
    except requests.RequestException as e:
        return None, f"Could not reach {base_url}: {e}"
    if resp.status_code != 200:
        try:
            detail = resp.json().get("error", resp.text[:300])
        except Exception:
            detail = resp.text[:300]
        return None, f"IDS-Dashboard returned {resp.status_code}: {detail}"
    try:
        return resp.json(), ""
    except Exception as e:
        return None, f"Unexpected response: {e}"


def get_local_intel_status() -> Dict[str, Any] | None:
    """Local intel.db's size/modified date/source dates - no network call.
    None if it hasn't been downloaded yet."""
    path = snort_ranking.INTEL_DB_PATH
    if not path.is_file():
        return None
    stat = path.stat()
    sources: Dict[str, str] = {}
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)) as con:
            if con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_metadata'"
            ).fetchone():
                for source, as_of, imported in con.execute(
                    "SELECT source, data_as_of, imported_at FROM source_metadata"
                ):
                    sources[source] = (as_of or imported or "")[:10]
    except sqlite3.Error:
        pass
    return {
        "bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
    }


_DOWNLOAD_MAX_ATTEMPTS = 6
_DOWNLOAD_RETRY_BACKOFF_SEC = 4


def download_intel_db(job_id: str, base_url: str, token: str) -> Dict[str, Any]:
    """Streams IDS-Dashboard's /intel/intel.db into snort_ranking.INTEL_DB_PATH,
    atomically (temp file + rename), reporting progress into the job.

    A ~1.5GB transfer over a real network has a real chance of the
    connection dropping partway through (IncompleteRead / ChunkedEncoding
    errors) before ever reaching the last byte - dashboard_app.py's original
    version treated that as fatal and made the user start the whole download
    over by hand. This instead retries a few times and, if the server
    supports HTTP Range requests (checked via a 206 response), resumes from
    the last byte actually written instead of re-downloading what already
    succeeded. If the server doesn't support ranges (responds 200 to a
    range request), it just falls back to restarting that one attempt from
    scratch rather than corrupting the file."""
    from . import jobs

    dest = snort_ranking.INTEL_DB_PATH
    part = dest.with_name(dest.name + ".part")
    headers_base = {"Authorization": f"Bearer {token}"} if token else {}
    url = base_url.rstrip("/") + "/intel/intel.db"

    done = part.stat().st_size if part.is_file() else 0
    total = 0
    last_error = ""

    for attempt in range(1, _DOWNLOAD_MAX_ATTEMPTS + 1):
        resuming = done > 0
        headers = dict(headers_base)
        if resuming:
            headers["Range"] = f"bytes={done}-"
        retry_note = f" (retry {attempt - 1}/{_DOWNLOAD_MAX_ATTEMPTS - 1})" if attempt > 1 else ""
        try:
            with requests.get(url, headers=headers, timeout=60, stream=True) as resp:
                if resp.status_code not in (200, 206):
                    try:
                        detail = resp.json().get("error", resp.text[:300])
                    except Exception:
                        detail = resp.text[:300]
                    part.unlink(missing_ok=True)
                    return {"ok": False, "message": f"IDS-Dashboard returned {resp.status_code}: {detail}"}

                # Range accepted (206) -> append from where we left off.
                # Range ignored (200 while resuming) -> server doesn't
                # support it, so the body is the WHOLE file again - start
                # this attempt's write over from scratch.
                if resuming and resp.status_code == 200:
                    done = 0
                mode = "ab" if (resuming and resp.status_code == 206) else "wb"

                content_length = int(resp.headers.get("Content-Length") or 0)
                total = (done + content_length) if resp.status_code == 206 else content_length

                with open(part, mode) as out:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            jobs.update_progress(
                                job_id, f"Downloading intel.db... {done >> 20}/{total >> 20} MB{retry_note}",
                                current=done, total=total,
                            )
            if total and done != total:
                last_error = f"connection dropped at {done >> 20}/{total >> 20} MB"
                if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                    time.sleep(_DOWNLOAD_RETRY_BACKOFF_SEC)
                continue
            part.replace(dest)
            recovered = f" (recovered after {attempt} attempt(s))" if attempt > 1 else ""
            return {"ok": True, "message": f"Saved intel.db ({done >> 20} MB).{recovered}"}
        except requests.RequestException as e:
            last_error = str(e)
            done = part.stat().st_size if part.is_file() else done
            if attempt < _DOWNLOAD_MAX_ATTEMPTS:
                time.sleep(_DOWNLOAD_RETRY_BACKOFF_SEC)
            continue

    part.unlink(missing_ok=True)
    return {
        "ok": False,
        "message": f"Could not reach {url} after {_DOWNLOAD_MAX_ATTEMPTS} attempts: {last_error}",
    }

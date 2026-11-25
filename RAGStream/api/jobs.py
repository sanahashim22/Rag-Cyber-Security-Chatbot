"""
jobs.py
-------
Minimal in-process background job manager. Ranking runs and the intel.db
download are long (seconds to minutes), so every such action starts a job
in a thread-pool worker and returns a job_id immediately; the frontend polls
GET /.../jobs/{id} for progress/result - the same role Streamlit's own
st.progress bar + session_state played in dashboard_app.py, just over HTTP.

In-memory only (a dict), same limitation dashboard_app.py's own
st.session_state already has: jobs don't survive a backend restart.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="job")
_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def start_job(fn: Callable[[str], Any]) -> str:
    """Runs fn(job_id) in a background thread. fn should call
    update_progress(job_id, ...) as it goes and return the final,
    JSON-serializable result."""
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "progress": {"message": "Starting...", "current": 0, "total": 0},
            "result": None,
            "error": None,
            "created_at": time.time(),
        }

    def _runner():
        try:
            result = fn(job_id)
            with _lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
        except Exception as e:
            with _lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = f"{e}\n{traceback.format_exc()}"

    _executor.submit(_runner)
    return job_id


def update_progress(job_id: str, message: str, current: int = 0, total: int = 0) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"] = {"message": message, "current": current, "total": total}


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None

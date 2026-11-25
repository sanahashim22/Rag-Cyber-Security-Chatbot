from fastapi import APIRouter

from . import alert_inputs_service as svc
from .schemas import PullAlertsRequest

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/scan")
def scan():
    """Polled every few seconds by the frontend - same role as
    dashboard_app.py's st.fragment(run_every=4) alert_inputs/ watcher.
    Lightweight: file names/counts only, no alert bodies - see
    /entries for those."""
    return svc.scan_alert_inputs()


@router.get("/entries")
def entries(file: str | None = None):
    """The actual alert entries for one file, or all combined when `file`
    is omitted - fetched once (file picked / Run ranking clicked), not on
    every poll."""
    return svc.get_alert_entries(file)


@router.post("/pull")
def pull(req: PullAlertsRequest):
    ok, message = svc.pull_from_ids_dashboard(
        req.base_url, req.token, length=req.length, last_n=req.last_n,
    )
    return {"ok": ok, "message": message}


@router.get("/ids-dashboard-window-lengths")
def window_lengths():
    return {"lengths": svc.IDS_DASHBOARD_WINDOW_LENGTHS}

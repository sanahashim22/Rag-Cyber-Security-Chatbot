from fastapi import APIRouter, HTTPException

from . import alert_inputs_service as svc
from . import jobs
from .schemas import IntelRemoteRequest

router = APIRouter(prefix="/api/intel", tags=["intel"])


@router.get("/local")
def local_status():
    status = svc.get_local_intel_status()
    return {"downloaded": status is not None, "status": status}


@router.post("/remote")
def remote_status(req: IntelRemoteRequest):
    info, err = svc.get_remote_intel_info(req.base_url, req.token)
    if info is None:
        return {"ok": False, "error": err}
    return {"ok": True, "info": info}


@router.post("/download")
def start_download(req: IntelRemoteRequest):
    job_id = jobs.start_job(lambda job_id: svc.download_intel_db(job_id, req.base_url, req.token))
    return {"job_id": job_id}


@router.get("/download/jobs/{job_id}")
def download_job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job

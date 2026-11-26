from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

import snort_ranking
from . import jobs, ranking_service, state
from .schemas import RunComparisonRequest, RunRankingRequest

router = APIRouter(prefix="/api/ranking", tags=["ranking"])


@router.get("/models")
def models():
    return {"models": ranking_service.RANKING_MODELS, "auto_run_limit": ranking_service.AUTO_RUN_LIMIT}


def _single_model_result(entries: List[Dict], model: str, run_judge: bool, run_mismatch: bool,
                          run_correlation: bool, repeats: int, job_id: str) -> Dict[str, Any]:
    processed = ranking_service.run_ranking(
        job_id, entries, model, state.rag_system, run_judge=run_judge, run_mismatch=run_mismatch,
        repeats=repeats,
    )
    correlation = ranking_service.run_correlation_analysis(
        entries, processed, model, run_correlation, job_id=job_id,
    )
    groups = (correlation or {}).get("groups") or []
    return {
        "kind": "single",
        "model": model,
        "processed": processed,
        "correlation": correlation,
        "stats": ranking_service.aggregate_stats(processed),
        "alert_table": ranking_service.build_alert_table(processed),
        "correlation_table": ranking_service.build_correlation_table(groups),
        "report_markdown": snort_ranking.build_report_markdown(processed, model),
        "timing_text": ranking_service.build_timing_report_text(processed, model, correlation=correlation),
    }


@router.post("/run")
def run(req: RunRankingRequest):
    if state.rag_system is None:
        raise HTTPException(503, "RAG system is still starting up - try again shortly.")

    def _job(job_id: str):
        return _single_model_result(
            req.entries, req.model, req.run_judge, req.run_mismatch, req.run_correlation,
            max(1, req.repeats), job_id,
        )

    job_id = jobs.start_job(_job)
    return {"job_id": job_id}


@router.post("/compare")
def compare(req: RunComparisonRequest):
    if state.rag_system is None:
        raise HTTPException(503, "RAG system is still starting up - try again shortly.")

    def _job(job_id: str) -> Dict[str, Any]:
        results: Dict[str, List[Dict]] = {}
        correlations: Dict[str, Optional[Dict]] = {}
        per_model: Dict[str, Any] = {}

        for model in ranking_service.RANKING_MODELS:
            processed = ranking_service.run_ranking(
                job_id, req.entries, model, state.rag_system, label_prefix=f"[{model}] ",
                run_judge=req.run_judge, run_mismatch=req.run_mismatch, repeats=max(1, req.repeats),
            )
            results[model] = processed
            correlation = ranking_service.run_correlation_analysis(
                req.entries, processed, model, req.run_correlation,
                job_id=job_id, label_prefix=f"[{model}] ",
            )
            correlations[model] = correlation
            groups = (correlation or {}).get("groups") or []
            per_model[model] = {
                "processed": processed,
                "correlation": correlation,
                "stats": ranking_service.aggregate_stats(processed),
                "alert_table": ranking_service.build_alert_table(processed),
                "correlation_table": ranking_service.build_correlation_table(groups),
                "report_markdown": snort_ranking.build_report_markdown(processed, model),
                "timing_text": ranking_service.build_timing_report_text(processed, model, correlation=correlation),
            }

        comparison_rows = []
        for idx in range(len(req.entries)):
            base_item = results[ranking_service.RANKING_MODELS[0]][idx]
            alert = base_item["alert_entry"].get("alert", {}) or {}
            row: Dict[str, Any] = {
                "alert_id": base_item["alert_entry"].get("alert_id", "?"),
                "source": alert.get("engine", "snort").capitalize(),
                "alert_message": alert.get("alert_message", "(no message)"),
                "actual_ranking": alert.get("priority", "?"),
                "sid": alert.get("sid", "?"),
                "models": {},
            }
            for model in ranking_service.RANKING_MODELS:
                item = results[model][idx]
                judge_scores = [r.get("judge_score") for r in item["runs"] if r.get("judge_score") is not None]
                avg_judge = (sum(judge_scores) / len(judge_scores)) if judge_scores else None
                row["models"][model] = {
                    "majority_rank": ranking_service.mode_rank(item["ranks"]),
                    "ranks": item["ranks"],
                    "avg_judge_score": round(avg_judge, 2) if avg_judge is not None else None,
                }
            comparison_rows.append(row)

        model_summary = [
            {"model": model, **ranking_service.aggregate_stats(results[model])}
            for model in ranking_service.RANKING_MODELS
        ]

        excel_bytes = ranking_service.build_comparison_excel(req.entries, results)

        return {
            "kind": "comparison",
            "models": ranking_service.RANKING_MODELS,
            "per_model": per_model,
            "comparison_rows": comparison_rows,
            "model_summary": model_summary,
            "timing_text": ranking_service.build_comparison_timing_report_text(
                req.entries, results, correlations=correlations,
            ),
            "excel_base64": base64.b64encode(excel_bytes).decode("ascii"),
        }

    job_id = jobs.start_job(_job)
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job

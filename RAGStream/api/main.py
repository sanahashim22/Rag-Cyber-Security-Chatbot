"""
api/main.py
-----------
FastAPI backend for the React dashboard. Run from RAGStream/:

    uvicorn api.main:app --reload --port 8502

Serves the API under /api/* and, once built, the React production bundle
(frontend/dist/) as static files at / - same-origin, so one port (and one
ngrok tunnel) covers both, with no CORS needed in that mode. Vite's own dev
server (npm run dev, proxying /api to this port) is used for local
development instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import routes_alerts, routes_chat, routes_intel, routes_ranking, state

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Snort Alert Severity Dashboard API")

# Allows the Vite dev server (a different origin/port) to call this API
# during local development. Harmless in production too, since the built
# frontend is served same-origin and never needs CORS at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_alerts.router)
app.include_router(routes_intel.router)
app.include_router(routes_ranking.router)
app.include_router(routes_chat.router)


@app.on_event("startup")
def _startup():
    state.build_rag_system()


@app.get("/api/health")
def health():
    return {"ok": True, "rag_system_ready": state.rag_system is not None}


@app.get("/api/meta")
def meta():
    import rag_core
    import snort_ranking

    is_local_ollama = rag_core.OLLAMA_HOST.startswith("http://localhost") or rag_core.OLLAMA_HOST.startswith("http://127.0.0.1")
    return {
        "ollama_host": rag_core.OLLAMA_HOST,
        "ollama_is_local": is_local_ollama,
        "doc_names": state.doc_names,
        "judge_model": snort_ranking.JUDGE_MODEL,
        "severity_meanings": snort_ranking.SEVERITY_MEANINGS,
    }


_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

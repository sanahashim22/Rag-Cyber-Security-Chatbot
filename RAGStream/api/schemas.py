"""Pydantic request models. Responses are returned as plain dicts (FastAPI
JSON-encodes them automatically) rather than hand-written response models -
the ranking result shapes come straight from snort_ranking.py's own
JSON-safe dicts, so a parallel Pydantic schema would just be a second,
harder-to-keep-in-sync copy of the same contract."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RunRankingRequest(BaseModel):
    entries: List[Dict[str, Any]]
    model: str
    run_judge: bool = True
    run_mismatch: bool = True
    run_correlation: bool = True
    repeats: int = 1


class RunComparisonRequest(BaseModel):
    entries: List[Dict[str, Any]]
    run_judge: bool = True
    run_mismatch: bool = True
    run_correlation: bool = True
    repeats: int = 1


class PullAlertsRequest(BaseModel):
    base_url: str
    token: str = ""
    length: Optional[str] = None
    last_n: Optional[int] = None


class IntelRemoteRequest(BaseModel):
    base_url: str
    token: str = ""


class ChatMessageRequest(BaseModel):
    question: str
    model: str

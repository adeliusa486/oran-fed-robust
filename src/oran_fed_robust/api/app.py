"""FastAPI service exposing the aggregation defense as a microservice.

This models how the aggregator would sit inside a near-RT RIC: clients POST their
flat update vectors and the service returns the aggregated update plus, for the
reputation rule, the current per-client trust scores (for auditing).

Run: uvicorn oran_fed_robust.api.app:app --reload
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..aggregation import build_aggregator
from ..aggregation.base import AggregationContext
from ..aggregation.reputation import ReputationAggregator

app = FastAPI(title="ORAN Federated Robust Aggregator", version="0.1.0")

# A single stateful aggregator instance (reputation persists across requests).
_AGG = build_aggregator("reputation", beta=0.8)


class AggregateRequest(BaseModel):
    updates: List[List[float]] = Field(..., description="List of flat update vectors")
    client_ids: Optional[List[int]] = Field(None, description="Stable client ids")
    aggregator: str = Field("reputation", description="Aggregator name")
    n_byzantine: int = Field(1, ge=0)


class AggregateResponse(BaseModel):
    aggregated: List[float]
    reputations: Optional[dict] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "aggregator": _AGG.name}


@app.post("/aggregate", response_model=AggregateResponse)
def aggregate(req: AggregateRequest) -> AggregateResponse:
    global _AGG
    if not req.updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    arr = np.asarray(req.updates, dtype=float)
    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="updates must be a 2D array")

    if req.aggregator != _AGG.name:
        _AGG = build_aggregator(req.aggregator, beta=0.8)

    ctx = AggregationContext(client_ids=req.client_ids, n_byzantine=req.n_byzantine)
    try:
        out = _AGG.aggregate(arr, ctx)
    except Exception as exc:  # surface aggregation errors cleanly
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    reps = _AGG.reputations() if isinstance(_AGG, ReputationAggregator) else None
    return AggregateResponse(aggregated=out.tolist(), reputations=reps)

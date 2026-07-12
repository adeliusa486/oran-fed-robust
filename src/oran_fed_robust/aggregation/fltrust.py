"""FLTrust aggregation (Cao et al., 2021).

Reweights client updates by ReLU(cosine similarity) to a trusted server update
computed on a small clean root dataset, then normalizes magnitudes to the
server update's norm.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Aggregator, AggregationContext


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


class FLTrust(Aggregator):
    name = "fltrust"

    def __init__(self, **_):
        pass

    def aggregate(self, updates: np.ndarray, ctx: Optional[AggregationContext] = None) -> np.ndarray:
        if ctx is None or ctx.server_update is None:
            # Fallback: without a root reference FLTrust reduces to the mean.
            return updates.mean(axis=0)
        g0 = ctx.server_update
        g0_norm = np.linalg.norm(g0) + 1e-12
        weights = np.array([max(0.0, _cosine(u, g0)) for u in updates])
        if weights.sum() <= 0:
            return np.zeros_like(g0)
        # rescale each update to the server update norm, then trust-weight
        rescaled = np.array([u / (np.linalg.norm(u) + 1e-12) * g0_norm for u in updates])
        return (weights[:, None] * rescaled).sum(axis=0) / weights.sum()

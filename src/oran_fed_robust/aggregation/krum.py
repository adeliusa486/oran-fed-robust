"""Krum aggregation (Blanchard et al., 2017).

Selects the single update minimizing the sum of squared distances to its
n - f - 2 nearest neighbours, where f is the assumed number of Byzantine clients.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Aggregator, AggregationContext


class Krum(Aggregator):
    name = "krum"

    def __init__(self, **_):
        pass

    def aggregate(self, updates: np.ndarray, ctx: Optional[AggregationContext] = None) -> np.ndarray:
        n = updates.shape[0]
        f = ctx.n_byzantine if ctx is not None else max(1, n // 5)
        # number of neighbours considered
        m = max(1, n - f - 2)
        # pairwise squared distances
        sq = np.sum(updates**2, axis=1)
        dists = sq[:, None] + sq[None, :] - 2 * updates @ updates.T
        np.fill_diagonal(dists, np.inf)
        scores = np.sort(dists, axis=1)[:, :m].sum(axis=1)
        best = int(np.argmin(scores))
        return updates[best]

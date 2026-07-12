"""Coordinate-wise trimmed-mean aggregation (Yin et al., 2018)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Aggregator, AggregationContext


class TrimmedMean(Aggregator):
    name = "trimmed_mean"

    def __init__(self, trim_ratio: float = 0.1, **_):
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio

    def aggregate(self, updates: np.ndarray, ctx: Optional[AggregationContext] = None) -> np.ndarray:
        n = updates.shape[0]
        k = int(self.trim_ratio * n)
        if k == 0:
            return updates.mean(axis=0)
        sorted_u = np.sort(updates, axis=0)
        trimmed = sorted_u[k : n - k]
        return trimmed.mean(axis=0)

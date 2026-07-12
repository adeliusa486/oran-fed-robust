"""Coordinate-wise median aggregation (Yin et al., 2018)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Aggregator, AggregationContext


class CoordinateMedian(Aggregator):
    name = "median"

    def __init__(self, **_):
        pass

    def aggregate(self, updates: np.ndarray, ctx: Optional[AggregationContext] = None) -> np.ndarray:
        return np.median(updates, axis=0)

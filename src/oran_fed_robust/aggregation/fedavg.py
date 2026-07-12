"""FedAvg: unweighted mean (no defense baseline)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Aggregator, AggregationContext


class FedAvg(Aggregator):
    name = "fedavg"

    def __init__(self, **_):
        # tolerate shared kwargs (e.g. beta, trim_ratio) passed by build_aggregator
        pass

    def aggregate(self, updates: np.ndarray, ctx: Optional[AggregationContext] = None) -> np.ndarray:
        return updates.mean(axis=0)

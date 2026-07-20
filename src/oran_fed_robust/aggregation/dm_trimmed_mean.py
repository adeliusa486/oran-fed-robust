"""Direction-and-magnitude-aware trimmed mean (DM-TM).

Two-stage robust aggregator, proposed to close the blind spot identified in
the clipping ablation (Section: Comparative and Ablation Analysis): rules
built on magnitude/distance statistics (Krum, coordinate-median) cannot see
an attack that keeps its update norm within the benign range but points the
update in the wrong direction (IPM).

Stage 1 (direction filter): drop the `direction_trim_ratio` fraction of
client updates least aligned -- by cosine similarity -- with a robust
reference direction (the coordinate-wise median of all updates that round).
This targets IPM directly: an IPM update is a negatively-scaled mean of
honest updates, so its cosine similarity to the honest consensus direction
is strongly negative, and it is filtered before any magnitude step runs.

Stage 2 (magnitude filter): apply ordinary coordinate-wise trimmed-mean
(`trim_ratio`) to the surviving updates, to absorb large-magnitude attacks
(sign-flip, fabricated-update injection) among the direction-filtered set.

No server-side clean reference set is required (unlike FLTrust) and the rule
is stateless across rounds (unlike the reputation aggregator), which isolates
the marginal value of a single direction-aware step layered on top of the
existing order-statistic family.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Aggregator, AggregationContext


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


class DirectionMagnitudeTrimmedMean(Aggregator):
    name = "dm_trimmed_mean"

    def __init__(self, trim_ratio: float = 0.1, direction_trim_ratio: float = 0.2, **_):
        if not 0.0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be in [0, 0.5)")
        if not 0.0 <= direction_trim_ratio < 0.5:
            raise ValueError("direction_trim_ratio must be in [0, 0.5)")
        self.trim_ratio = trim_ratio
        self.direction_trim_ratio = direction_trim_ratio

    def aggregate(self, updates: np.ndarray, ctx: Optional[AggregationContext] = None) -> np.ndarray:
        n = updates.shape[0]
        reference = np.median(updates, axis=0)

        # Stage 1: direction filter.
        k_dir = int(self.direction_trim_ratio * n)
        if k_dir > 0:
            sims = np.array([_cosine_similarity(updates[i], reference) for i in range(n)])
            keep_idx = np.argsort(sims)[k_dir:]  # drop the k_dir lowest-similarity clients
            survivors = updates[keep_idx]
        else:
            survivors = updates

        # Stage 2: magnitude filter (coordinate-wise trimmed-mean) on survivors.
        m = survivors.shape[0]
        k_mag = int(self.trim_ratio * n)
        if k_mag == 0 or m - 2 * k_mag <= 0:
            k_mag = int(self.trim_ratio * m)
        if k_mag == 0 or m - 2 * k_mag <= 0:
            return survivors.mean(axis=0)
        sorted_u = np.sort(survivors, axis=0)
        trimmed = sorted_u[k_mag: m - k_mag]
        return trimmed.mean(axis=0)

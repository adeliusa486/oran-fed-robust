"""Proposed: heterogeneity-aware reputation-weighted aggregation.

Implements Eqs. (1)-(3) of the manuscript:
  * divergence d_i = cosine distance to the coordinate-median reference,
  * reputation r_i = beta*r_i + (1-beta)*(1 - phi(d_i))  (EWMA over rounds),
  * trust weights w_i = relu(r_i) / sum_j relu(r_j),
  * aggregate = sum_i w_i * update_i.

The key property is *temporal evidence accumulation*: state is keyed by client id
and persists across rounds, so a single round of honest non-IID divergence does
not condemn a client -- only persistent anomalies erode reputation.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .base import Aggregator, AggregationContext


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return 1.0 - float(np.dot(a, b) / denom)


def _phi(d: float) -> float:
    """Calibrated divergence -> suspicion in [0, 1].

    Logistic map centered at the neutral cosine distance of 1.0 (orthogonal).
    Updates aligned with consensus (d<1) yield low suspicion; anti-aligned
    updates (d>1, e.g. sign-flips) yield high suspicion.
    """
    return float(1.0 / (1.0 + np.exp(-4.0 * (d - 1.0))))


class ReputationAggregator(Aggregator):
    name = "reputation"

    def __init__(self, beta: float = 0.8, init_reputation: float = 0.5, **_):
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        self.beta = beta
        self.init_reputation = init_reputation
        self._reputation: Dict[int, float] = {}

    def reputations(self) -> Dict[int, float]:
        """Expose current reputation state (for auditing / governance)."""
        return dict(self._reputation)

    def aggregate(self, updates: np.ndarray, ctx: Optional[AggregationContext] = None) -> np.ndarray:
        n = updates.shape[0]
        client_ids = (
            ctx.client_ids if (ctx is not None and ctx.client_ids is not None) else list(range(n))
        )
        if len(client_ids) != n:
            raise ValueError("client_ids length must match number of updates")

        reference = np.median(updates, axis=0)  # poisoning-resistant anchor

        # Magnitude defense: clip each update's norm to the median norm. Cosine
        # reputation controls *direction*; norm-clipping removes the *magnitude*
        # channel that lets sign-flip / fabricated updates dominate a weighted
        # average even at low weight. Together they give both heterogeneity
        # tolerance (soft, history-aware weighting) and Byzantine robustness.
        norms = np.linalg.norm(updates, axis=1)
        ref_norm = float(np.median(norms))
        clipped = updates.copy()
        for i in range(n):
            if norms[i] > ref_norm and norms[i] > 0:
                clipped[i] = updates[i] * (ref_norm / norms[i])

        weights = np.zeros(n)
        for i, cid in enumerate(client_ids):
            d = _cosine_distance(updates[i], reference)
            prev = self._reputation.get(cid, self.init_reputation)
            r = self.beta * prev + (1.0 - self.beta) * (1.0 - _phi(d))
            self._reputation[cid] = r
            weights[i] = max(r, 0.0)

        if weights.sum() <= 0:
            return reference  # safe fallback
        weights /= weights.sum()
        return (weights[:, None] * clipped).sum(axis=0)

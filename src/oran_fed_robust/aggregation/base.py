"""Aggregator interface and shared context."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class AggregationContext:
    """Side information some aggregators need.

    Attributes:
        client_ids: stable ids of the participating clients (for stateful rules).
        server_update: a trusted reference update (FLTrust).
        n_byzantine: assumed upper bound on malicious clients (Krum).
    """

    client_ids: Optional[List[int]] = None
    server_update: Optional[np.ndarray] = None
    n_byzantine: int = 0
    extra: dict = field(default_factory=dict)


class Aggregator:
    """Base class: combine a stack of client updates into one update vector."""

    name: str = "base"

    def aggregate(self, updates: np.ndarray, ctx: Optional[AggregationContext] = None) -> np.ndarray:
        """Args:
            updates: array of shape [n_clients, dim].
            ctx: optional :class:`AggregationContext`.

        Returns:
            Aggregated update vector of shape [dim].
        """
        raise NotImplementedError

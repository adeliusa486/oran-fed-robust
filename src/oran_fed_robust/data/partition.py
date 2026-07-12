"""Non-IID partitioning utilities.

The Dirichlet label partition is the standard mechanism for inducing
base-station-level heterogeneity: each client receives class proportions drawn
from Dir(alpha). Small alpha => strong heterogeneity (skewed per-client labels).
"""
from __future__ import annotations

from typing import List

import numpy as np


def dirichlet_label_partition(
    labels: np.ndarray, n_clients: int, alpha: float, rng: np.random.Generator
) -> List[np.ndarray]:
    """Partition sample indices across clients using a Dirichlet label prior.

    Args:
        labels: integer label array of shape [N].
        n_clients: number of clients (base stations).
        alpha: Dirichlet concentration; smaller => more skew.
        rng: numpy Generator for reproducibility.

    Returns:
        List of index arrays, one per client.
    """
    n_classes = int(labels.max()) + 1
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))
        # cumulative split points
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(idx_c, cuts)):
            client_indices[client_id].extend(chunk.tolist())
    return [np.array(sorted(ix), dtype=int) for ix in client_indices]

"""Synthetic Open-RAN-style KPM dataset generator.

Each class is a Gaussian blob in feature space, loosely modeling distinct
traffic/QoS regimes observable in E2 key performance measurements (KPMs).
Heterogeneity across base stations is induced by a Dirichlet label partition,
so different "stations" see different class mixes -- the non-IID condition that
breaks distance-based robust aggregation.

This is a STAND-IN for real OpenRAN Gym / ns-O-RAN exports. The data interface
(ClientDataset) is what the rest of the pipeline depends on, so swapping in real
traffic only requires producing ClientDataset objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .partition import dirichlet_label_partition


@dataclass
class ClientDataset:
    """Per-client (base-station) supervised dataset."""

    client_id: int
    x: np.ndarray  # [n, d]
    y: np.ndarray  # [n]

    def __len__(self) -> int:
        return len(self.y)


def _class_centers(n_classes: int, n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Well-separated class centers so the task is learnable but non-trivial."""
    return rng.normal(0.0, 3.0, size=(n_classes, n_features))


def generate_federated_dataset(
    n_clients: int,
    n_features: int,
    n_classes: int,
    samples_per_client: int,
    dirichlet_alpha: float,
    seed: int = 42,
):
    """Generate a federated synthetic dataset.

    Returns:
        clients: list of ClientDataset
        test:    (x_test, y_test) global held-out set
        root:    (x_root, y_root) small clean reference set (used by FLTrust)
    """
    rng = np.random.default_rng(seed)
    centers = _class_centers(n_classes, n_features, rng)

    total = n_clients * samples_per_client
    # Draw a global pool, then partition non-IID across clients.
    y_pool = rng.integers(0, n_classes, size=total)
    x_pool = centers[y_pool] + rng.normal(0.0, 1.0, size=(total, n_features))

    parts = dirichlet_label_partition(y_pool, n_clients, dirichlet_alpha, rng)
    clients: List[ClientDataset] = []
    for cid, idx in enumerate(parts):
        if len(idx) == 0:
            # guarantee every client has at least one sample
            idx = rng.integers(0, total, size=8)
        clients.append(ClientDataset(client_id=cid, x=x_pool[idx], y=y_pool[idx]))

    # Independent global test set.
    n_test = max(500, total // 10)
    y_test = rng.integers(0, n_classes, size=n_test)
    x_test = centers[y_test] + rng.normal(0.0, 1.0, size=(n_test, n_features))

    # Small clean root dataset for FLTrust (server-held).
    n_root = 100
    y_root = rng.integers(0, n_classes, size=n_root)
    x_root = centers[y_root] + rng.normal(0.0, 1.0, size=(n_root, n_features))

    return clients, (x_test, y_test), (x_root, y_root)

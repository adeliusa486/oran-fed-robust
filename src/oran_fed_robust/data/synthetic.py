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


def _class_centers(
    n_classes: int, n_features: int, rng: np.random.Generator, center_scale: float
) -> np.ndarray:
    """Class centers. ``center_scale`` controls class separability: smaller
    values overlap the Gaussian blobs, making the task non-trivial so that
    aggregation rules can actually be distinguished."""
    return rng.normal(0.0, center_scale, size=(n_classes, n_features))


def generate_federated_dataset(
    n_clients: int,
    n_features: int,
    n_classes: int,
    samples_per_client: int,
    dirichlet_alpha: float,
    seed: int = 42,
    center_scale: float = 1.2,
    noise_scale: float = 1.0,
    label_noise: float = 0.05,
    covariate_shift: float = 0.6,
):
    """Generate a federated synthetic dataset.

    The task is deliberately made *non-trivial* and *heterogeneous* so it is a
    meaningful controlled benchmark for robust aggregation:

    * ``center_scale`` / ``noise_scale`` set class overlap (learnable, not 100%).
    * ``label_noise`` flips a fraction of labels to a random class.
    * ``covariate_shift`` gives each base station its own feature-space offset,
      so honest client updates *genuinely diverge* -- the condition under which
      distance-based defenses misclassify honest heterogeneity as attack.

    Returns:
        clients: list of ClientDataset
        test:    (x_test, y_test) global held-out set
        root:    (x_root, y_root) small clean reference set (used by FLTrust)
    """
    rng = np.random.default_rng(seed)
    centers = _class_centers(n_classes, n_features, rng, center_scale)

    def _draw(y: np.ndarray, shift: np.ndarray) -> np.ndarray:
        x = centers[y] + shift + rng.normal(0.0, noise_scale, size=(len(y), n_features))
        return x

    def _corrupt(y: np.ndarray) -> np.ndarray:
        if label_noise <= 0:
            return y
        mask = rng.random(len(y)) < label_noise
        y = y.copy()
        y[mask] = rng.integers(0, n_classes, size=int(mask.sum()))
        return y

    total = n_clients * samples_per_client
    y_pool = rng.integers(0, n_classes, size=total)
    parts = dirichlet_label_partition(y_pool, n_clients, dirichlet_alpha, rng)

    # Per-station covariate shift => honest updates legitimately diverge.
    shifts = rng.normal(0.0, covariate_shift, size=(n_clients, n_features))

    clients: List[ClientDataset] = []
    for cid, idx in enumerate(parts):
        if len(idx) == 0:
            idx = rng.integers(0, total, size=8)
        y_i = _corrupt(y_pool[idx])
        x_i = _draw(y_i, shifts[cid])
        clients.append(ClientDataset(client_id=cid, x=x_i, y=y_i))

    # Independent global test set (no covariate shift, no label noise: clean eval).
    n_test = max(500, total // 10)
    y_test = rng.integers(0, n_classes, size=n_test)
    x_test = _draw(y_test, np.zeros(n_features))

    # Small clean root dataset for FLTrust (server-held).
    n_root = 100
    y_root = rng.integers(0, n_classes, size=n_root)
    x_root = _draw(y_root, np.zeros(n_features))

    return clients, (x_test, y_test), (x_root, y_root)

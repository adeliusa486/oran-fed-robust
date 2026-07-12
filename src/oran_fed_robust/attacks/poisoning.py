"""Model-poisoning attacks.

Two families are supported:
  * data-level: ``label_flip`` corrupts local supervision before training.
  * update-level: ``sign_flip`` and ``fabricated`` perturb the submitted update.

The federated loop routes malicious clients through the appropriate path based
on :func:`is_data_level_attack`.
"""
from __future__ import annotations

import numpy as np


def is_data_level_attack(name: str) -> bool:
    return name == "label_flip"


def poison_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Cyclic label flip: y -> (y + 1) mod C."""
    return (y + 1) % n_classes


def apply_update_attack(
    update: np.ndarray,
    name: str,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Transform an honest update vector into a malicious one (update-level)."""
    if name == "sign_flip":
        return -scale * update
    if name == "fabricated":
        # craft a large, plausible-looking but adversarial direction
        noise = rng.normal(0.0, 1.0, size=update.shape)
        norm = np.linalg.norm(update) + 1e-12
        return scale * norm * noise / (np.linalg.norm(noise) + 1e-12)
    if name in ("none", "label_flip"):
        return update
    raise ValueError(f"Unknown update attack: {name}")

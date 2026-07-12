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


def is_batch_attack(name: str) -> bool:
    """Collusion-aware attacks that are crafted jointly from the round's benign
    updates (white-box, coordinated across the malicious clients)."""
    return name in ("alie", "ipm")


def craft_batch_attack(
    name: str,
    benign_updates: np.ndarray,
    n_malicious: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the single malicious update submitted by every colluding attacker.

    * ``alie`` -- A Little Is Enough (Baruch et al., NeurIPS 2019): shift the
      coordinate-wise mean of benign updates by ``z`` standard deviations, staying
      within natural inter-client variance so that median / trimmed-mean cannot
      filter it. ``scale`` sets ``z``.
    * ``ipm`` -- Inner Product Manipulation (Xie et al., UAI 2020): submit a
      negatively scaled mean of benign updates so the aggregate has negative inner
      product with the true gradient, while keeping magnitude modest. ``scale``
      sets ``epsilon``.
    """
    mu = benign_updates.mean(axis=0)
    if name == "alie":
        sigma = benign_updates.std(axis=0)
        z = scale if scale > 0 else 1.5
        return mu - z * sigma
    if name == "ipm":
        eps = scale if scale > 0 else 0.5
        return -eps * mu
    raise ValueError(f"Unknown batch attack: {name}")


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
    if name == "adaptive":
        # Stealthy intermittent adversary: on an *active* round, flip direction
        # with only a MODERATE magnitude (norm-matched to the honest update) so
        # the perturbation stays within single-round detection tolerance. The
        # attacker is active only intermittently (see FederatedTrainer), so no
        # single round is clearly anomalous -- but the same clients offend
        # repeatedly, which a history-aware defense can accumulate.
        norm = np.linalg.norm(update) + 1e-12
        return -1.5 * update  # direction-reversed, ~1.5x norm (survives clipping partly)
    if name in ("none", "label_flip"):
        return update
    raise ValueError(f"Unknown update attack: {name}")

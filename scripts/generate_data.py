"""Generate and cache a synthetic federated dataset to disk (npz)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from oran_fed_robust.config import load_config
from oran_fed_robust.data import generate_federated_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/processed/dataset.npz")
    args = ap.parse_args()

    cfg = load_config(args.config)
    clients, (x_test, y_test), (x_root, y_root) = generate_federated_dataset(
        n_clients=cfg.data.n_clients,
        n_features=cfg.data.n_features,
        n_classes=cfg.data.n_classes,
        samples_per_client=cfg.data.samples_per_client,
        dirichlet_alpha=cfg.data.dirichlet_alpha,
        seed=cfg.data.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"x_test": x_test, "y_test": y_test, "x_root": x_root, "y_root": y_root}
    for c in clients:
        payload[f"client_{c.client_id}_x"] = c.x
        payload[f"client_{c.client_id}_y"] = c.y
    np.savez_compressed(out, **payload)
    print(f"saved {len(clients)} clients to {out}")


if __name__ == "__main__":
    main()

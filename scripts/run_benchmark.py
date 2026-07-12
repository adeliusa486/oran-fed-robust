"""Run the full attack x defense benchmark sweep and write results.

Usage:
    python scripts/run_benchmark.py --config configs/default.yaml
    python scripts/run_benchmark.py --quick   # tiny sweep for smoke testing
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from oran_fed_robust.aggregation import build_aggregator
from oran_fed_robust.config import list_aggregators, load_config
from oran_fed_robust.data import generate_federated_dataset
from oran_fed_robust.logging_utils import get_logger
from oran_fed_robust.training import FederatedTrainer

log = get_logger()


def run_single(cfg, agg_name: str, attack_name: str) -> dict:
    clients, test_set, root_set = generate_federated_dataset(
        n_clients=cfg.data.n_clients,
        n_features=cfg.data.n_features,
        n_classes=cfg.data.n_classes,
        samples_per_client=cfg.data.samples_per_client,
        dirichlet_alpha=cfg.data.dirichlet_alpha,
        seed=cfg.data.seed,
    )
    aggregator = build_aggregator(agg_name, trim_ratio=cfg.agg.trim_ratio, beta=cfg.agg.beta)
    comp = 0.0 if attack_name == "none" else cfg.attack.compromise_fraction
    trainer = FederatedTrainer(
        clients, test_set, root_set, aggregator,
        n_features=cfg.data.n_features, n_classes=cfg.data.n_classes,
        attack_name=attack_name, compromise_fraction=comp, attack_scale=cfg.attack.scale,
        participants_per_round=cfg.train.participants_per_round,
        local_epochs=cfg.train.local_epochs, lr=cfg.train.lr,
        batch_size=cfg.train.batch_size, seed=cfg.train.seed,
    )
    history = trainer.fit(cfg.train.rounds)
    final = history[-1]
    return {
        "aggregator": agg_name,
        "attack": attack_name,
        "alpha": cfg.data.dirichlet_alpha,
        "compromise": comp,
        "accuracy": round(final.accuracy, 4),
        "macro_f1": round(final.macro_f1, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny sweep for smoke tests")
    ap.add_argument("--attacks", nargs="*", default=["none", "sign_flip", "fabricated"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg.data.n_clients = 12
        cfg.data.samples_per_client = 120
        cfg.train.rounds = 8
        cfg.train.participants_per_round = 6

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for attack in args.attacks:
        for agg in list_aggregators():
            log.info("running attack=%s aggregator=%s", attack, agg)
            rows.append(run_single(cfg, agg, attack))

    csv_path = out_dir / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log.info("wrote %d results to %s", len(rows), csv_path)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

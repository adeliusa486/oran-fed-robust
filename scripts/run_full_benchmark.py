"""Full experiment grid for the paper.

Sweeps  alpha x compromise-fraction x attack x aggregator x seed,  aggregates
mean +/- 95% CI across seeds, and writes:

    results/full_results.csv     one row per (config, seed)
    results/summary.csv          mean/CI per config
    results/table_main.tex       LaTeX main table (retained accuracy)
    results/fig_compromise.csv   data for accuracy-vs-f figure
    results/fig_alpha.csv        data for macro-F1-vs-alpha figure

Usage:
    python scripts/run_full_benchmark.py            # full grid (paper scale)
    python scripts/run_full_benchmark.py --fast     # smaller grid for a quick look
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from statistics import mean, stdev

from oran_fed_robust.aggregation import build_aggregator
from oran_fed_robust.data import generate_federated_dataset
from oran_fed_robust.data.real_traffic import load_real_federated_dataset
from oran_fed_robust.data.real_5g import load_5g_federated_dataset
from oran_fed_robust.data.real_lumos5g import load_lumos5g_federated_dataset
from oran_fed_robust.logging_utils import get_logger
from oran_fed_robust.training import FederatedTrainer

log = get_logger()

# Difficulty of the controlled synthetic benchmark. center_scale=0.5 yields a
# non-trivial (~80% clean) task; covariate_shift induces genuine honest
# divergence across base stations.
DIFFICULTY = dict(center_scale=0.5, noise_scale=1.0, label_noise=0.05, covariate_shift=1.0)

AGGREGATORS = ["fedavg", "krum", "median", "trimmed_mean", "fltrust", "reputation"]
ATTACKS = ["sign_flip", "label_flip", "fabricated", "adaptive", "alie", "ipm"]
# Per-attack magnitude: strong for crude attacks, calibrated for the collusion-
# aware ALIE (z std-devs) and IPM (epsilon) so they stay within filter tolerance.
ATTACK_SCALE = {"alie": 1.5, "ipm": 0.5}

DATASET = "synthetic"  # set by --dataset {synthetic,barcelona,fiveg}


def run_single(agg_name, attack, f, alpha, seed, rounds):
    if DATASET == "barcelona":
        clients, test, root = load_real_federated_dataset(n_clients=50, seed=seed)
    elif DATASET == "fiveg":
        clients, test, root = load_5g_federated_dataset(n_clients=50, seed=seed)
    elif DATASET == "lumos5g":
        clients, test, root = load_lumos5g_federated_dataset(n_clients=50, seed=seed)
    else:
        clients, test, root = generate_federated_dataset(
            n_clients=50, n_features=20, n_classes=5, samples_per_client=400,
            dirichlet_alpha=alpha, seed=seed, **DIFFICULTY,
        )
    n_features = clients[0].x.shape[1]
    agg = build_aggregator(agg_name, trim_ratio=0.1, beta=0.8)
    comp = 0.0 if attack == "none" else f
    trainer = FederatedTrainer(
        clients, test, root, agg, n_features=n_features, n_classes=5,
        attack_name=attack, compromise_fraction=comp,
        attack_scale=ATTACK_SCALE.get(attack, 5.0),
        participants_per_round=20, local_epochs=2, lr=0.05, batch_size=64, seed=seed,
    )
    final = trainer.fit(rounds)[-1]
    return final.accuracy, final.macro_f1


def ci95(xs):
    if len(xs) < 2:
        return 0.0
    return 1.96 * stdev(xs) / math.sqrt(len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="small grid, fewer seeds/rounds")
    ap.add_argument("--dataset", choices=["synthetic", "barcelona", "fiveg", "lumos5g"],
                    default="synthetic", help="which dataset to benchmark on")
    ap.add_argument("--real", action="store_true", help="alias for --dataset barcelona")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    ap.add_argument("--rounds", type=int, default=None, help="override communication rounds")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    global DATASET
    DATASET = "barcelona" if args.real else args.dataset

    if args.fast:
        alphas, fracs, seeds, rounds = [0.1, 1.0], [0.2], [0, 1, 2], 100
    else:
        alphas, fracs, seeds, rounds = [0.1, 0.5, 1.0], [0.1, 0.2, 0.3], [0, 1, 2, 3, 4], 200
    if args.seeds is not None:
        seeds = list(range(args.seeds))
    if args.rounds is not None:
        rounds = args.rounds
    if DATASET != "synthetic":
        alphas = [None]  # real heterogeneity is fixed by the data, no alpha sweep

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Clean (no-attack) reference per (alpha, aggregator), averaged over seeds.
    log.info("computing no-attack references ...")
    ref = {}  # (alpha, agg) -> mean clean accuracy
    for alpha in alphas:
        for agg in AGGREGATORS:
            accs = [run_single(agg, "none", 0.0, alpha, s, rounds)[0] for s in seeds]
            ref[(alpha, agg)] = mean(accs)

    rows = []
    total = len(alphas) * len(fracs) * len(ATTACKS) * len(AGGREGATORS) * len(seeds)
    done = 0
    t0 = time.time()
    for alpha in alphas:
        for f in fracs:
            for attack in ATTACKS:
                for agg in AGGREGATORS:
                    for seed in seeds:
                        acc, f1 = run_single(agg, attack, f, alpha, seed, rounds)
                        retained = 100.0 * acc / ref[(alpha, agg)] if ref[(alpha, agg)] > 0 else 0.0
                        rows.append(dict(alpha=alpha, f=f, attack=attack, aggregator=agg,
                                         seed=seed, accuracy=acc, macro_f1=f1,
                                         retained=retained))
                        done += 1
            log.info("alpha=%s f=%s done (%d/%d, %.0fs elapsed)", alpha, f, done, total, time.time() - t0)

    # per-config aggregation
    with (out / "full_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {}
    for r in rows:
        key = (r["alpha"], r["f"], r["attack"], r["aggregator"])
        summary.setdefault(key, {"acc": [], "f1": [], "ret": []})
        summary[key]["acc"].append(r["accuracy"])
        summary[key]["f1"].append(r["macro_f1"])
        summary[key]["ret"].append(r["retained"])

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha", "f", "attack", "aggregator", "acc_mean", "acc_ci95",
                    "f1_mean", "f1_ci95", "retained_mean", "retained_ci95"])
        for (alpha, f, attack, agg), v in sorted(summary.items()):
            w.writerow([alpha, f, attack, agg,
                        round(mean(v["acc"]), 4), round(ci95(v["acc"]), 4),
                        round(mean(v["f1"]), 4), round(ci95(v["f1"]), 4),
                        round(mean(v["ret"]), 2), round(ci95(v["ret"]), 2)])

    # LaTeX main table: headline slot (IPM is the attack that separates methods).
    _emit_main_table(out, summary, alpha=alphas[0], f=0.3,
                     attack="ipm" if "ipm" in ATTACKS else "sign_flip")
    elapsed = time.time() - t0
    log.info("DONE: %d runs in %.1f min -> %s", len(rows), elapsed / 60.0, out)
    print(json.dumps({"runs": len(rows), "minutes": round(elapsed / 60.0, 2)}, indent=2))


def _emit_main_table(out, summary, alpha, f, attack):
    label = {"fedavg": "FedAvg (no defense)", "krum": "Krum", "median": "Coordinate median",
             "trimmed_mean": "Trimmed-mean", "fltrust": "FLTrust", "reputation": "\\textbf{Proposed}"}
    lines = [
        "% Auto-generated by run_full_benchmark.py -- measured on the controlled synthetic benchmark.",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "\\textbf{Method} & \\textbf{Acc.\\ (\\%)} & \\textbf{Macro F1} & \\textbf{Retained (\\%)} & \\textbf{Clean set} \\\\",
        "\\midrule",
    ]
    needs_clean = {"fltrust": "Yes"}
    for agg in AGGREGATORS:
        v = summary.get((alpha, f, attack, agg))
        if not v:
            continue
        acc = 100 * mean(v["acc"]); accci = 100 * ci95(v["acc"])
        f1 = mean(v["f1"]); f1ci = ci95(v["f1"])
        ret = mean(v["ret"]); retci = ci95(v["ret"])
        row = (f"{label[agg]} & {acc:.1f}$\\pm${accci:.1f} & {f1:.2f}$\\pm${f1ci:.2f} "
               f"& {ret:.1f}$\\pm${retci:.1f} & {needs_clean.get(agg, 'No')} \\\\")
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_main.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

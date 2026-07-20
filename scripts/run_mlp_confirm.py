"""Non-convex robustness confirmation on real Barcelona data.

Re-runs the full methods x attacks grid with a two-layer ReLU MLP (mlp.py)
in place of the convex softmax model, to test whether the IPM failure of the
distance/order-statistic rules survives a non-convex control model.

Writes results_mlp/{full_results.csv, summary.csv} in the same schema as the
main benchmark. Every value is measured over multiple seeds.
"""
import sys, csv, math, time
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oran_fed_robust.models import MLPClassifier
from oran_fed_robust.aggregation import build_aggregator
from oran_fed_robust.data.real_traffic import load_real_federated_dataset
from oran_fed_robust.training import FederatedTrainer

AGGS = ["fedavg", "krum", "median", "trimmed_mean", "fltrust", "reputation", "dm_trimmed_mean"]
ATTACKS = ["sign_flip", "label_flip", "fabricated", "adaptive", "alie", "ipm", "none"]
ATTACK_SCALE = {"alie": 1.5, "ipm": 0.5}
SEEDS = [0, 1, 2, 3, 4]
ROUNDS = 150
FRACTIONS = {"ipm": [0.1, 0.2, 0.3]}   # sweep IPM; other attacks at 0.3 only

def ci95(xs):
    return 1.96 * stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0

def run(agg_name, attack, f, seed):
    clients, test, root = load_real_federated_dataset(n_clients=50, seed=seed)
    nf = clients[0].x.shape[1]
    agg = build_aggregator(agg_name, trim_ratio=0.1, beta=0.8, direction_trim_ratio=0.2)
    comp = 0.0 if attack == "none" else f
    tr = FederatedTrainer(clients, test, root, agg, n_features=nf, n_classes=5,
        attack_name=attack, compromise_fraction=comp,
        attack_scale=ATTACK_SCALE.get(attack, 5.0),
        participants_per_round=20, local_epochs=2, lr=0.05, batch_size=64,
        seed=seed, model_cls=MLPClassifier)
    r = tr.fit(ROUNDS)[-1]
    return r.accuracy, r.macro_f1

def main():
    out = Path(__file__).resolve().parents[1] / "results_mlp"
    out.mkdir(exist_ok=True)
    rows = []
    clean = {}  # (agg,seed) -> clean acc
    t0 = time.time()
    # clean baselines first
    for agg in AGGS:
        for s in SEEDS:
            acc, f1 = run(agg, "none", 0.0, s)
            clean[(agg, s)] = acc
    for attack in ATTACKS:
        if attack == "none":
            continue
        for f in FRACTIONS.get(attack, [0.3]):
            for agg in AGGS:
                for s in SEEDS:
                    acc, f1 = run(agg, attack, f, s)
                    ret = 100 * acc / clean[(agg, s)] if clean[(agg, s)] > 0 else 0.0
                    rows.append(dict(f=f, attack=attack, aggregator=agg, seed=s,
                                     accuracy=round(acc, 4), macro_f1=round(f1, 4),
                                     retained=round(ret, 2)))
            print(f"  done {attack} f={f}  ({time.time()-t0:.0f}s)")
    with (out / "full_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["f", "attack", "aggregator", "seed",
                                           "accuracy", "macro_f1", "retained"])
        w.writeheader(); w.writerows(rows)
    # summary
    agg_map = {}
    for r in rows:
        agg_map.setdefault((r["f"], r["attack"], r["aggregator"]), []).append(r)
    cleanm = {agg: 100 * mean([clean[(agg, s)] for s in SEEDS]) for agg in AGGS}
    with (out / "summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f", "attack", "aggregator", "acc_mean", "acc_ci95",
                    "f1_mean", "f1_ci95", "retained_mean", "retained_ci95"])
        for (f, at, ag), rs in sorted(agg_map.items()):
            accs = [x["accuracy"] for x in rs]; f1s = [x["macro_f1"] for x in rs]
            rets = [x["retained"] for x in rs]
            w.writerow([f, at, ag, round(100*mean(accs), 2), round(100*ci95(accs), 2),
                        round(100*mean(f1s), 2), round(100*ci95(f1s), 2),
                        round(mean(rets), 2), round(ci95(rets), 2)])
    print("clean:", {k: round(v, 1) for k, v in cleanm.items()})
    print("wrote", out, f"({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()

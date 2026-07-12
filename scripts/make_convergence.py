"""Training-convergence figures: accuracy vs communication round (like the
per-round accuracy curves in FL papers). Records the full per-round accuracy
history for each aggregation rule under a fixed attack, on each real dataset.

Usage: python scripts/make_convergence.py
Writes fig_convergence.pdf (3 panels, one per dataset, IPM attack).
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oran_fed_robust.aggregation import build_aggregator
from oran_fed_robust.data.real_traffic import load_real_federated_dataset
from oran_fed_robust.data.real_5g import load_5g_federated_dataset
from oran_fed_robust.data.real_lumos5g import load_lumos5g_federated_dataset
from oran_fed_robust.training import FederatedTrainer

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
    "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 200,
})

ROOT = Path(__file__).resolve().parents[1]
AGGS = ["fedavg", "krum", "median", "trimmed_mean", "fltrust", "reputation"]
LABEL = {"fedavg": "FedAvg", "krum": "Krum", "median": "Median",
         "trimmed_mean": "Trimmed-mean", "fltrust": "FLTrust", "reputation": "Reputation"}
COLOR = {"fedavg": "#111111", "krum": "#D55E00", "median": "#0072B2",
         "trimmed_mean": "#117733", "fltrust": "#8C8C8C", "reputation": "#AA4499"}
DATASETS = [
    ("Barcelona LTE", lambda s: load_real_federated_dataset(n_clients=50, seed=s), 8, 200),
    ("Raca 5G", lambda s: load_5g_federated_dataset(n_clients=50, seed=s), 7, 150),
    ("Lumos5G mmWave", lambda s: load_lumos5g_federated_dataset(n_clients=50, seed=s), 8, 150),
]
ATTACK, F, SCALE = "ipm", 0.3, 0.5


def history(loader, nfeat, rounds, agg_name):
    clients, test, root = loader(0)
    agg = build_aggregator(agg_name, trim_ratio=0.1, beta=0.8)
    tr = FederatedTrainer(clients, test, root, agg, n_features=nfeat, n_classes=5,
                          attack_name=ATTACK, compromise_fraction=F, attack_scale=SCALE,
                          participants_per_round=20, local_epochs=2, lr=0.05,
                          batch_size=64, seed=0)
    return [r.accuracy * 100 for r in tr.fit(rounds)]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, (title, loader, nfeat, rounds) in zip(axes, DATASETS):
        for agg in AGGS:
            ys = history(loader, nfeat, rounds, agg)
            ax.plot(np.arange(1, len(ys) + 1), ys, lw=1.4, color=COLOR[agg], label=LABEL[agg])
        ax.set_title(title, pad=6)
        ax.set_xlabel("Communication round")
        ax.grid(True, ls="-", lw=0.4, alpha=0.25)
        print("done", title)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               columnspacing=1.1, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ROOT.parent / "fig_convergence.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "assets" / "fig_convergence.png", bbox_inches="tight", dpi=200)
    print("wrote fig_convergence.pdf")


if __name__ == "__main__":
    main()

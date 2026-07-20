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
AGGS = ["fedavg", "krum", "median", "trimmed_mean", "fltrust", "reputation", "dm_trimmed_mean"]
LABEL = {"fedavg": "FedAvg", "krum": "Krum", "median": "Median",
         "trimmed_mean": "Trimmed-mean", "fltrust": "FLTrust", "reputation": "Reputation", "dm_trimmed_mean": "DM-TM"}
COLOR = {"fedavg": "#000000", "krum": "#E69F00", "median": "#0072B2",
         "trimmed_mean": "#56B4E9", "fltrust": "#7F7F7F", "reputation": "#9467BD", "dm_trimmed_mean": "#D55E00"}
DATASETS = [
    ("Barcelona LTE", lambda s: load_real_federated_dataset(n_clients=50, seed=s), 8, 150),
    ("Raca 5G", lambda s: load_5g_federated_dataset(n_clients=50, seed=s), 7, 150),
    ("Lumos5G mmWave", lambda s: load_lumos5g_federated_dataset(n_clients=50, seed=s), 8, 150),
]
ATTACK, F, SCALE = "ipm", 0.3, 0.5


MARK = {"fedavg": "o", "krum": "^", "median": "s", "trimmed_mean": "D",
        "fltrust": "v", "reputation": "P", "dm_trimmed_mean": "*"}


def history(loader, nfeat, rounds, agg_name, attack):
    clients, test, root = loader(0)
    agg = build_aggregator(agg_name, trim_ratio=0.1, beta=0.8)
    comp = 0.0 if attack == "none" else F
    tr = FederatedTrainer(clients, test, root, agg, n_features=nfeat, n_classes=5,
                          attack_name=attack, compromise_fraction=comp, attack_scale=SCALE,
                          participants_per_round=20, local_epochs=2, lr=0.05,
                          batch_size=64, seed=0)
    return [r.accuracy * 100 for r in tr.fit(rounds)]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
    for ax, (title, loader, nfeat, rounds) in zip(axes, DATASETS):
        me = max(6, rounds // 12)                      # marker spacing
        ref = history(loader, nfeat, rounds, "fedavg", "none")[-1]  # no-attack ceiling
        for agg in AGGS:
            ys = history(loader, nfeat, rounds, agg, ATTACK)
            ax.plot(np.arange(1, len(ys) + 1), ys, lw=1.3, color=COLOR[agg],
                    marker=MARK[agg], markevery=me, ms=4.5, mfc=COLOR[agg],
                    mec="white", mew=0.5, label=LABEL[agg])
        ax.axhline(ref, ls="--", lw=1.1, color="black", label="No-attack ceiling")
        ax.set_title(title, pad=6)
        ax.set_xlabel("Communication round")
        ax.grid(True, ls="-", lw=0.4, alpha=0.22)
        print("done", title)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.13),
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout()
    fig.savefig(ROOT.parent / "fig_convergence.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "assets" / "fig_convergence.png", bbox_inches="tight", dpi=200)
    print("wrote fig_convergence.pdf")


ATTACKS = ["sign_flip", "label_flip", "fabricated", "adaptive", "alie", "ipm"]
ATT_LABEL = {"sign_flip": "Sign-flip", "label_flip": "Label-flip", "fabricated": "Fabricated",
             "adaptive": "Adaptive", "alie": "ALIE", "ipm": "IPM"}
ATT_SCALE = {"alie": 1.5, "ipm": 0.5}


def per_attack():
    """6-panel per-attack convergence on Barcelona (accuracy vs round)."""
    global ATTACK, SCALE
    title, loader, nfeat, rounds = DATASETS[0]
    me = max(6, rounds // 12)
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6), sharex=True)
    ref = history(loader, nfeat, rounds, "fedavg", "none")[-1]
    for ax, attack in zip(axes.ravel(), ATTACKS):
        ATTACK = attack
        SCALE = ATT_SCALE.get(attack, 5.0)
        for agg in AGGS:
            ys = history(loader, nfeat, rounds, agg, attack)
            ax.plot(np.arange(1, len(ys) + 1), ys, lw=1.2, color=COLOR[agg],
                    marker=MARK[agg], markevery=me, ms=4, mfc=COLOR[agg],
                    mec="white", mew=0.5, label=LABEL[agg])
        ax.axhline(ref, ls="--", lw=1.0, color="black", label="No-attack ceiling")
        ax.set_title(ATT_LABEL[attack], pad=5)
        ax.grid(True, ls="-", lw=0.4, alpha=0.22)
        print("done attack", attack)
    for ax in axes[-1, :]:
        ax.set_xlabel("Communication round")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.07),
               columnspacing=1.4, handletextpad=0.5)
    fig.suptitle("Barcelona LTE: per-round accuracy under each attack ($f=0.3$)", y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(ROOT.parent / "fig_attack_convergence.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "assets" / "fig_attack_convergence.png", bbox_inches="tight", dpi=200)
    print("wrote fig_attack_convergence.pdf")


if __name__ == "__main__":
    main()
    per_attack()

"""Publication-quality figures (matplotlib) from the measured result grids.

Produces vector PDFs used directly in the manuscript:
    fig_ipm_panels.pdf  -- accuracy vs compromise fraction under IPM, one panel
                           per real dataset, with 95% CI bands (the key finding).
    fig_heatmap.pdf     -- Barcelona methods x attacks accuracy heatmap.
    fig_heatmap.png     -- PNG copy for the repository README.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.dpi": 200,
})

ROOT = Path(__file__).resolve().parents[1]

# Optional live no-attack ceiling: the same measured quantity the convergence
# figures draw (final-round FedAvg accuracy with no attack). Imported lazily so
# the figure script still runs when the datasets/training stack are unavailable.
try:
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from oran_fed_robust.aggregation import build_aggregator
    from oran_fed_robust.data.real_traffic import load_real_federated_dataset
    from oran_fed_robust.data.real_5g import load_5g_federated_dataset
    from oran_fed_robust.data.real_lumos5g import load_lumos5g_federated_dataset
    from oran_fed_robust.training import FederatedTrainer
    _CEIL_OK = True
except Exception:  # pragma: no cover - datasets not present
    _CEIL_OK = False

_CEIL_CFG = {
    "Barcelona LTE": (lambda s: load_real_federated_dataset(n_clients=50, seed=s), 8, 200),
    "Raca 5G": (lambda s: load_5g_federated_dataset(n_clients=50, seed=s), 7, 150),
    "Lumos5G mmWave": (lambda s: load_lumos5g_federated_dataset(n_clients=50, seed=s), 8, 150),
}


def _no_attack_ceiling(title):
    """Measured no-attack FedAvg final accuracy for a dataset (identical to the
    ceiling drawn in the convergence figures); None if it cannot be measured."""
    if not _CEIL_OK or title not in _CEIL_CFG:
        return None
    try:
        loader, nfeat, rounds = _CEIL_CFG[title]
        clients, test, root = loader(0)
        agg = build_aggregator("fedavg", trim_ratio=0.1, beta=0.8)
        tr = FederatedTrainer(clients, test, root, agg, n_features=nfeat, n_classes=5,
                              attack_name="none", compromise_fraction=0.0, attack_scale=0.5,
                              participants_per_round=20, local_epochs=2, lr=0.05,
                              batch_size=64, seed=0)
        return [r.accuracy * 100 for r in tr.fit(rounds)][-1]
    except Exception:  # pragma: no cover
        return None


AGGS = ["fedavg", "krum", "median", "trimmed_mean", "fltrust", "reputation"]
LABEL = {"fedavg": "FedAvg", "krum": "Krum", "median": "Median",
         "trimmed_mean": "Trimmed-mean", "fltrust": "FLTrust", "reputation": "Reputation"}
ATTACKS = ["sign_flip", "label_flip", "fabricated", "adaptive", "alie", "ipm"]
ATT_LABEL = {"sign_flip": "Sign-flip", "label_flip": "Label-flip", "fabricated": "Fabricated",
             "adaptive": "Adaptive", "alie": "ALIE", "ipm": "IPM"}
# Professional colorblind-aware palette (no red, no green).
COLOR = {"fedavg": "#000000", "krum": "#E69F00", "median": "#0072B2",
         "trimmed_mean": "#8C564B", "fltrust": "#7F7F7F", "reputation": "#9467BD"}
MARK = {"fedavg": "o", "krum": "^", "median": "s", "trimmed_mean": "D",
        "fltrust": "v", "reputation": "P"}


def _plot(ax, x, ys, es, agg):
    ax.plot(x, ys, marker=MARK[agg], ms=4.5, lw=1.6, color=COLOR[agg],
            mfc=COLOR[agg], mec="white", mew=0.6, label=LABEL[agg], zorder=3)
    ax.fill_between(x, ys - es, ys + es, color=COLOR[agg], alpha=0.10, lw=0, zorder=1)


def _style(ax):
    ax.grid(True, ls="-", lw=0.4, alpha=0.25, zorder=0)
    ax.tick_params(labelsize=9)


def load_summary(path: Path):
    """Return dict[(f, attack, agg)] = (acc_mean, acc_ci95)."""
    d = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d[(r["f"], r["attack"], r["aggregator"])] = (
                float(r["acc_mean"]) * 100, float(r["acc_ci95"]) * 100)
    return d


def ipm_panels(datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.6, 3.0), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    fracs = ["0.1", "0.2", "0.3"]
    x = [0.1, 0.2, 0.3]
    for ax, (title, d) in zip(axes, datasets):
        for agg in AGGS:
            ys = np.array([d.get((f, "ipm", agg), (np.nan, 0))[0] for f in fracs])
            es = np.array([d.get((f, "ipm", agg), (np.nan, 0))[1] for f in fracs])
            _plot(ax, x, ys, es, agg)
        ref = _no_attack_ceiling(title)
        if ref is not None:
            ax.axhline(ref, ls="--", lw=1.1, color="black", label="No-attack ceiling")
        ax.set_title(title, pad=6)
        ax.set_xlabel("Compromise fraction $f$")
        ax.set_xlim(0.07, 0.33); ax.set_xticks(x)
        _style(ax)
    axes[0].set_ylabel("Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.13), frameon=False, columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout()
    out = ROOT.parent / "fig_ipm_panels.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(ROOT / "assets" / "fig_ipm_panels.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out)


def heatmap(d, title, stem):
    mat = np.array([[d.get(("0.3", a, agg), (np.nan, 0))[0] for a in ATTACKS]
                    for agg in AGGS])
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    im = ax.imshow(mat, cmap="cividis", vmin=0, vmax=90, aspect="auto")
    ax.set_xticks(range(len(ATTACKS)), [ATT_LABEL[a] for a in ATTACKS], rotation=30, ha="right")
    ax.set_yticks(range(len(AGGS)), [LABEL[a] for a in AGGS])
    for i in range(len(AGGS)):
        for j in range(len(ATTACKS)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v < 42 else "black", fontsize=7.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Accuracy (%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(ROOT.parent / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "assets" / f"{stem}.png", bbox_inches="tight", dpi=200)
    print("wrote", ROOT.parent / f"{stem}.pdf")
    plt.close(fig)


def attack_grid(d, title, stem):
    """2x3 grid: accuracy vs compromise fraction, one panel per attack."""
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    fracs = ["0.1", "0.2", "0.3"]
    x = [0.1, 0.2, 0.3]
    for ax, attack in zip(axes.ravel(), ATTACKS):
        for agg in AGGS:
            ys = np.array([d.get((f, attack, agg), (np.nan, 0))[0] for f in fracs])
            es = np.array([d.get((f, attack, agg), (np.nan, 0))[1] for f in fracs])
            _plot(ax, x, ys, es, agg)
        ax.set_title(ATT_LABEL[attack])
        ax.set_xlim(0.07, 0.33); ax.set_xticks(x)
        _style(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel("Compromise fraction $f$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle(title, y=1.10, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT.parent / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def heatmap_row(datasets, stem):
    """1xN heatmaps (methods x attacks) side by side, one per dataset."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.0 * len(datasets), 3.0))
    if len(datasets) == 1:
        axes = [axes]
    im = None
    for ax, (title, d) in zip(axes, datasets):
        mat = np.array([[d.get(("0.3", a, agg), (np.nan, 0))[0] for a in ATTACKS] for agg in AGGS])
        im = ax.imshow(mat, cmap="cividis", vmin=0, vmax=90, aspect="auto")
        ax.set_xticks(range(len(ATTACKS)), [ATT_LABEL[a] for a in ATTACKS], rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(AGGS)), [LABEL[a] for a in AGGS] if ax is axes[0] else [""] * len(AGGS), fontsize=7.5)
        for i in range(len(AGGS)):
            for j in range(len(ATTACKS)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="white" if v < 42 else "black", fontsize=6.5)
        ax.set_title(title, fontsize=9)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Accuracy (%)")
    out = ROOT.parent / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(ROOT / "assets" / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", out)


def main():
    barc = load_summary(ROOT / "results_real" / "summary.csv")
    datasets = [("Barcelona LTE", barc)]
    for name, folder in [("Raca 5G", "results_5g"), ("Lumos5G mmWave", "results_lumos")]:
        p = ROOT / folder / "summary.csv"
        if p.exists():
            datasets.append((name, load_summary(p)))
    ipm_panels(datasets)
    heatmap(barc, "Barcelona LTE: accuracy by rule and attack ($f=0.3$)", "fig_heatmap")
    attack_grid(barc, "Barcelona LTE: accuracy vs. compromise fraction, by attack", "fig_attack_grid")
    heatmap_row(datasets, "fig_heatmap_row")


if __name__ == "__main__":
    main()

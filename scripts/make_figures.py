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
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

ROOT = Path(__file__).resolve().parents[1]
AGGS = ["fedavg", "krum", "median", "trimmed_mean", "fltrust", "reputation"]
LABEL = {"fedavg": "FedAvg", "krum": "Krum", "median": "Median",
         "trimmed_mean": "Trimmed-mean", "fltrust": "FLTrust", "reputation": "Reputation"}
ATTACKS = ["sign_flip", "label_flip", "fabricated", "adaptive", "alie", "ipm"]
ATT_LABEL = {"sign_flip": "Sign-flip", "label_flip": "Label-flip", "fabricated": "Fabricated",
             "adaptive": "Adaptive", "alie": "ALIE", "ipm": "IPM"}
# Colorblind-safe (Okabe-Ito) palette.
COLOR = {"fedavg": "#000000", "krum": "#D55E00", "median": "#0072B2",
         "trimmed_mean": "#009E73", "fltrust": "#999999", "reputation": "#CC79A7"}
MARK = {"fedavg": "o", "krum": "^", "median": "s", "trimmed_mean": "D",
        "fltrust": "v", "reputation": "X"}


def load_summary(path: Path):
    """Return dict[(f, attack, agg)] = (acc_mean, acc_ci95)."""
    d = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d[(r["f"], r["attack"], r["aggregator"])] = (
                float(r["acc_mean"]) * 100, float(r["acc_ci95"]) * 100)
    return d


def ipm_panels(datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2, 2.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    fracs = ["0.1", "0.2", "0.3"]
    x = [0.1, 0.2, 0.3]
    for ax, (title, d) in zip(axes, datasets):
        for agg in AGGS:
            ys = [d.get((f, "ipm", agg), (np.nan, 0))[0] for f in fracs]
            es = [d.get((f, "ipm", agg), (np.nan, 0))[1] for f in fracs]
            ys, es = np.array(ys), np.array(es)
            ax.plot(x, ys, marker=MARK[agg], ms=4, lw=1.4, color=COLOR[agg],
                    label=LABEL[agg])
            ax.fill_between(x, ys - es, ys + es, color=COLOR[agg], alpha=0.12, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Compromise fraction $f$")
        ax.set_xticks(x)
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    axes[0].set_ylabel("Accuracy (\\%)" if plt.rcParams["text.usetex"] else "Accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = ROOT.parent / "fig_ipm_panels.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def heatmap(d, title, stem):
    mat = np.array([[d.get(("0.3", a, agg), (np.nan, 0))[0] for a in ATTACKS]
                    for agg in AGGS])
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=90, aspect="auto")
    ax.set_xticks(range(len(ATTACKS)), [ATT_LABEL[a] for a in ATTACKS], rotation=30, ha="right")
    ax.set_yticks(range(len(AGGS)), [LABEL[a] for a in AGGS])
    for i in range(len(AGGS)):
        for j in range(len(ATTACKS)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="black" if 25 < v < 75 else "white", fontsize=7.5)
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
            ax.plot(x, ys, marker=MARK[agg], ms=3.5, lw=1.3, color=COLOR[agg], label=LABEL[agg])
            ax.fill_between(x, ys - es, ys + es, color=COLOR[agg], alpha=0.10, lw=0)
        ax.set_title(ATT_LABEL[attack], fontsize=9)
        ax.set_xticks(x)
        ax.grid(True, ls=":", lw=0.5, alpha=0.6)
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
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=90, aspect="auto")
        ax.set_xticks(range(len(ATTACKS)), [ATT_LABEL[a] for a in ATTACKS], rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(AGGS)), [LABEL[a] for a in AGGS] if ax is axes[0] else [""] * len(AGGS), fontsize=7.5)
        for i in range(len(AGGS)):
            for j in range(len(ATTACKS)):
                v = mat[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        color="black" if 25 < v < 75 else "white", fontsize=6.5)
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

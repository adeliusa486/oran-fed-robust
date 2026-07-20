"""Generate LaTeX tables/figure-data for the paper from summary.csv.

Usage: python scripts/make_paper_tables.py --dir results_real --f 0.3
Emits into <dir>/:
    table_by_attack.tex   methods x attacks accuracy matrix at compromise f
    fig_compromise.dat     accuracy vs f for each method under the IPM attack
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

AGG_ORDER = ["fedavg", "krum", "median", "trimmed_mean", "fltrust", "reputation", "dm_trimmed_mean"]
AGG_LABEL = {"fedavg": "FedAvg", "krum": "Krum", "median": "Median",
             "trimmed_mean": "Trimmed-mean", "fltrust": "FLTrust",
             "reputation": "Reputation", "dm_trimmed_mean": "DM-TM"}
ATT_ORDER = ["sign_flip", "label_flip", "fabricated", "adaptive", "alie", "ipm"]
ATT_LABEL = {"sign_flip": "Sign-flip", "label_flip": "Label-flip",
             "fabricated": "Fabricated", "adaptive": "Adaptive",
             "alie": "ALIE", "ipm": "IPM"}


def load(path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    d = {}
    for r in rows:
        d[(r["attack"], r["aggregator"], r["f"])] = (
            float(r["acc_mean"]), float(r["acc_ci95"]))
    fset = sorted({r["f"] for r in rows})
    attacks = [a for a in ATT_ORDER if any(r["attack"] == a for r in rows)]
    return d, fset, attacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_real")
    ap.add_argument("--f", default="0.3")
    args = ap.parse_args()
    out = Path(args.dir)
    d, fset, attacks = load(out / "summary.csv")
    f = args.f if args.f in fset else fset[-1]

    # methods x attacks accuracy matrix, bold=best per attack, underline=worst
    # defended rule per attack (FedAvg, i.e. "no defense", is excluded from the
    # worst-marking since it is not a defense to indict).
    best_per_attack, worst_per_attack = {}, {}
    for a in attacks:
        vals = {agg: d[(a, agg, f)][0] for agg in AGG_ORDER if (a, agg, f) in d}
        if vals:
            best_per_attack[a] = max(vals, key=vals.get)
            defended = {k: v for k, v in vals.items() if k != "fedavg"}
            if defended:
                worst_per_attack[a] = min(defended, key=defended.get)

    # diverging blue(good)->orange(bad) cell shading around the mid of the range,
    # matching the color-coded tables common in FL-security papers (e.g. SoK).
    allv = [100 * d[(a, agg, f)][0] for a in attacks for agg in AGG_ORDER if (a, agg, f) in d]
    lo, hi = min(allv), max(allv)
    mid = 0.5 * (lo + hi)

    def shade(v):
        if hi - lo < 1e-6:
            return ""
        span = max(hi - mid, mid - lo)
        inten = min(38, abs(v - mid) / span * 38) if span > 1e-6 else 0
        colour = "MidnightBlue" if v >= mid else "BurntOrange"
        return f"\\cellcolor{{{colour}!{inten:.0f}}}"

    lines = ["% methods x attacks accuracy (%) at f=" + f + ", mean over seeds.",
             "% requires \\usepackage[table,dvipsnames]{xcolor} in the preamble.",
             "\\begin{tabular}{l" + "c" * len(attacks) + "}", "\\toprule",
             "\\textbf{Method} & " + " & ".join(ATT_LABEL[a] for a in attacks) + " \\\\",
             "\\midrule"]
    for agg in AGG_ORDER:
        cells = []
        for a in attacks:
            v = d.get((a, agg, f))
            if not v:
                cells.append("--"); continue
            acc = 100 * v[0]
            s = f"{acc:.1f}"
            if best_per_attack.get(a) == agg:
                s = "\\textbf{" + s + "}"
            elif worst_per_attack.get(a) == agg:
                s = "\\underline{" + s + "}"
            cells.append(shade(acc) + s)
        lines.append(f"{AGG_LABEL[agg]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_by_attack.tex").write_text("\n".join(lines), encoding="utf-8")

    # accuracy vs f under IPM (pgfplots-friendly)
    if any(a == "ipm" for a in attacks):
        dat = ["f " + " ".join(AGG_ORDER)]
        for ff in fset:
            row = [ff] + [f"{100*d[('ipm', agg, ff)][0]:.2f}" if ('ipm', agg, ff) in d else "nan"
                          for agg in AGG_ORDER]
            dat.append(" ".join(row))
        (out / "fig_compromise.dat").write_text("\n".join(dat), encoding="utf-8")
    print("wrote table_by_attack.tex and fig_compromise.dat to", out)


if __name__ == "__main__":
    main()

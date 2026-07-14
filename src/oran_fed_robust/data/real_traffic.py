"""Real Open-RAN-style federated dataset from the Barcelona LTE traces.

Source: the "Federated Traffic Prediction for 5G and Beyond Challenge" dataset
released with Perifanis et al., *Federated Learning for 5G Base Station Traffic
Forecasting* (Computer Networks, 2023). Three real LTE base stations --- ElBorn,
LesCorts, PobleSec --- each with per-2-minute Key Performance Measurements
(downlink/uplink volume, active users, MCS statistics, resource-block usage).

We turn this into a supervised **xApp control task**: classify the cell's
downlink-load regime (five global quintiles of downlink volume) from the
remaining KPMs --- the kind of load/congestion classification a traffic-steering
or slicing xApp performs. Federated clients are formed by splitting each base
station's *time-ordered* trace into contiguous windows, so heterogeneity across
clients is **real** (different base stations and different times of day/week
genuinely differ in load and user mix), not synthetically induced.

The loader returns the same interface as the synthetic generator, so every
aggregator, attack, and the training harness work unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .synthetic import ClientDataset

BASE_STATIONS = ("ElBorn", "LesCorts", "PobleSec")
# KPM features used as inputs; the label is derived from ``down`` (and its direct
# resource correlates rb_down* are dropped to avoid trivial leakage).
FEATURES = ("up", "rnti_count", "mcs_down", "mcs_down_var",
            "mcs_up", "mcs_up_var", "rb_up", "rb_up_var")
TARGET = "down"
N_CLASSES = 5


def _default_data_dir() -> Path:
    # <repo>/data/barcelona
    return Path(__file__).resolve().parents[3] / "data" / "barcelona"


def _read_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features[n,d], target[n], order[n]) parsed from one station CSV."""
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    col = {c: i for i, c in enumerate(header)}
    rows = np.genfromtxt(path, delimiter=",", skip_header=1,
                         usecols=[col[c] for c in FEATURES] + [col[TARGET]],
                         dtype=float)
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    x = rows[:, :len(FEATURES)]
    tgt = rows[:, -1]
    order = np.arange(len(x))  # CSVs are already time-ordered
    return x, tgt, order


def load_real_federated_dataset(
    n_clients: int = 50,
    data_dir: Optional[str] = None,
    test_fraction: float = 0.2,
    root_size: int = 100,
    seed: int = 42,
):
    """Load the Barcelona LTE traces as a federated classification dataset.

    Returns:
        clients: list[ClientDataset]   (real base-station x time-window shards)
        test:    (x_test, y_test)      stratified global held-out set
        root:    (x_root, y_root)      small clean reference set (for FLTrust)
    """
    rng = np.random.default_rng(seed)
    ddir = Path(data_dir) if data_dir else _default_data_dir()
    missing = [b for b in BASE_STATIONS if not (ddir / f"{b}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"Barcelona CSVs not found in {ddir} (missing {missing}). "
            "Run scripts/get_real_data.py first.")

    per_station = {b: _read_csv(ddir / f"{b}.csv") for b in BASE_STATIONS}

    # Disjoint train/test split FIRST, per station, so the global held-out test
    # set shares no sample with any training client. Standardization statistics
    # and the quintile bin edges are fit on the TRAIN pool only (no test leakage).
    station_train = {}
    tr_x_parts, tr_t_parts, te_x_parts, te_t_parts = [], [], [], []
    for b, (x, t, _order) in per_station.items():
        n = len(x)
        perm = rng.permutation(n)
        n_test = int(test_fraction * n)
        te_idx = perm[:n_test]
        tr_idx = np.sort(perm[n_test:])  # keep time order within the train shard
        station_train[b] = (x[tr_idx], t[tr_idx])
        tr_x_parts.append(x[tr_idx]); tr_t_parts.append(t[tr_idx])
        te_x_parts.append(x[te_idx]); te_t_parts.append(t[te_idx])

    train_x = np.vstack(tr_x_parts); train_t = np.concatenate(tr_t_parts)
    test_x = np.vstack(te_x_parts); test_t = np.concatenate(te_t_parts)

    # Quintile bins + standardization fit on TRAIN targets/features only.
    edges = np.quantile(train_t, np.linspace(0, 1, N_CLASSES + 1)[1:-1])
    def to_class(t):
        return np.digitize(t, edges).astype(int)
    mu, sd = train_x.mean(axis=0), train_x.std(axis=0) + 1e-8

    # Allocate clients across stations proportionally to their (train) length;
    # each client is a contiguous real time window of that station's train trace.
    total = sum(len(v[0]) for v in station_train.values())
    clients: List[ClientDataset] = []
    cid = 0
    for b, (x, t) in station_train.items():
        k = max(1, round(n_clients * len(x) / total))
        xs = (x - mu) / sd
        ys = to_class(t)
        bounds = np.linspace(0, len(x), k + 1).astype(int)
        for j in range(k):
            lo, hi = bounds[j], bounds[j + 1]
            if hi - lo < 10:
                continue
            clients.append(ClientDataset(client_id=cid, x=xs[lo:hi], y=ys[lo:hi]))
            cid += 1

    # Disjoint global test set; clean root set drawn from TRAIN only (FLTrust).
    xs_test = (test_x - mu) / sd
    ys_test = to_class(test_t)
    xs_train_all = (train_x - mu) / sd
    ys_train_all = to_class(train_t)
    root_idx = rng.permutation(len(xs_train_all))[:root_size]
    return (clients,
            (xs_test, ys_test),
            (xs_train_all[root_idx], ys_train_all[root_idx]))

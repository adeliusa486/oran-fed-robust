"""Third real dataset: the Lumos5G commercial mmWave 5G measurement dataset.

Source: Narayanan et al., "Lumos5G: Mapping and Predicting Commercial mmWave 5G
Throughput" (ACM IMC 2020), a public dataset of commercial mmWave 5G
measurements collected in Minneapolis, MN, with per-sample radio KPIs (LTE and
5G-NR reference-signal power/quality, SINR, signal strength), mobility, and
achieved downlink throughput.

We build the same style of supervised xApp task as for the other datasets:
classify the downlink-throughput regime (five global quintiles) from the radio
and context measurements a scheduling/link-adaptation xApp would observe.
Federated clients are formed from the real measurement runs (distinct
trajectories, towers, and mobility), so heterogeneity across clients is genuine.

This is the third independent real dataset in the study (after the Barcelona LTE
base-station traces and the Raca et al. 5G production traces); it differs in
country, operator, research group, and radio technology (mmWave), which is what
makes it a strong external-validity check. It returns the same interface as the
other loaders, so every aggregator, attack, and the training harness are unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np

from .synthetic import ClientDataset

FEATURES = ("abstractSignalStr", "movingSpeed", "lte_rssi", "lte_rsrp",
            "lte_rsrq", "nr_ssRsrp", "nr_ssRsrq", "nr_ssSinr")
TARGET = "Throughput"
RUN_COL = "run_num"
N_CLASSES = 5
_INT_SENTINEL = 2147483647.0

_CACHE: dict = {}


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "lumos5g"


def _read(ddir: Path):
    """Parse Lumos5G-v1.0.csv into (features, target, run_id). Cached per dir."""
    key = str(ddir.resolve())
    if key in _CACHE:
        return _CACHE[key]
    path = ddir / "Lumos5G-v1.0.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/get_real_data.py --lumos5g first.")
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    col = {c: i for i, c in enumerate(header)}
    want = [col[c] for c in FEATURES] + [col[TARGET], col[RUN_COL]]
    rows = np.genfromtxt(path, delimiter=",", skip_header=1, usecols=want, dtype=float)
    rows[rows == _INT_SENTINEL] = np.nan
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    x = rows[:, :len(FEATURES)]
    tgt = rows[:, len(FEATURES)]
    run = rows[:, len(FEATURES) + 1].astype(int)
    out = (x, tgt, run)
    _CACHE[key] = out
    return out


def load_lumos5g_federated_dataset(
    n_clients: int = 50,
    data_dir: Optional[str] = None,
    test_fraction: float = 0.2,
    root_size: int = 100,
    seed: int = 42,
    max_per_client: int = 400,
):
    """Load Lumos5G as a federated throughput-classification dataset."""
    rng = np.random.default_rng(seed)
    ddir = Path(data_dir) if data_dir else _default_data_dir()
    x, t, run = _read(ddir)

    # Rank-based global quintiles => globally balanced classes; per-client skew real.
    order = np.argsort(np.argsort(t))
    y = np.floor(order / len(t) * N_CLASSES).astype(int)
    y = np.clip(y, 0, N_CLASSES - 1)

    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
    xs = (x - mu) / sd

    # Clients: keep real run boundaries; order by run and split into contiguous
    # shards so each client is one or a few real measurement runs (genuine non-IID).
    run_order = np.argsort(run, kind="stable")
    xs, y = xs[run_order], y[run_order]
    clients: List[ClientDataset] = []
    bounds = np.linspace(0, len(xs), n_clients + 1).astype(int)
    for cid in range(n_clients):
        lo, hi = bounds[cid], bounds[cid + 1]
        if hi - lo < 10:
            continue
        idx = np.arange(lo, hi)
        if hi - lo > max_per_client:
            idx = rng.choice(idx, size=max_per_client, replace=False)
            idx.sort()
        clients.append(ClientDataset(client_id=cid, x=xs[idx], y=y[idx]))

    perm = rng.permutation(len(xs))
    n_test = min(int(test_fraction * len(xs)), 5000)
    test_idx, rest = perm[:n_test], perm[n_test:]
    root_idx = rest[:root_size]
    return (clients, (xs[test_idx], y[test_idx]), (xs[root_idx], y[root_idx]))

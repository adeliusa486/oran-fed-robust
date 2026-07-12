"""Second real dataset: the Raca et al. 5G production-network KPI traces.

Source: Raca et al., "Beyond throughput, the next generation: a 5G dataset with
channel and context metrics" (ACM MMSys 2020), a public dataset of real 5G
client-side Key Performance Measurements collected on a commercial Irish
operator network (RSRP, RSRQ, SNR, CQI, RSSI, throughput, mobility). Repository:
https://github.com/uccmisl/5Gdataset.

We build a supervised **xApp link-adaptation-style task**: classify the downlink
throughput regime (five global quintiles of DL bitrate) from the radio KPMs a
scheduling/link-adaptation xApp observes. Federated clients are formed from the
real measurement sessions (distinct cells, mobility patterns, and applications),
so heterogeneity across clients is genuine, exactly as for the Barcelona loader.

Returns the same interface as the synthetic and Barcelona loaders, so every
aggregator, attack, and the training harness work unchanged.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional

import numpy as np

from .synthetic import ClientDataset

FEATURES = ("RSRP", "RSRQ", "SNR", "CQI", "RSSI", "Speed", "UL_bitrate")
TARGET = "DL_bitrate"
N_CLASSES = 5


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "fiveg"


_CACHE: dict = {}


def _read_all(ddir: Path):
    """Read every session CSV, returning (features, target, session_id) arrays.

    Parsing all sessions is the expensive step, so results are cached per
    directory: the benchmark calls this hundreds of times across seeds and
    configurations, and re-parsing each time would dominate the runtime."""
    key = str(ddir.resolve())
    if key in _CACHE:
        return _CACHE[key]
    files = sorted(glob.glob(str(ddir / "**" / "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No 5G session CSVs under {ddir}. Run scripts/get_real_data.py --fiveg first.")
    xs, ts, sess = [], [], []
    for sid, f in enumerate(files):
        header = None
        rows = []
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.rstrip("\n").split(",")
                if header is None:
                    header = {c: i for i, c in enumerate(parts)}
                    if TARGET not in header:
                        break
                    continue
                rows.append(parts)
        if not rows or header is None or TARGET not in header:
            continue
        def col(name, p):
            try:
                return float(p[header[name]])
            except (ValueError, IndexError, KeyError):
                return 0.0
        for p in rows:
            xs.append([col(c, p) for c in FEATURES])
            ts.append(col(TARGET, p))
            sess.append(sid)
    x = np.asarray(xs, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = (x, np.asarray(ts, dtype=float), np.asarray(sess, dtype=int))
    _CACHE[key] = out
    return out


def load_5g_federated_dataset(
    n_clients: int = 50,
    data_dir: Optional[str] = None,
    test_fraction: float = 0.2,
    root_size: int = 100,
    seed: int = 42,
    max_per_client: int = 400,
):
    """Load the 5G traces as a federated throughput-classification dataset.

    ``max_per_client`` caps each client's sample count (the raw sessions are
    large) so per-round training cost is comparable across datasets; the cap is
    applied by uniform subsampling within each client's real, time-ordered shard.
    """
    rng = np.random.default_rng(seed)
    ddir = Path(data_dir) if data_dir else _default_data_dir()
    x, t, sess = _read_all(ddir)

    # Global quintile bins over ranked target => globally balanced classes;
    # per-client skew stays real. Rank-based to tolerate the mass at zero.
    order = np.argsort(np.argsort(t))
    y = np.floor(order / len(t) * N_CLASSES).astype(int)
    y = np.clip(y, 0, N_CLASSES - 1)

    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
    xs = (x - mu) / sd

    # Clients: keep real session boundaries; pool sessions in order and split
    # into n_clients contiguous shards so each client spans one or a few real
    # sessions (distinct cells/mobility/apps => genuine non-IID).
    sess_order = np.argsort(sess, kind="stable")
    xs, y = xs[sess_order], y[sess_order]
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
    n_test = min(int(test_fraction * len(xs)), 5000)  # cap: 5k suffices for accuracy
    test_idx, rest = perm[:n_test], perm[n_test:]
    root_idx = rest[:root_size]
    return (clients, (xs[test_idx], y[test_idx]), (xs[root_idx], y[root_idx]))

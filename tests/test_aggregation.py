import numpy as np

from oran_fed_robust.aggregation import build_aggregator
from oran_fed_robust.aggregation.base import AggregationContext


def _honest_with_outliers(rng, n=10, d=8, n_bad=3):
    updates = rng.normal(0.0, 0.1, size=(n, d)) + 1.0  # honest cluster near 1.0
    updates[:n_bad] = 50.0  # gross outliers
    return updates


def test_fedavg_is_pulled_by_outliers():
    rng = np.random.default_rng(0)
    u = _honest_with_outliers(rng)
    out = build_aggregator("fedavg").aggregate(u)
    assert out.mean() > 5.0  # mean dragged far from honest ~1.0


def test_median_resists_outliers():
    rng = np.random.default_rng(0)
    u = _honest_with_outliers(rng)
    out = build_aggregator("median").aggregate(u)
    assert abs(out.mean() - 1.0) < 0.5


def test_krum_selects_honest_update():
    rng = np.random.default_rng(0)
    u = _honest_with_outliers(rng)
    out = build_aggregator("krum").aggregate(u, AggregationContext(n_byzantine=3))
    assert abs(out.mean() - 1.0) < 0.5


def test_trimmed_mean_resists_outliers():
    rng = np.random.default_rng(0)
    u = _honest_with_outliers(rng)
    out = build_aggregator("trimmed_mean", trim_ratio=0.3).aggregate(u)
    assert abs(out.mean() - 1.0) < 0.5


def test_reputation_is_stateful_and_downweights_persistent_attacker():
    agg = build_aggregator("reputation", beta=0.8)
    rng = np.random.default_rng(0)
    ctx = AggregationContext(client_ids=[0, 1, 2, 3, 4])
    for _ in range(15):
        u = rng.normal(0.0, 0.1, size=(5, 6)) + 1.0
        u[0] = -50.0  # client 0 persistently anti-aligned (sign-flip style)
        agg.aggregate(u, ctx)
    reps = agg.reputations()
    # persistent attacker should end with the lowest reputation
    assert reps[0] == min(reps.values())


def test_reputation_requires_matching_ids():
    agg = build_aggregator("reputation")
    u = np.ones((3, 4))
    try:
        agg.aggregate(u, AggregationContext(client_ids=[0, 1]))
        assert False, "expected ValueError"
    except ValueError:
        pass

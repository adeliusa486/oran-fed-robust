import numpy as np

from oran_fed_robust.data.partition import dirichlet_label_partition


def test_partition_covers_all_samples():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=400)
    parts = dirichlet_label_partition(labels, n_clients=10, alpha=0.5, rng=rng)
    assert len(parts) == 10
    total = sum(len(p) for p in parts)
    assert total == len(labels)


def test_strong_heterogeneity_is_skewed():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 5, size=1000)
    parts = dirichlet_label_partition(labels, n_clients=20, alpha=0.05, rng=rng)
    # with very small alpha, at least one client should be highly class-skewed
    max_skew = 0.0
    for idx in parts:
        if len(idx) == 0:
            continue
        counts = np.bincount(labels[idx], minlength=5) / len(idx)
        max_skew = max(max_skew, counts.max())
    assert max_skew > 0.6

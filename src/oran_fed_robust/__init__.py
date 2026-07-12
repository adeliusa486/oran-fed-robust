"""oran_fed_robust: Heterogeneity-aware robust federated aggregation benchmark for Open-RAN.

This package provides:
  * synthetic Open-RAN-style non-IID data generation,
  * poisoning attacks (sign-flip, label-flip, fabricated-update),
  * robust aggregation rules (FedAvg, Krum, median, trimmed-mean, FLTrust)
    plus the proposed reputation-weighted aggregator,
  * a federated training/evaluation harness.

NOTE: results produced with the bundled synthetic generator are for benchmarking
the *defenses*, not a claim about real operator traffic. Replace the synthetic
data source with OpenRAN Gym / ns-O-RAN exports for real measurements.
"""

__version__ = "0.1.0"

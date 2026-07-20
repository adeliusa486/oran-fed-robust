"""Aggregation rules registry."""
from __future__ import annotations

from .base import Aggregator, AggregationContext
from .fedavg import FedAvg
from .krum import Krum
from .median import CoordinateMedian
from .trimmed_mean import TrimmedMean
from .fltrust import FLTrust
from .reputation import ReputationAggregator
from .dm_trimmed_mean import DirectionMagnitudeTrimmedMean


def build_aggregator(name: str, **kwargs) -> Aggregator:
    name = name.lower()
    registry = {
        "fedavg": FedAvg,
        "krum": Krum,
        "median": CoordinateMedian,
        "trimmed_mean": TrimmedMean,
        "fltrust": FLTrust,
        "reputation": ReputationAggregator,
        "dm_trimmed_mean": DirectionMagnitudeTrimmedMean,
    }
    if name not in registry:
        raise ValueError(f"Unknown aggregator '{name}'. Options: {sorted(registry)}")
    return registry[name](**kwargs)


__all__ = [
    "Aggregator",
    "AggregationContext",
    "FedAvg",
    "Krum",
    "CoordinateMedian",
    "TrimmedMean",
    "FLTrust",
    "ReputationAggregator",
    "DirectionMagnitudeTrimmedMean",
    "build_aggregator",
]

"""Configuration loading and schema.

Uses a lightweight dataclass + PyYAML loader rather than Hydra to keep the
dependency surface minimal. Hydra/OmegaConf can be layered on later without
changing call sites, since :func:`load_config` simply returns a dataclass.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    n_clients: int = 50
    n_features: int = 20
    n_classes: int = 5
    samples_per_client: int = 400
    dirichlet_alpha: float = 0.1
    seed: int = 42


@dataclass
class TrainConfig:
    rounds: int = 50
    participants_per_round: int = 20
    local_epochs: int = 2
    lr: float = 0.05
    batch_size: int = 64
    seed: int = 42


@dataclass
class AttackConfig:
    name: str = "fabricated"  # one of: none, sign_flip, label_flip, fabricated
    compromise_fraction: float = 0.2
    scale: float = 5.0


@dataclass
class AggConfig:
    name: str = "reputation"  # fedavg|krum|median|trimmed_mean|fltrust|reputation
    trim_ratio: float = 0.1
    beta: float = 0.8  # reputation memory
    root_size: int = 100  # FLTrust clean root dataset size


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    agg: AggConfig = field(default_factory=AggConfig)
    output_dir: str = "results"


def _merge(dc: Any, overrides: Dict[str, Any]) -> Any:
    """Recursively apply a nested dict of overrides onto a dataclass instance."""
    for key, value in overrides.items():
        if not hasattr(dc, key):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(dc, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)
    return dc


def load_config(path: str | Path | None = None) -> Config:
    """Load a :class:`Config`, optionally overriding defaults from a YAML file."""
    cfg = Config()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _merge(cfg, raw)


def list_aggregators() -> List[str]:
    return ["fedavg", "krum", "median", "trimmed_mean", "fltrust", "reputation"]


def list_attacks() -> List[str]:
    return ["none", "sign_flip", "label_flip", "fabricated"]

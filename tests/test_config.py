from pathlib import Path

from oran_fed_robust.config import Config, list_aggregators, list_attacks, load_config


def test_default_config():
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.data.n_clients == 50


def test_yaml_override(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  n_clients: 7\ntrain:\n  rounds: 3\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.data.n_clients == 7
    assert cfg.train.rounds == 3
    # untouched keys keep defaults
    assert cfg.data.n_classes == 5


def test_registries():
    assert "reputation" in list_aggregators()
    assert "fabricated" in list_attacks()

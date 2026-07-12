"""Smoke: package and submodules import cleanly."""


def test_top_level_import():
    import oran_fed_robust

    assert oran_fed_robust.__version__


def test_submodule_imports():
    from oran_fed_robust import aggregation, attacks, config, data, evaluation, models, training  # noqa: F401
    from oran_fed_robust.api import app  # noqa: F401

    assert hasattr(app, "app")

# Contributing

Contributions are welcome.

## Development setup

```bash
pip install -e ".[dev]"
PYTHONPATH=src pytest
ruff check src tests scripts
```

## Guidelines

- Keep the `Aggregator` interface stable; add new rules under `aggregation/` and register them in `aggregation/__init__.py`.
- Add a unit test for every new aggregator or attack.
- Use type hints and docstrings; keep functions small and pure where possible.
- Run `ruff` and `pytest` before opening a PR.

## Adding a new aggregator

1. Subclass `Aggregator`, set `name`, implement `aggregate`.
2. Register it in `build_aggregator`.
3. Add a test in `tests/test_aggregation.py`.

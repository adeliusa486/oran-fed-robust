.PHONY: install dev test smoke benchmark api lint clean

install:
	pip install .

dev:
	pip install -e ".[dev]"

test:
	PYTHONPATH=src pytest

smoke:
	PYTHONPATH=src python scripts/run_benchmark.py --quick

benchmark:
	PYTHONPATH=src python scripts/run_benchmark.py --config configs/default.yaml

api:
	PYTHONPATH=src uvicorn oran_fed_robust.api.app:app --host 0.0.0.0 --port 8000

lint:
	ruff check src tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info results/*.csv results/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +

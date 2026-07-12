# Usage Guide

## Run a single configuration programmatically

```python
from oran_fed_robust.aggregation import build_aggregator
from oran_fed_robust.data import generate_federated_dataset
from oran_fed_robust.training import FederatedTrainer

clients, test_set, root_set = generate_federated_dataset(
    n_clients=50, n_features=20, n_classes=5,
    samples_per_client=400, dirichlet_alpha=0.1, seed=42,
)

trainer = FederatedTrainer(
    clients, test_set, root_set,
    aggregator=build_aggregator("reputation", beta=0.8),
    n_features=20, n_classes=5,
    attack_name="fabricated", compromise_fraction=0.2, attack_scale=5.0,
    participants_per_round=20, local_epochs=2, lr=0.05, seed=42,
)
history = trainer.fit(rounds=50, log_every=10)
print("final:", history[-1])
```

## Benchmark sweep

```bash
python scripts/run_benchmark.py --config configs/default.yaml --attacks none sign_flip fabricated label_flip
```
Outputs `results/benchmark.csv` and `results/benchmark.json`.

## API

```bash
uvicorn oran_fed_robust.api.app:app --port 8000
curl -X POST localhost:8000/aggregate \
  -H 'content-type: application/json' \
  -d '{"updates": [[1,1,1],[1.1,0.9,1],[50,50,50]], "client_ids":[0,1,2], "aggregator":"median"}'
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: oran_fed_robust` | run with `PYTHONPATH=src` or `pip install -e .` |
| API import error | ensure `fastapi`, `uvicorn`, `pydantic` installed |
| Slow full sweep | use `--quick`, or lower `rounds`/`n_clients` in the config |

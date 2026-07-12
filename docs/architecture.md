# Architecture

## Components

| Module | Responsibility |
|---|---|
| `data/synthetic.py` | Generate synthetic Open-RAN-style KPM data; non-IID via Dirichlet |
| `data/partition.py` | Dirichlet label partition (base-station heterogeneity) |
| `models/softmax.py` | NumPy softmax classifier with flat-vector params |
| `attacks/poisoning.py` | sign-flip, label-flip, fabricated-update |
| `aggregation/*` | FedAvg, Krum, median, trimmed-mean, FLTrust, reputation |
| `training/federated.py` | Round orchestration, attack routing, evaluation |
| `evaluation/metrics.py` | Accuracy, macro-F1 |
| `api/app.py` | FastAPI aggregation microservice |

## Data flow

```
clients (non-IID) ──local SGD──▶ updates ──[attacks]──▶ aggregator (RIC) ──▶ global model ──broadcast──▶ clients
```

## Aggregation interface

All rules implement `Aggregator.aggregate(updates, ctx) -> vector`, where `updates` is `[n_clients, dim]` and `ctx` (`AggregationContext`) carries optional `client_ids` (stateful reputation), `server_update` (FLTrust), and `n_byzantine` (Krum).

## Swapping the model to PyTorch

Provide an object with `get_params() -> np.ndarray`, `set_params(vec)`, `predict(x)`, and `local_train(...) -> np.ndarray`. The aggregation and training code are framework-agnostic because they operate on flat parameter vectors.

## Swapping in real data

Produce `ClientDataset` objects from real OpenRAN Gym / ns-O-RAN E2 KPM exports and pass them to `FederatedTrainer`. The Dirichlet generator is only the default synthetic source.

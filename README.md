# oran-fed-robust

**Heterogeneity-aware robust federated aggregation: a benchmark and defense for Open-RAN intelligence.**

This repository benchmarks the standard "robust" federated-learning aggregation rules — FedAvg, Krum, coordinate-wise median, trimmed-mean, FLTrust — against model-poisoning attacks under **base-station-level non-IID data**, and provides a **reputation-weighted aggregator** that separates legitimate heterogeneity from adversarial divergence by accumulating evidence across rounds.

> **Scope & honesty note.** The bundled dataset is a *synthetic* Open-RAN-style KPM generator used to study aggregation behaviour reproducibly on any machine. It is a stand-in for OpenRAN Gym / ns-O-RAN exports — swap in real traffic by producing `ClientDataset` objects. Numbers you generate here describe the **defenses on synthetic data**, not real operator traffic.

## Features

- 6 aggregation rules behind one interface (`build_aggregator`).
- 3 poisoning attacks: sign-flip, label-flip (data-level), fabricated-update injection.
- Dirichlet non-IID partitioning with tunable `alpha`.
- Federated training harness + benchmark sweep → CSV/JSON.
- FastAPI microservice exposing the aggregator as a near-RT-RIC-style service.
- Tests, Docker, CI, config-driven design.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start

```bash
# tiny end-to-end sweep (seconds)
PYTHONPATH=src python scripts/run_benchmark.py --quick

# full sweep from config
PYTHONPATH=src python scripts/run_benchmark.py --config configs/default.yaml

# run tests
PYTHONPATH=src pytest

# serve the aggregation API
PYTHONPATH=src uvicorn oran_fed_robust.api.app:app --port 8000
curl localhost:8000/health
```

Or with `make`: `make smoke`, `make test`, `make benchmark`, `make api`.

## How it works

Each round: clients train locally → updates are computed as `local − global` → malicious clients perturb their data (label-flip) or update (sign-flip / fabricated) → the aggregator combines updates → the global model is updated. See [docs/architecture.md](docs/architecture.md).

The proposed **reputation aggregator** computes each client's cosine distance to the coordinate-median reference, maps it through a calibrated suspicion function, and maintains an EWMA reputation per client (memory `beta`). Trust weights are the normalized reputations. Because the state persists across rounds, a single round of honest non-IID divergence does not condemn a client — only persistent anomalies are suppressed.

## Repository layout

```
src/oran_fed_robust/   core package (data, models, attacks, aggregation, training, api)
scripts/               benchmark + data generation entry points
tests/                 unit + smoke + API tests
configs/               YAML experiment configs
docs/                  architecture & usage guides
```

## Using real Open-RAN data

Implement a loader that yields `ClientDataset(client_id, x, y)` per base station from your OpenRAN Gym / ns-O-RAN E2 KPM exports, and pass the list to `FederatedTrainer`. Nothing else changes.

## License

MIT — see [LICENSE](LICENSE).

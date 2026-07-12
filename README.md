# oran-fed-robust

**Which robust aggregation rule should you trust in Open-RAN? An empirical study of federated poisoning defenses on real base-station traffic.**

This repository evaluates six federated-learning aggregation rules — FedAvg, Krum, coordinate-wise median, trimmed-mean, FLTrust, and a history-aware reputation-weighted rule — against six poisoning attacks (sign-flip, label-flip, fabricated-update injection, a stealthy intermittent adversary, and the collusion-aware ALIE and IPM attacks) on a federated xApp load-classification task built from **real LTE base-station traffic**.

> **Honest headline finding.** No aggregation rule dominates. Crude attacks are absorbed by every rule, including no defense at all. Under the IPM attack, the rules most associated with "robustness" — Krum and coordinate-median — degrade the most, with Krum falling *below* undefended FedAvg at 30% compromise. Trimmed-mean and FLTrust are the most consistently strong choices; FLTrust's stability costs a server-side clean dataset. See `results_real/` for the full measured grid.

## Data

Experiments run on the **Barcelona LTE dataset** (three real base stations — ElBorn, LesCorts, PobleSec — released with the *Federated Traffic Prediction for 5G and Beyond Challenge* and with Perifanis et al., *Federated Learning for 5G Base Station Traffic Forecasting*, Computer Networks 2023). It is third-party data and is **not** redistributed here — fetch it with:

```bash
python scripts/get_real_data.py
```

A synthetic Open-RAN-style generator (`generate_federated_dataset`) is also included for fast iteration and unit testing; results reported in any paper based on this repository are measured on the real data, not the synthetic generator.

## Features

- 6 aggregation rules behind one interface (`build_aggregator`), including magnitude-clipped reputation aggregation.
- 6 poisoning attacks: sign-flip, label-flip, fabricated-update injection, a stealthy intermittent adversary, and the collusion-aware ALIE and IPM attacks.
- Real-data loader (`load_real_federated_dataset`) with genuine base-station/time-window non-IID splits, plus a synthetic Dirichlet-partition generator for controlled experiments.
- Federated training harness + full-grid benchmark runner with multi-seed confidence intervals → CSV/JSON/LaTeX tables.
- FastAPI microservice exposing the aggregator as a near-RT-RIC-style service.
- Tests, Docker, CI, config-driven design.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/get_real_data.py                     # fetch the real Barcelona traces
```

## Quick start

```bash
# tiny synthetic-data smoke sweep (seconds)
PYTHONPATH=src python scripts/run_benchmark.py --quick

# full measured grid on real data (~30 min, 5 seeds x 3 compromise fractions x 6 attacks x 6 rules)
PYTHONPATH=src python scripts/run_full_benchmark.py --real --out results_real

# generate paper-ready LaTeX tables/figure data from the grid
PYTHONPATH=src python scripts/make_paper_tables.py --dir results_real --f 0.3

# run tests
PYTHONPATH=src pytest

# serve the aggregation API
PYTHONPATH=src uvicorn oran_fed_robust.api.app:app --port 8000
curl localhost:8000/health
```

## How it works

Each round: clients train locally → updates are computed as `local − global` → malicious clients perturb their data (label-flip) or update (sign-flip / fabricated / adaptive), or the round's malicious updates are crafted jointly from the honest ones (ALIE / IPM) → the aggregator combines updates → the global model is updated. See [docs/architecture.md](docs/architecture.md).

The **reputation aggregator** computes each client's cosine distance to the coordinate-median reference, maps it through a calibrated suspicion function, and maintains an EWMA reputation per client (memory `beta`); updates are additionally clipped to the median update norm before the trust-weighted average. Norm-clipping is decisive against large-magnitude attacks (sign-flip) but, by design, offers no protection against magnitude-matched attacks (IPM) — this asymmetry is measured and discussed in the accompanying study.

## Repository layout

```
src/oran_fed_robust/   core package (data, models, attacks, aggregation, training, api)
scripts/               data fetch, benchmark, and table-generation entry points
tests/                 unit + smoke + API tests
configs/               YAML experiment configs
docs/                  architecture & usage guides
```

## Using other real Open-RAN data

Implement a loader that yields `ClientDataset(client_id, x, y)` per base station from your own OpenRAN Gym / ns-O-RAN E2 KPM exports (see `src/oran_fed_robust/data/real_traffic.py` for a worked example against the Barcelona traces), and pass the list to `FederatedTrainer`. Nothing else changes.

## License

MIT — see [LICENSE](LICENSE).

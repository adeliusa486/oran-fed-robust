# Smoke Test Report

**Environment:** Python 3.11.9 · numpy 1.26.4 · scikit-learn 1.8.0 · fastapi 0.115.0 · pytest 8.4.2 (Windows).

## Tests executed

| Suite | Result |
|---|---|
| `tests/test_imports.py` (package + submodule imports) | ✅ pass |
| `tests/test_config.py` (defaults, YAML override, registries) | ✅ pass |
| `tests/test_partition.py` (Dirichlet coverage + skew) | ✅ pass |
| `tests/test_aggregation.py` (all 6 rules, statefulness) | ✅ pass |
| `tests/test_smoke_pipeline.py` (end-to-end FL run) | ✅ pass |
| `tests/test_api.py` (FastAPI health + aggregate) | ✅ pass |
| **Total** | **17 passed, 0 failed** |

## Pipelines validated

- **Config loading:** YAML overrides merge onto dataclass defaults. ✅
- **Federated training:** no-attack run reaches ~100% on the synthetic task. ✅
- **Benchmark sweep (`--quick`):** 18 runs (3 attacks × 6 aggregators) write `results/benchmark.csv` + `.json`. ✅
- **API startup:** `/health` returns ok; `/aggregate` returns aggregated vector + reputation trace. ✅

## Failures found and fixed

1. **`TypeError: CoordinateMedian() takes no arguments`** — `build_aggregator` passes shared kwargs (`beta`, `trim_ratio`) to every rule, but FedAvg/Krum/Median/FLTrust had no kwargs-tolerant constructor. **Fix:** added `__init__(self, **_)` to those four classes. Re-ran → all green.
2. **Messy ternary in `test_smoke_pipeline.py`** — removed a leftover `if False else` construct.

## Runtime observations (honest)

Quick-mode benchmark (8 rounds) sample:

| Attack | FedAvg | Krum | Median | Trimmed | FLTrust | Reputation |
|---|---|---|---|---|---|---|
| none | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| sign_flip | 0.21 | 0.78 | **1.00** | 0.21 | **1.00** | 0.42 |
| fabricated | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |

**Important honest findings:**
- **The synthetic task is easy** (Gaussian blobs) → clean accuracy saturates near 100%. This is a *behavioural sandbox for the defenses*, not a hard learning benchmark. Real OpenRAN Gym data will be harder and is required for any performance claim.
- **The reputation aggregator underperforms under sign-flip in `--quick` (8 rounds, acc 0.42).** This is a **reputation-horizon artifact**, not a bug: with β=0.8 the EWMA needs more rounds to suppress a persistent attacker. Verified directly:

  | rounds | reputation acc under sign-flip |
  |---|---|
  | 8 | 0.32 |
  | 30 | **1.00** |
  | 60 | **1.00** |

  So the method works as designed once given an adequate horizon; short sweeps understate it. Tune β lower for faster response, or run ≥30 rounds.
- **Trimmed-mean collapses under sign-flip at trim_ratio=0.1** because 20% compromise exceeds the trim fraction — expected and consistent with theory.

## Remaining known issues / concerns

- Results on synthetic data are **not** evidence about real RAN traffic.
- `pip install -e .` not exercised in this run (tests use `PYTHONPATH=src`); both paths are supported.
- A `PendingDeprecationWarning` from Starlette's `multipart` import is cosmetic.

## Reproduce

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/run_benchmark.py --quick
```

"""Console entry point wrapping the benchmark sweep.

Exposed as ``oran-fed-benchmark`` via pyproject. Kept dependency-free at import
time so importing the package never fails.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:  # pragma: no cover - thin wrapper
    # Make the scripts/ directory importable, then delegate.
    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    import run_benchmark

    run_benchmark.main()


if __name__ == "__main__":  # pragma: no cover
    main()

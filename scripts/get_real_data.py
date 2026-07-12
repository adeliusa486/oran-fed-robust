"""Fetch the real Barcelona LTE base-station traces used by data/real_traffic.py.

Source: the "Federated Traffic Prediction for 5G and Beyond Challenge" dataset,
released with Perifanis et al., Federated Learning for 5G Base Station Traffic
Forecasting, Computer Networks (2023). Third-party data, not redistributed in
this repository -- this script clones the upstream repository and copies out
the three station CSVs.

Usage:
    python scripts/get_real_data.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/vperifan/Federated-Time-Series-Forecasting.git"
STATIONS = ("ElBorn", "LesCorts", "PobleSec")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "data" / "barcelona"
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, tmp], check=True)
        src = Path(tmp) / "dataset"
        for station in STATIONS:
            csv = src / f"{station}.csv"
            if not csv.exists():
                print(f"warning: {csv} not found upstream", file=sys.stderr)
                continue
            shutil.copy(csv, out_dir / f"{station}.csv")
            print(f"wrote {out_dir / f'{station}.csv'}")

    print(f"Done. Real federated dataset ready at {out_dir}")


if __name__ == "__main__":
    main()

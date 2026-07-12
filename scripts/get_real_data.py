"""Fetch the two real datasets used by the study (third-party, not redistributed).

* Barcelona LTE base-station traces (data/real_traffic.py) -- released with
  Perifanis et al., Federated Learning for 5G Base Station Traffic Forecasting,
  Computer Networks (2023).
* Raca et al. 5G production-network trace (data/real_5g.py) -- Beyond throughput,
  the next generation: a 5G dataset with channel and context metrics, MMSys 2020.

Usage:
    python scripts/get_real_data.py            # fetch both
    python scripts/get_real_data.py --barcelona
    python scripts/get_real_data.py --fiveg
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

BARCELONA_REPO = "https://github.com/vperifan/Federated-Time-Series-Forecasting.git"
STATIONS = ("ElBorn", "LesCorts", "PobleSec")
FIVEG_REPO = "https://github.com/uccmisl/5Gdataset.git"


def _data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def fetch_barcelona() -> None:
    out_dir = _data_root() / "barcelona"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", BARCELONA_REPO, tmp], check=True)
        src = Path(tmp) / "dataset"
        for station in STATIONS:
            csv = src / f"{station}.csv"
            if not csv.exists():
                print(f"warning: {csv} not found upstream", file=sys.stderr)
                continue
            shutil.copy(csv, out_dir / f"{station}.csv")
            print(f"wrote {out_dir / f'{station}.csv'}")
    print(f"Barcelona dataset ready at {out_dir}")


def fetch_fiveg() -> None:
    out_dir = _data_root() / "fiveg"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--depth", "1", FIVEG_REPO, tmp], check=True)
        zips = list(Path(tmp).glob("*.zip"))
        if not zips:
            print("warning: no dataset zip found in 5G repo", file=sys.stderr)
            return
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(out_dir)
        n = len(list(out_dir.rglob("*.csv")))
        print(f"5G dataset ready at {out_dir} ({n} session CSVs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--barcelona", action="store_true")
    ap.add_argument("--fiveg", action="store_true")
    args = ap.parse_args()
    both = not (args.barcelona or args.fiveg)
    if args.barcelona or both:
        fetch_barcelona()
    if args.fiveg or both:
        fetch_fiveg()


if __name__ == "__main__":
    main()

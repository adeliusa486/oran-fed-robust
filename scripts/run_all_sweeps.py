"""Run all benchmark sweeps, confirmation sweeps, tables, and figures sequentially."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMMANDS = [
    ["python", "scripts/run_full_benchmark.py", "--dataset", "barcelona", "--out", "results_real"],
    ["python", "scripts/run_full_benchmark.py", "--dataset", "fiveg", "--out", "results_5g"],
    ["python", "scripts/run_full_benchmark.py", "--dataset", "lumos5g", "--out", "results_lumos"],
    ["python", "scripts/run_mlp_confirm.py"],
    ["python", "scripts/make_paper_tables.py", "--dir", "results_real", "--f", "0.3"],
    ["python", "scripts/make_paper_tables.py", "--dir", "results_5g", "--f", "0.3"],
    ["python", "scripts/make_paper_tables.py", "--dir", "results_lumos", "--f", "0.3"],
    ["python", "scripts/make_figures.py"],
    ["python", "scripts/make_convergence.py"],
]

def main():
    t_start = time.time()
    print("=== Starting all benchmark sweeps and figure generation ===", flush=True)
    for i, cmd in enumerate(COMMANDS, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(COMMANDS)}] Running: {' '.join(cmd)}", flush=True)
        res = subprocess.run(cmd, cwd=ROOT)
        if res.returncode != 0:
            print(f"ERROR: Command {' '.join(cmd)} failed with code {res.returncode}", flush=True)
            sys.exit(res.returncode)
        print(f"--> Done in {time.time()-t0:.1f}s", flush=True)
    print(f"\n=== ALL SUCCEEDED in {(time.time()-t_start)/60:.2f} minutes ===", flush=True)

if __name__ == "__main__":
    main()

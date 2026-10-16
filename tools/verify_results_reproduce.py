"""Re-run each stored pareto sweep and compare it with the stored result CSV.

Fields are compared numerically within --tol, since thread-pool summation order
changes the last bit of a mean. --fix overwrites stale stored files.

    python tools/verify_results_reproduce.py                  # everything
    python tools/verify_results_reproduce.py --only hospital  # one dataset
    python tools/verify_results_reproduce.py --fix
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# RUNS: (stored dir, dataset, backend, extra args); the backend tag is written into
# the result file, so it must match the stored one.
COMMON = ["--experiment", "pareto", "--alphas", "0.05", "0.1", "0.2",
          "--seeds", "10", "--fast-verify", "on"]

SMALL = ("hospital", "beers", "flights", "rayyan")
THREE = ("hospital", "beers", "flights")


def _raha(ds):
    return ["--scoring", "detected", "--detection", f"log:csvs/raha_{ds}_detected.csv"]


RUNS = (
    [("results_baran_errors", d, f"baran:csvs/baran_{d}_mapped.csv", ["--scoring", "errors"])
     for d in SMALL + ("tax",)]
    + [("results_baran_detected", d, f"baran:csvs/baran_{d}_mapped.csv", ["--scoring", "detected"])
       for d in SMALL + ("tax",)]
    + [("results_bclean", d, f"bclean:csvs/bclean_{d}_mapped.csv", ["--scoring", "errors"]) for d in THREE]
    + [("results_bclean_keep_oracle", d, f"bclean_keep:csvs/bclean_keep_{d}_mapped.csv",
        ["--scoring", "detected"]) for d in THREE]
    + [("results_bclean_plus", d, f"bclean_plus:csvs/bclean_plus_{d}_mapped.csv", ["--scoring", "errors"])
       for d in THREE]
    + [("results_holoclean", d, f"holoclean:csvs/holoclean_{d}_mapped.csv", ["--scoring", "errors"])
       for d in THREE]
    + [("results_holoclean_keep_oracle", d, f"holoclean_keep:csvs/holoclean_keep_{d}_mapped.csv",
        ["--scoring", "detected"]) for d in THREE]
    + [("results_gidcl", d, f"gidcl:csvs/gidcl_{d}_mapped.csv", ["--scoring", "errors"])
       for d in SMALL + ("tax",)]
    + [("results_gidcl_owndet", d, f"gidcl:csvs/gidcl_{d}_mapped.csv",
        ["--scoring", "detected", "--detection", f"log:csvs/gidcl_{d}_detected.csv"])
       for d in SMALL + ("tax",)]
    + [("results_zeroec", d, f"zeroec:csvs/zeroec_{d}_mapped.csv", ["--scoring", "errors"]) for d in SMALL]
    + [("results_raha_baran", d, f"baran_raha:csvs/baran_raha_{d}_mapped.csv", _raha(d)) for d in SMALL]
    + [("results_raha_baran", "tax", "baran:csvs/baran_tax_mapped.csv", _raha("tax"))]
    + [("results_raha_bclean", d, f"bclean_keep:csvs/bclean_keep_{d}_mapped.csv", _raha(d)) for d in THREE]
    + [("results_raha_holoclean", d, f"holoclean_keep:csvs/holoclean_keep_{d}_mapped.csv", _raha(d))
       for d in THREE]
    + [("results_constraints_baran", d, f"baran:csvs/baran_{d}_mapped.csv",
        ["--scoring", "detected", "--detection", "constraints"]) for d in ("hospital", "flights")]
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument("--fix", action="store_true", help="rewrite stale stored files")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="numeric tolerance; below this a difference is float noise")
    args = ap.parse_args()
    os.chdir(ROOT)

    def compare(stored, fresh, tol=1e-9):
        """Field-by-field comparison with numeric tolerance ``tol``."""
        A, B = list(csv.DictReader(open(stored))), list(csv.DictReader(open(fresh)))
        if len(A) != len(B):
            return False, f"row count {len(A)} vs {len(B)}"
        for a, b in zip(A, B):
            if set(a) != set(b):
                return False, "column set differs"
            for k in a:
                x, y = a[k], b[k]
                if x == y:
                    continue
                try:
                    if abs(float(x) - float(y)) <= tol:
                        continue
                except (TypeError, ValueError):
                    pass
                return False, f"{a.get('baseline','?')}/{a.get('strata','-')} alpha={a.get('alpha')} {k}: {x} vs {y}"
        return True, ""

    same, stale, missing = [], [], []
    for rdir, ds, backend, extra in RUNS:
        if args.only and ds not in args.only:
            continue
        stored = os.path.join("experiments", rdir, f"{ds}_pareto.csv")
        label = f"{rdir}/{ds}"
        if not os.path.exists(stored):
            missing.append(label)
            print(f"  MISSING  {label}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, "-m", "bench.run_study", "--dataset", ds,
                   "--backends", backend, *COMMON, *extra, "--out", tmp]
            env = dict(os.environ, CARE_PROGRESS="0")
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            fresh = os.path.join(tmp, f"{ds}_pareto.csv")
            if p.returncode != 0 or not os.path.exists(fresh):
                missing.append(label)
                print(f"  FAILED   {label}: {p.stderr.strip().splitlines()[-1:]}")
                continue
            equal, why = compare(stored, fresh, tol=args.tol)
            if equal:
                same.append(label)
                print(f"  ok       {label}")
            else:
                stale.append(label)
                print(f"  DIFFERS  {label}\n             {why}")
                if args.fix:
                    shutil.copyfile(fresh, stored)
                    print("             rewritten (--fix)")

    print(f"\nreproduce: {len(same)}   differ: {len(stale)}   missing/failed: {len(missing)}")
    if stale:
        print("Stale files feed make_figures.py -- regenerate, then check which "
              "reported numbers moved before trusting the paper.")
    return 1 if stale or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

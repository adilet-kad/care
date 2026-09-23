"""Re-run every sweep behind the paper and diff it against the stored result CSV.

Reproducibility asks one question above all: does running the documented
command reproduce the reported number? This answers it mechanically. Each entry
in ``RUNS`` names a stored result file and the exact ``bench.run_study``
invocation that should regenerate it; the script runs each into a temporary
directory and compares field by field, numerically. The comparison is deliberately
not byte-exact: the sweep sums per-seed scores across a thread pool, so the last
bit of a mean varies between runs, and flagging those one-ULP artefacts as failures
would train you to ignore the tool. `--tol` controls the threshold.

Why this exists as a tool rather than a test: a full pass re-runs every sweep in
the paper, which is minutes for the small corpora and considerably longer for
tax, so it cannot live in CI. Run it before submitting an artifact, and after any
change to the proposer, verifier or certifier.

    python tools/verify_results_reproduce.py                 # everything
    python tools/verify_results_reproduce.py --only hospital # one dataset
    python tools/verify_results_reproduce.py --fix           # rewrite stale files

``--fix`` regenerates the stored file in place. Use it deliberately: if a stored
result no longer reproduces, one of two things is true, and they need different
responses. Either the code changed and the paper's number is now stale (regenerate,
then re-run make_figures.py and check which reported numbers moved), or the code
regressed (find the regression instead). The script cannot tell these apart, so it
never rewrites anything unless asked.
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

# (stored dir, dataset, extra args). The backend tag must match the one used
# originally, because it is written into the result file's last column.
COMMON = ["--experiment", "pareto", "--alphas", "0.05", "0.1", "0.2",
          "--seeds", "10", "--fast-verify", "on"]

RUNS = [
    ("results_baran_errors", "hospital", "baran:csvs/baran_hospital_mapped.csv", ["--scoring", "errors"]),
    ("results_baran_errors", "beers", "baran:csvs/baran_beers_mapped.csv", ["--scoring", "errors"]),
    ("results_baran_errors", "flights", "baran:csvs/baran_flights_mapped.csv", ["--scoring", "errors"]),
    ("results_baran_errors", "rayyan", "baran:csvs/baran_rayyan_mapped.csv", ["--scoring", "errors"]),
    ("results_baran_errors", "tax", "baran:csvs/baran_tax_mapped.csv", ["--scoring", "errors"]),
    ("results_baran_detected", "hospital", "baran:csvs/baran_hospital_mapped.csv", ["--scoring", "detected"]),
    ("results_baran_detected", "beers", "baran:csvs/baran_beers_mapped.csv", ["--scoring", "detected"]),
    ("results_baran_detected", "flights", "baran:csvs/baran_flights_mapped.csv", ["--scoring", "detected"]),
    ("results_baran_detected", "rayyan", "baran:csvs/baran_rayyan_mapped.csv", ["--scoring", "detected"]),
    ("results_holoclean", "hospital", "holoclean:csvs/holoclean_hospital_mapped.csv", ["--scoring", "errors"]),
    ("results_bclean", "hospital", "bclean:csvs/bclean_hospital_mapped.csv", ["--scoring", "errors"]),
    ("results_bclean", "beers", "bclean:csvs/bclean_beers_mapped.csv", ["--scoring", "errors"]),
    ("results_bclean", "flights", "bclean:csvs/bclean_flights_mapped.csv", ["--scoring", "errors"]),
    ("results_detected", "hospital", "jellyfish:csvs/jellyfish_hospital_mapped.csv", ["--scoring", "detected"]),
    ("results_detected", "beers", "jellyfish:csvs/jellyfish_beers_mapped.csv", ["--scoring", "detected"]),
    ("results_detected", "flights", "jellyfish:csvs/jellyfish_flights_mapped.csv", ["--scoring", "detected"]),
    ("results_detected", "rayyan", "jellyfish:csvs/jellyfish_rayyan_mapped.csv", ["--scoring", "detected"]),
    ("results_gpt4o", "hospital", "gpt4o:csvs/gpt4o_hospital_mapped.csv", ["--scoring", "detected"]),
    ("results_gpt4o", "beers", "gpt4o:csvs/gpt4o_beers_mapped.csv", ["--scoring", "detected"]),
    ("results_gpt4o", "flights", "gpt4o:csvs/gpt4o_flights_mapped.csv", ["--scoring", "detected"]),
    ("results_gpt4o", "rayyan", "gpt4o:csvs/gpt4o_rayyan_mapped.csv", ["--scoring", "detected"]),
    ("results_raha_baran", "hospital", "baran_raha:csvs/baran_raha_hospital_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_hospital_detected.csv"]),
    ("results_raha_baran", "beers", "baran_raha:csvs/baran_raha_beers_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_beers_detected.csv"]),
    ("results_raha_baran", "flights", "baran_raha:csvs/baran_raha_flights_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_flights_detected.csv"]),
    ("results_raha_baran", "rayyan", "baran_raha:csvs/baran_raha_rayyan_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_rayyan_detected.csv"]),
    # HoloClean's keep log carries two provenance tags, so the sweep auto-adds
    # strata=mondrian_src and the stored file has 15 rows rather than 12. If a rerun
    # produces 12, the exporter regressed to a single SOURCE and D23's guard is off.
    ("results_holoclean_keep_oracle", "hospital", "holoclean_keep:csvs/holoclean_keep_hospital_mapped.csv",
     ["--scoring", "detected"]),
    # beers and flights use denial constraints we wrote (deviation: not upstream's);
    # see integrations/holoclean/constraints/README.md.
    ("results_holoclean", "beers", "holoclean:csvs/holoclean_beers_mapped.csv",
     ["--scoring", "errors"]),
    ("results_holoclean", "flights", "holoclean:csvs/holoclean_flights_mapped.csv",
     ["--scoring", "errors"]),
    ("results_holoclean_keep_oracle", "beers", "holoclean_keep:csvs/holoclean_keep_beers_mapped.csv",
     ["--scoring", "detected"]),
    ("results_holoclean_keep_oracle", "flights", "holoclean_keep:csvs/holoclean_keep_flights_mapped.csv",
     ["--scoring", "detected"]),
    ("results_raha_holoclean", "beers", "holoclean_keep:csvs/holoclean_keep_beers_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_beers_detected.csv"]),
    ("results_raha_holoclean", "flights", "holoclean_keep:csvs/holoclean_keep_flights_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_flights_detected.csv"]),
    ("results_raha_holoclean", "hospital", "holoclean_keep:csvs/holoclean_keep_hospital_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_hospital_detected.csv"]),
    ("results_bclean_keep_oracle", "hospital", "bclean_keep:csvs/bclean_keep_hospital_mapped.csv", ["--scoring", "detected"]),
    ("results_bclean_keep_oracle", "beers", "bclean_keep:csvs/bclean_keep_beers_mapped.csv", ["--scoring", "detected"]),
    ("results_bclean_keep_oracle", "flights", "bclean_keep:csvs/bclean_keep_flights_mapped.csv", ["--scoring", "detected"]),
    ("results_raha_bclean", "hospital", "bclean_keep:csvs/bclean_keep_hospital_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_hospital_detected.csv"]),
    ("results_raha_bclean", "beers", "bclean_keep:csvs/bclean_keep_beers_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_beers_detected.csv"]),
    ("results_raha_bclean", "flights", "bclean_keep:csvs/bclean_keep_flights_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_flights_detected.csv"]),
    ("results_raha_jelly", "hospital", "jellyfish:csvs/jellyfish_hospital_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_hospital_detected.csv"]),
    ("results_raha_jelly", "beers", "jellyfish:csvs/jellyfish_beers_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_beers_detected.csv"]),
    ("results_raha_jelly", "flights", "jellyfish:csvs/jellyfish_flights_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_flights_detected.csv"]),
    ("results_raha_jelly", "rayyan", "jellyfish:csvs/jellyfish_rayyan_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_rayyan_detected.csv"]),
    ("results_constraints_baran", "flights", "baran:csvs/baran_flights_mapped.csv",
     ["--scoring", "detected", "--detection", "constraints"]),
    ("results_constraints_baran", "hospital", "baran:csvs/baran_hospital_mapped.csv",
     ["--scoring", "detected", "--detection", "constraints"]),
    ("results_constraints_jelly", "flights", "jellyfish:csvs/jellyfish_flights_mapped.csv",
     ["--scoring", "detected", "--detection", "constraints"]),
    ("results_retclean", "flights", "retclean:csvs/retclean_flights_mapped.csv", ["--scoring", "detected"]),
    # tax: the Jellyfish run is a 25,000-row shard, so the sweep must be restricted to
    # the same rows or the 105,591 unseen rows score as escalations.
    ("results_jelly_tax25k", "tax", "jellyfish:csvs/jellyfish_tax_mapped.csv",
     ["--scoring", "detected", "--row-range", "0", "25000"]),
    ("results_baran_detected", "tax", "baran:csvs/baran_tax_mapped.csv", ["--scoring", "detected"]),
    # tax's non-oracle row uses a reduced detector; see docs/REPRODUCE.md D18/D19.
    ("results_raha_baran", "tax", "baran:csvs/baran_tax_mapped.csv",
     ["--scoring", "detected", "--detection", "log:csvs/raha_tax_detected.csv"]),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument("--fix", action="store_true", help="rewrite stale stored files")
    ap.add_argument("--tol", type=float, default=1e-9,
                    help="numeric tolerance; below this a difference is float noise")
    args = ap.parse_args()
    os.chdir(ROOT)

    def compare(stored, fresh, tol=1e-9):
        """Numeric comparison, not byte comparison.

        The sweep dispatches proposals across a thread pool, so the order in which
        per-seed scores are summed varies between runs and the last bit of a mean
        can differ. Those are one-ULP artefacts (~2e-16), not result changes, and a
        byte comparison reports them as failures -- which trains you to ignore the
        tool. Anything above `tol` is a real difference and is shown.
        """
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

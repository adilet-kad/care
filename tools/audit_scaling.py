"""Verification and calibration cost as the table grows (first N rows of tax).

Measures FD index build time and memory, per-candidate incremental verification time
for 1-3 FD sets, and calibration time for 1, 10 and ~100 strata on Baran's repairs.

    python tools/audit_scaling.py   -> experiments/scaling.csv
    python tools/audit_scaling.py --sizes 10000 50000
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
import tracemalloc

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CARE_PROGRESS", "0")

FD_SETS = {1: [(["zip"], ["city"])],
           2: [(["zip"], ["city", "state"])],
           3: [(["zip"], ["city", "state"]), (["area_code"], ["state"])]}
SIZES = [10_000, 25_000, 50_000, 100_000, 200_000]


def _subsample(n, tmp):
    d = os.path.join(tmp, "tax")
    os.makedirs(d, exist_ok=True)
    for f in ("dirty.csv", "clean.csv"):
        with open(f"data/tax/{f}", encoding="utf-8") as src, \
                open(os.path.join(d, f), "w", encoding="utf-8") as dst:
            for i, line in enumerate(src):
                if i > n:
                    break
                dst.write(line)
    return tmp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sizes", nargs="*", type=int, default=SIZES)
    ap.add_argument("--out", default="experiments/scaling.csv")
    a = ap.parse_args()
    from bench.log_proposer import LogProposer
    from bench.datasets import load
    from care.conformal import ConformalController, by_column, marginal_strata
    from care.core.artifact import CellKey, DQRequirements
    from care.core.registry import ConstraintRegistry, clear_registry, register_constraint
    from care.core.repair import CalibrationRecord, Violation
    from care.predicates import ALL_CONSTRAINTS
    from care.verify.incremental import IncrementalVerifier

    clear_registry()
    for c in ALL_CONSTRAINTS:
        register_constraint(c)
    reg = ConstraintRegistry.from_requirements(
        DQRequirements(enabled={"dq.completeness.floor", "dq.consistency.fd"}), strict=True)

    rows = []
    for n in a.sizes:
        tmp = tempfile.mkdtemp(prefix="care_scale_")
        try:
            art, defects, _ = load("tax", data_dir=_subsample(n, tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        cells = sum(1 for _ in art.iter_cells())
        proposer = LogProposer("csvs/baran_tax_mapped.csv", source_id="baran")
        cands = []
        for ref in defects.gold:
            cands.extend(proposer.propose(Violation(cid="q", refs=[ref]), art, options={}))
        row = {"rows": n, "cells": cells, "candidates": len(cands)}
        for k, fds in FD_SETS.items():
            options = {"dq.consistency.fd": {"fds": fds}}
            tracemalloc.start()
            t0 = time.perf_counter()
            inc = IncrementalVerifier(art, reg, options)
            t_build = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            t0 = time.perf_counter()
            out = [inc.assess(c) for c in cands]
            t_inc = time.perf_counter() - t0
            row[f"fd{k}_build_s"] = round(t_build, 3)
            row[f"fd{k}_index_mb"] = round(peak / 2**20, 1)
            row[f"fd{k}_verify_us"] = round(1e6 * t_inc / max(1, len(cands)), 2)
            row[f"fd{k}_rejected"] = sum(1 for r in out if r is not None and not r.feasible)
            del inc
        # calibration wall-clock: threshold search over the fixed grid, all strata
        gold = {k: str(v).lower().strip() for k, v in defects.gold.items()}
        ctl0 = ConformalController(bound="exact")
        reps = {c.target_ref: c for c in ctl0.assign_s_hat(cands) if c.target_ref in gold}
        refs = sorted(reps)
        col = by_column()
        strata_fns = {1: marginal_strata, 10: col,
                      100: lambda r: f"{col(r)}:{int(CellKey.parse(r.target_ref).row_id) % 10}"}
        for m, sf in strata_fns.items():
            import random
            rng = random.Random(0)
            cal = rng.sample(refs, int(0.4 * len(refs)))
            recs = [CalibrationRecord(target_ref=r, s_hat=reps[r].s_hat,
                                      correct=(str(reps[r].proposed_value).lower().strip() == gold[r]),
                                      stratum=sf(reps[r])) for r in cal]
            ctl = ConformalController(bound="exact", strata_fn=sf)
            t0 = time.perf_counter()
            thr = ctl.calibrate(recs, alpha=0.1, delta=0.1)
            t_cal = time.perf_counter() - t0
            row[f"strata{m}_n"] = len(thr.thresholds)
            row[f"strata{m}_labels"] = len(recs)
            row[f"strata{m}_calibrate_s"] = round(t_cal, 4)
        rows.append(row)
        print({k: v for k, v in row.items()}, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

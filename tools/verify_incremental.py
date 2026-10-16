"""Incremental verification under FDs: agreement with the exact verifier, and cost.

Compares feasible, s_margin and delta_objective of IncrementalVerifier.assess and
Verifier.assess on Baran's candidates, and times both. The FDs hold on the clean
tables (--check-fds).

    python tools/verify_incremental.py hospital --sample 0   # every candidate
    python tools/verify_incremental.py tax --sample 30
    -> experiments/incremental_verify.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CARE_PROGRESS", "0")

FDS = {
    "hospital": [(["ZipCode"], ["City", "State", "CountyName"]),
                 (["PhoneNumber"], ["ZipCode", "City"]),
                 (["ProviderNumber"], ["HospitalName", "PhoneNumber"]),
                 (["HospitalName"], ["PhoneNumber"]),
                 (["MeasureCode"], ["MeasureName", "Condition"]),
                 (["City"], ["CountyName"])],
    "tax": [(["zip"], ["city", "state"]),
            (["area_code"], ["state"])],
}
LOGS = {"hospital": "csvs/baran_hospital_mapped.csv", "tax": "csvs/baran_tax_mapped.csv"}
OUT = "experiments/incremental_verify.csv"


def check_fds(ds):
    import collections
    rows = list(csv.DictReader(open(f"data/{ds}/clean.csv", newline="", encoding="utf-8")))
    for det, deps in FDS[ds]:
        for dep in deps:
            g = collections.defaultdict(set)
            for r in rows:
                g[tuple(r[d] for d in det)].add(r[dep])
            bad = sum(1 for v in g.values() if len(v) > 1)
            print(f"  {'+'.join(det)} -> {dep}: {len(g)} groups, {bad} inconsistent in clean")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", choices=sorted(FDS))
    ap.add_argument("--log", default=None)
    ap.add_argument("--sample", type=int, default=0,
                    help="candidates checked on the exact path (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-fds", action="store_true")
    a = ap.parse_args()
    if a.check_fds:
        check_fds(a.dataset)

    from bench.log_proposer import LogProposer
    from bench.study import load
    from care.core.artifact import DQRequirements
    from care.core.registry import ConstraintRegistry, clear_registry, register_constraint
    from care.predicates import ALL_CONSTRAINTS
    from care.verify.incremental import IncrementalVerifier
    from care.verify.projector import hard_violation_set
    from care.verify.verifier import Verifier

    clear_registry()
    for c in ALL_CONSTRAINTS:
        register_constraint(c)
    reg = ConstraintRegistry.from_requirements(
        DQRequirements(enabled={"dq.completeness.floor", "dq.consistency.fd"}), strict=True)
    options = {"dq.consistency.fd": {"fds": FDS[a.dataset]}}

    art, defects, _ = load(a.dataset)
    n_cells = sum(1 for _ in art.iter_cells())
    proposer = LogProposer(a.log or LOGS[a.dataset], source_id="baran")
    from care.core.repair import Violation
    cands = []
    for ref in defects.gold:
        cands.extend(proposer.propose(Violation(cid="q", refs=[ref]), art, options={}))
    n_fd = sum(len(deps) for _, deps in FDS[a.dataset])
    print(f"{a.dataset}: {n_cells:,} cells, {len(FDS[a.dataset])} FDs ({n_fd} dependent "
          f"columns), {len(cands):,} candidates")

    t0 = time.perf_counter()
    inc = IncrementalVerifier(art, reg, options)
    t_build = time.perf_counter() - t0
    print(f"  incremental index built and self-checked in {t_build:.2f} s "
          f"({len(inc.before):,} existing hard violations)")

    t0 = time.perf_counter()
    inc_out = [inc.assess(c) for c in cands]
    t_inc = time.perf_counter() - t0
    undecided = sum(r is None for r in inc_out)
    rejected = sum(1 for r in inc_out if r is not None and not r.feasible)
    print(f"  incremental: {t_inc:.2f} s for all candidates "
          f"({1e6 * t_inc / max(1, len(cands)):.1f} us each), {undecided} undecided, "
          f"{rejected:,} rejected by a hard constraint")

    idx = list(range(len(cands)))
    if a.sample and a.sample < len(idx):
        idx = sorted(random.Random(a.seed).sample(idx, a.sample))
    t0 = time.perf_counter()
    before = hard_violation_set(art, reg, options)          # computed once
    t_before = time.perf_counter() - t0
    exact = Verifier()
    mism, n_ex, t_ex = 0, 0, 0.0
    for k, i in enumerate(idx):
        t0 = time.perf_counter()
        e = exact.assess(cands[i], art, reg, options=options, before=before)
        t_ex += time.perf_counter() - t0
        n_ex += 1
        r = inc_out[i]
        if r is None or (r.feasible, r.s_margin, r.delta_objective) != \
                (e.feasible, e.s_margin, e.delta_objective):
            mism += 1
            print(f"  MISMATCH {cands[i].target_ref}: exact "
                  f"{(e.feasible, e.s_margin, e.delta_objective)} vs incremental "
                  f"{None if r is None else (r.feasible, r.s_margin, r.delta_objective)}")
        if (k + 1) % 10 == 0 and t_ex > 60:
            print(f"  exact path: {k + 1}/{len(idx)} ({t_ex:.0f} s)", flush=True)
    per_ex = t_ex / max(1, n_ex)
    per_inc = t_inc / max(1, len(cands))
    print(f"  exact path: {n_ex} candidates in {t_ex:.1f} s ({per_ex:.3f} s each, plus "
          f"{t_before:.1f} s hoisted scan); projected for all {len(cands):,}: "
          f"{per_ex * len(cands) / 3600:.2f} h")
    print(f"  IDENTICAL on {n_ex - mism}/{n_ex} compared candidates; "
          f"speed-up per candidate {per_ex / per_inc:,.0f}x")

    row = {"dataset": a.dataset, "cells": n_cells, "fds": len(FDS[a.dataset]),
           "dependent_cols": n_fd, "candidates": len(cands),
           "existing_violations": len(inc.before), "rejected": rejected,
           "undecided": undecided, "build_s": round(t_build, 3),
           "incremental_total_s": round(t_inc, 3), "incremental_us_per_cand":
           round(1e6 * per_inc, 2), "exact_compared": n_ex, "mismatches": mism,
           "exact_s_per_cand": round(per_ex, 4), "exact_projected_h":
           round(per_ex * len(cands) / 3600, 3)}
    os.makedirs("experiments", exist_ok=True)
    new = not os.path.exists(OUT)
    rows = [] if new else [r for r in csv.DictReader(open(OUT)) if r["dataset"] != a.dataset]
    rows.append(row)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {OUT}")
    return 1 if mism else 0


if __name__ == "__main__":
    raise SystemExit(main())

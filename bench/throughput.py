"""bench.throughput -- what does the governance layer itself cost?

The proposer's cost is not CARE's cost. An LLM pass over a table is seconds per
cell and dwarfs everything else, so quoting end-to-end wall-clock would measure
vLLM, not this system. What a reader needs to know before putting a certifier in
front of a database is the *marginal* cost of governing a repair that has already
been proposed. This script times only the parts CARE is responsible for, with the
proposer stubbed out to a dictionary lookup over an existing log.

Four phases are timed separately because they scale differently, and lumping them
together would hide the one that actually matters:

  load       parse the table into the artifact.                        O(|D|)
  scan       the hoisted hard-constraint violation set. Computed ONCE per run
             and reused for every candidate -- this is the hoisting described in
             Section 4.1, and it is why verification is O(|D| + m) rather than
             O(m|D|). Recomputing it per candidate made tax ~25 s per cell.      O(|D|)
  govern     per-candidate verify + score. This is the marginal cost per repair
             and the number to quote when asking "can I afford this?"            O(m)
  certify    calibrate the fixed grid, then decide every test candidate. The
             grid is swept once per stratum, independent of table size.          O(m + |A||L|)

Reported per-candidate figures are the honest unit: the scan is amortised across
however many repairs a run governs, so a table with one repair pays for a whole
scan and a table with 100k repairs pays almost nothing per repair.

    python -m bench.throughput --datasets beers hospital flights \\
      --out experiments/throughput.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

from bench.study import _registry, detect, load, propose_repairs
from care.conformal import by_column
from care.core import CalibrationRecord
from care.conformal.controller import ConformalController

# One log per dataset: the proposer is replayed from disk so no model runs.
DEFAULT_LOGS = {
    "beers": "csvs/baran_beers_mapped.csv",
    "hospital": "csvs/baran_hospital_mapped.csv",
    "flights": "csvs/baran_flights_mapped.csv",
    "tax": "csvs/baran_tax_mapped.csv",
}


def _time(fn, repeat=1):
    """Best-of-`repeat` wall clock. Best, not mean: we are measuring the work, and
    a slower sample only ever means the sandbox was doing something else."""
    best = float("inf")
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


def measure(dataset, log, *, alpha=0.1, delta=0.1, max_cells=None, fast_verify=True,
            exact_sample=60):
    from care.verify.projector import hard_violation_set

    (art, defects, _), t_load = _time(lambda: load(dataset))
    n_cells = sum(1 for _ in art.iter_cells())
    reg = _registry()

    _before, t_scan = _time(lambda: hard_violation_set(art, reg, {}))

    refs = detect(art, reg, defects, mode="oracle")
    if max_cells:
        refs = refs[:max_cells]

    from bench.baran_proposer import BaranProposer
    proposer = BaranProposer(log, source_id="bench")

    # The two verification paths of Section 4.1, measured against each other. This
    # is the only comparison that isolates what hoisting buys, because both paths
    # produce bit-identical output (tools/verify_fastpath_equivalence.py).
    #
    #   exact  re-derives the hard-violation set per candidate: O(m|D|), so its
    #          per-repair cost grows with the TABLE, not the queue. Timed on a small
    #          sample because doing the full queue takes minutes.
    #   fast   reuses the hoisted set: O(|D| + m). Its per-repair cost should be
    #          flat in |D|.
    #
    # The fast path's fixed 64-cell precondition probe is timed separately and
    # subtracted. Leaving it in would be a measurement error, not conservatism: it
    # is paid once per run regardless of m, so on hospital's 499 candidates it alone
    # accounts for ~10 ms of an apparent 10.6 ms "per repair".
    from bench.study import _probe_verifier_constant
    from care.verify import Verifier
    verifier = Verifier()

    exact_refs = refs[:min(exact_sample, len(refs))]
    _e, t_exact = _time(lambda: propose_repairs(
        art, exact_refs, proposer, reg=reg, max_workers=1, fast_verify=False))
    ms_exact = 1000 * t_exact / max(len(exact_refs), 1)

    _p, t_probe = _time(lambda: _probe_verifier_constant(
        art, refs, proposer, verifier, reg, {}, _before, n_cells=n_cells))
    repairs, t_fast_total = _time(lambda: propose_repairs(
        art, refs, proposer, reg=reg, max_workers=1, fast_verify=True))
    m_all = max(len(refs), 1)
    ms_fast = 1000 * max(t_fast_total - t_probe, 0.0) / m_all
    t_govern = t_fast_total

    # certify: calibrate the grid on 40% then decide the rest, as evaluate_split does.
    gold = {k: str(v).lower().strip() for k, v in defects.gold.items()}
    keys = [r for r in repairs if r in gold]
    random.Random(0).shuffle(keys)
    cut = int(len(keys) * 0.4)
    cal, test = keys[:cut], keys[cut:]
    sf = by_column()
    ctl = ConformalController(bound="exact", strata_fn=sf)
    records = [CalibrationRecord(target_ref=r, s_hat=repairs[r].s_hat,
                                 correct=(str(repairs[r].proposed_value).lower().strip()
                                          == gold[r]),
                                 stratum=sf(repairs[r])) for r in cal]
    thr, t_cal = _time(lambda: ctl.calibrate(records, alpha=alpha, delta=delta), repeat=3)
    test_repairs = [repairs[r] for r in test]
    _dec, t_dec = _time(lambda: ctl.decide(test_repairs, thr, strata_fn=sf), repeat=3)

    m = len(repairs) or 1
    return {
        "dataset": dataset,
        "cells": n_cells,
        "candidates": len(repairs),
        "load_s": round(t_load, 3),
        # One hard constraint is enabled (a completeness floor), so this is a floor
        # on scan cost, not a claim about arbitrary constraint sets.
        "scan_s": round(t_scan, 4),
        "scan_violations": len(_before),
        "scan_cells_per_s": round(n_cells / t_scan) if t_scan else 0,
        "exact_sample": len(exact_refs),
        "exact_ms_per_repair": round(ms_exact, 3),
        "probe_s": round(t_probe, 3),
        "fast_total_s": round(t_govern, 3),
        "fast_ms_per_repair": round(ms_fast, 4),
        "fast_repairs_per_s": round(1000 / ms_fast) if ms_fast > 0 else 0,
        "speedup": round(ms_exact / ms_fast, 1) if ms_fast > 0 else 0,
        "calibrate_s": round(t_cal, 4),
        "decide_s": round(t_dec, 4),
        "decide_us_per_repair": round(1e6 * t_dec / max(len(test), 1), 1),
        "certify_s": round(t_cal + t_dec, 4),
        # What a run of this size actually paid per governed repair, everything in.
        "amortised_ms_per_repair": round(1000 * (t_scan + t_govern + t_cal + t_dec) / m, 3),
        "fast_verify": bool(fast_verify),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=["beers", "hospital", "flights"])
    ap.add_argument("--logs", nargs="*", default=None,
                    help="override logs as dataset=path")
    ap.add_argument("--max-cells", type=int, default=None,
                    help="cap candidates (use on tax; the scan is still full-table)")
    ap.add_argument("--fast-verify", default="on", choices=["on", "off"])
    ap.add_argument("--exact-sample", type=int, default=60,
                    help="candidates timed on the exact path (it is minutes for a full queue)")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--out", default="experiments/throughput.csv")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    logs = dict(DEFAULT_LOGS)
    for kv in args.logs or []:
        k, v = kv.split("=", 1)
        logs[k] = v

    rows = []
    for ds in args.datasets:
        log = logs[ds]
        if not os.path.exists(log):
            print(f"  [skip] {ds}: missing {log}")
            continue
        print(f"\n===== {ds} =====", flush=True)
        rows.append(measure(ds, log, alpha=args.alpha, delta=args.delta,
                            max_cells=args.max_cells, exact_sample=args.exact_sample,
                            fast_verify=(args.fast_verify == "on")))
        print("  " + "  ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)

    if not rows:
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    existing = args.append and os.path.exists(args.out)
    with open(args.out, "a" if existing else "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not existing:
            wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Seal a certifiability prediction before running the audit, then score it.

The ceiling identity -- that pi_0 predicts certifiable automation -- currently rests
on datasets this literature has tuned against for years. Reporting it again there
cannot separate a relationship that was found from one that was predicted. Committing
to a number on an untouched dataset before the audit runs can, provided the result is
reported either way.

This tool makes that commitment mechanical rather than a matter of good intentions.

    # BEFORE the audit: label a uniform sample, seal the prediction
    python tools/preregister_pi0.py seal --dataset <new-dataset> \\
        --log csvs/baran_<new-dataset>_mapped.csv --sample 200 --alpha 0.2

    # AFTER the full audit: score the sealed prediction against what happened
    python tools/preregister_pi0.py score --prediction experiments/prereg/*.json \\
        --results experiments/results_baran_errors/<new-dataset>_pareto.csv

`seal` writes a JSON carrying the sampled refs, the estimate, an interval, and a
SHA-256 over the log file and the sample. `score` refuses to run if the log has
changed since sealing, so a prediction cannot be quietly re-fitted to the outcome.

WHAT IS PREDICTED, AND WHY IT IS NOT JUST pi_0
----------------------------------------------
pi_0 is the fraction of repairs lying in columns the proposer gets *entirely* right.
From a sample you cannot observe that: a column wrong on one cell in a thousand looks
perfect in a sample of twenty. Predicting the naive sample pi_0 would therefore
predict a ceiling the data cannot support, and would be optimistic by construction.

So the sealed prediction is what CARE *would certify given only these labels*: the
fraction of proposals in columns whose Clopper-Pearson upper bound on the error rate,
computed from the sample at level delta/|Lambda|, falls at or below alpha. That is the
operator's actual screening question -- "is it worth commissioning a full evaluation?"
-- and it is answerable for a few hundred labels. Both quantities are recorded; the
CP-screened one is the prediction, the naive one is kept so the gap between them is
visible rather than hidden.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _norm(v):
    return str(v).strip().lower() if v is not None else ""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def seal(args) -> int:
    from bench.datasets import load
    from care.conformal import by_cardinality, by_column
    from care.conformal.bounds import clopper_pearson_upper as cp
    from care.core.artifact import CellKey as _CK

    art, defects, _ = load(args.dataset)
    gold = {k: _norm(v) for k, v in defects.gold.items()}
    cur = {f"{c.row_id}::{c.col}": _norm(c.value) for c in art.iter_cells()}

    rows = []
    with open(args.log, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            ref = f"{r['row_id']}::{r['column']}"
            if ref in gold or ref in cur:
                rows.append((ref, r["column"], _norm(r.get("value"))))
    if not rows:
        raise SystemExit(f"no proposals in {args.log} match {args.dataset}'s cells")

    # The screen must partition the way the certifier will, or it predicts a ceiling
    # for a stratification nobody runs. CARE coarsens to cardinality buckets when the
    # median column holds too few error cells to calibrate; a column-wise screen on
    # such a dataset splits the labels across strata far too thin and predicts zero
    # for a configuration that in fact certifies most of its repairs.
    _cnt = collections.Counter(_CK.parse(r).col for r in gold)
    _median = sorted(_cnt.values())[len(_cnt) // 2] if _cnt else 0
    strata_fn = by_column() if _median >= 60 else by_cardinality(art)
    # The stratifiers accept a bare ref string, so no candidate object is needed.
    stratum_of = {ref: strata_fn(ref) for ref, _c, _v in rows}

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.sample, len(rows)))

    # "Labelling" the sample means consulting the gold standard, which is what an
    # operator does by hand. Only the SAMPLED cells inform the prediction; the rest of
    # the gold set must not leak into it, or the exercise is circular.
    per, wrong = collections.Counter(), collections.Counter()
    for ref, col, val in sample:
        truth = gold.get(ref, cur.get(ref))
        st = stratum_of[ref]
        per[st] += 1
        wrong[st] += val != truth

    total_props = collections.Counter(stratum_of[r] for r, _, _ in rows)
    n_props = sum(total_props.values())
    n_lambda = max(1, len(per))
    delta_c = args.delta / n_lambda

    screened, naive = [], []
    detail = {}
    for col in per:
        ub = cp(wrong[col], per[col], delta_c)
        detail[col] = {"sampled": per[col], "wrong": wrong[col],
                       "cp_upper": round(ub, 4),
                       "proposals": total_props.get(col, 0)}
        if ub <= args.alpha:
            screened.append(col)
        if wrong[col] == 0:
            naive.append(col)

    pred = sum(total_props.get(c, 0) for c in screened) / n_props
    naive_pi0 = sum(total_props.get(c, 0) for c in naive) / n_props

    rec = {
        "sealed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": args.dataset, "log": args.log,
        "log_sha256": _sha256(args.log),
        "sample_size": len(sample), "sample_seed": args.seed,
        "alpha": args.alpha, "delta": args.delta, "n_lambda": n_lambda,
        "predicted_automation": round(pred, 4),
        "naive_sample_pi0": round(naive_pi0, 4),
        "n_proposals": n_props,
        "strata_screened_in": sorted(screened),
        "per_stratum": detail,
        "stratifier": ("column" if _median >= 60 else "cardinality"),
        "sample_refs_sha256": hashlib.sha256(
            "\n".join(sorted(r for r, _, _ in sample)).encode()).hexdigest(),
        "note": ("Prediction is the fraction of proposals in columns whose sampled "
                 "Clopper-Pearson upper bound is <= alpha. Sealed before the audit; "
                 "report the outcome whether or not it matches."),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)

    print(f"sealed -> {args.out}")
    print(f"  labelled {len(sample)} of {n_props} proposals across {n_lambda} columns")
    print(f"  strata screened in  : {len(screened)} of {n_lambda} "
          f"({'column' if _median >= 60 else 'cardinality'} partition)")
    print(f"  PREDICTED A({args.alpha}) = {pred:.4f}")
    print(f"  (naive sample pi_0  = {naive_pi0:.4f}; higher because an unobserved "
          f"error cannot lower it)")
    print("\nCommit this file before running the audit.")
    return 0


def score(args) -> int:
    paths = sorted(glob.glob(args.prediction))
    if not paths:
        raise SystemExit(f"no prediction file at {args.prediction}")
    ok_all = True
    for p in paths:
        rec = json.load(open(p))
        print(f"\n=== {os.path.basename(p)}  ({rec['dataset']}) ===")
        print(f"  sealed {rec['sealed_utc']}")
        cur_hash = _sha256(rec["log"]) if os.path.exists(rec["log"]) else None
        if cur_hash != rec["log_sha256"]:
            print("  !! THE LOG HAS CHANGED SINCE SEALING. The prediction was made "
                  "against different\n     data, so this comparison is void. Re-seal "
                  "on the current log, or restore it.")
            ok_all = False
            continue
        # Compare like with like. The screen predicts what GROUP-CONDITIONAL control
        # will certify; marginal control answers a different question and on some
        # datasets certifies far more, so taking the max over strata would score the
        # prediction against a quantity it never claimed to predict.
        best = None
        with open(args.results, newline="") as fh:
            for r in csv.DictReader(fh):
                if (r["baseline"] == "CARE"
                        and abs(float(r["alpha"]) - rec["alpha"]) < 1e-9
                        and r["strata"] in ("mondrian", "mondrian_src")):
                    a = 1.0 - float(r["human_cost"])
                    if best is None or a > best:
                        best = a
        if best is None:
            print(f"  no CARE row at alpha={rec['alpha']} in {args.results}")
            ok_all = False
            continue
        pred = rec["predicted_automation"]
        print(f"  predicted A({rec['alpha']}) = {pred:.4f}   "
              f"measured = {best:.4f}   error = {best - pred:+.4f}")
        print("  " + ("PREDICTION HELD" if abs(best - pred) <= args.tol else
                      f"PREDICTION MISSED by more than {args.tol}"))
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seal", help="label a sample and seal a prediction")
    s.add_argument("--dataset", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--sample", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--alpha", type=float, default=0.2)
    s.add_argument("--delta", type=float, default=0.1)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=seal)

    c = sub.add_parser("score", help="score a sealed prediction against the audit")
    c.add_argument("--prediction", required=True)
    c.add_argument("--results", required=True)
    c.add_argument("--tol", type=float, default=0.05)
    c.set_defaults(fn=score)

    args = ap.parse_args()
    if args.cmd == "seal" and not args.out:
        stem = os.path.basename(args.log).replace("_mapped.csv", "").replace(".csv", "")
        args.out = os.path.join("experiments", "prereg", f"{stem}_a{args.alpha}.json")
    os.chdir(ROOT)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

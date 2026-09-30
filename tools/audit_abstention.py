"""audit_abstention.py -- decompose what an abstaining proposer actually did.

A single "accuracy" number is the wrong summary for a log that carries abstentions,
and reading one produced a wrong conclusion on the first verify-correct run: beers
looked like 0.000 accuracy, i.e. a proposer that had become useless, when in fact 40
of its 79 proposals on error cells contained the gold value inside a longer string
and 87 of its 182 proposals were imputations into cells that were correctly empty.
Those are three different phenomena and only one of them is an accuracy failure.

So this reports five disjoint buckets over the cells the proposer did NOT abstain on:

  exact        on a true error cell, value == gold after lowercase+strip. The only
               bucket the paper's accuracy figure should count.
  contains     on a true error cell, gold appears as a whole token inside the value
               ("12 oz. Keg" against gold "12"). Wrong under the exact-match protocol
               (deviation D10) but not a modelling failure; reported separately so a
               reader can see how much of the error is the metric.
  wrong        on a true error cell, neither of the above.
  harm_empty   wrote into a cell that was correctly EMPTY -- null imputation. Damage,
               but a distinct pathology from overwriting real content, and the same
               one that got HoloClean/beers excluded from Table 1.
  harm_value   overwrote a non-empty, already-correct cell. The costly kind.

plus two diagnostics that are NOT accuracy:

  malformed    the value contains a reasoning marker ("0 -> 3", "no change needed").
               Which side is the answer is unrecoverable (max_tokens truncates some),
               so these are counted, never guessed at. `export_jellyfish.py` now
               abstains on them at generation time; logs exported before that fix
               still carry them, which is why this bucket exists.
  recall       fraction of the dataset's error cells the proposer chose to act on.
               With an abstention gate this is the gate's recall, and it is the price
               paid for whatever the harm columns show.

Every bucket is reported twice: over all non-abstained rows, and with `malformed`
excluded, so a reader can see that the conclusion does not depend on how they are
treated.

    python tools/audit_abstention.py --logs "csvs/jellyfish_vc_*_mapped.csv"
    python tools/audit_abstention.py --logs "csvs/jellyfish_*_mapped.csv" --examples 5
    -> experiments/abstention.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REASON = re.compile(r"(?:->|=>|→|\bno change needed\b)", re.IGNORECASE)


def _norm(v):
    return str(v).strip().lower() if v is not None else ""


def split_log_name(path, known):
    """('jellyfish_vc', 'beers') from 'csvs/jellyfish_vc_beers_mapped.csv'."""
    stem = os.path.basename(path)
    for suf in ("_mapped.csv", "_pred.csv", ".csv"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    for ds in sorted(known, key=len, reverse=True):
        if stem.endswith("_" + ds):
            return stem[: -len(ds) - 1], ds
    return None, None


def contains_gold(value, gold):
    """Gold appears as a whole token inside value -- right content, wrong form."""
    if not gold:
        return False
    return re.search(rf"(^|\W){re.escape(gold)}(\W|$)", value) is not None


def audit(path, art, gold, examples=0):
    cur = {f"{c.row_id}::{c.col}": _norm(c.value) for c in art.iter_cells()}
    b = collections.Counter()
    ex = collections.defaultdict(list)
    n_rows = 0
    for r in csv.DictReader(open(path, newline="", encoding="utf-8", errors="replace")):
        n_rows += 1
        raw = r.get("value") or ""
        if not raw.strip():
            b["abstain"] += 1
            continue
        v = _norm(raw)
        ref = f"{r['row_id']}::{r['column']}"
        mal = bool(REASON.search(raw))
        if mal:
            b["malformed"] += 1
        if ref in gold:
            g = gold[ref]
            if v == g:
                k = "exact"
            elif contains_gold(v, g):
                k = "contains"
            else:
                k = "wrong"
            b[k] += 1
            if mal:
                b[k + "_mal"] += 1
        elif ref in cur:
            k = "harm_empty" if cur[ref] == "" else "harm_value"
            b[k] += 1
            if mal:
                b[k + "_mal"] += 1
        else:
            b["unmatched"] += 1
            k = "unmatched"
        if examples and len(ex[k]) < examples:
            ex[k].append((r["row_id"], r["column"], cur.get(ref, "?"), raw,
                          gold.get(ref, "(not an error)")))
    b["rows"] = n_rows
    b["on_err"] = b["exact"] + b["contains"] + b["wrong"]
    b["proposals"] = n_rows - b["abstain"]
    return b, ex


def main() -> int:
    from bench.datasets import DATASETS, load

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--examples", type=int, default=0,
                    help="print N sample rows per bucket (use to sanity-check a claim)")
    ap.add_argument("--out", default="experiments/abstention.csv")
    args = ap.parse_args()
    os.chdir(ROOT)

    paths = sorted({p for g in args.logs for p in glob.glob(g)})
    if not paths:
        raise SystemExit(f"no logs matched {args.logs}")

    cache, rows = {}, []
    for p in paths:
        tag, ds = split_log_name(p, DATASETS)
        if not ds:
            print(f"  [skip] {p}: no registered dataset matches the name")
            continue
        if ds not in cache:
            art, defects, _ = load(ds)
            cache[ds] = (art, {k: _norm(v) for k, v in defects.gold.items()})
        art, gold = cache[ds]
        b, ex = audit(p, art, gold, args.examples)
        n_err = len(gold)
        rows.append({
            "proposer": tag, "dataset": ds, "log_rows": b["rows"],
            "error_cells": n_err, "abstentions": b["abstain"],
            "abstain_frac": round(b["abstain"] / max(b["rows"], 1), 4),
            "proposals": b["proposals"],
            "on_error_cells": b["on_err"],
            "gate_recall": round(b["on_err"] / max(n_err, 1), 4),
            "gate_precision": round(b["on_err"] / max(b["proposals"], 1), 4),
            "exact": b["exact"], "contains_gold": b["contains"], "wrong": b["wrong"],
            "acc_exact": round(b["exact"] / max(b["on_err"], 1), 4),
            "acc_lenient": round((b["exact"] + b["contains"]) / max(b["on_err"], 1), 4),
            "harm_empty": b["harm_empty"], "harm_value": b["harm_value"],
            "malformed": b["malformed"], "unmatched": b["unmatched"],
            # same accuracy with malformed rows removed from both parts of the ratio
            "acc_exact_nomal": round(
                (b["exact"] - b["exact_mal"]) / max(b["on_err"] - b["exact_mal"]
                                                    - b["contains_mal"] - b["wrong_mal"], 1), 4),
            "log": p,
        })
        for k, items in sorted(ex.items()):
            for rid, col, dirty, prop, g in items:
                print(f"  [{ds}/{k}] row {rid} {col}: dirty={dirty!r} "
                      f"proposed={prop!r} gold={g!r}")

    if not rows:
        raise SystemExit("no log produced a result")
    rows.sort(key=lambda r: (r["proposer"], r["dataset"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'proposer':<14}{'dataset':<10}{'abst%':>7}{'props':>7}{'onErr':>7}"
          f"{'recall':>8}{'prec':>7} | {'exact':>6}{'cont':>6}{'wrong':>6}"
          f"{'accX':>7}{'accL':>7} | {'h_emp':>6}{'h_val':>6}{'malf':>6}")
    for r in rows:
        print(f"{r['proposer']:<14}{r['dataset']:<10}{r['abstain_frac']*100:>6.1f}%"
              f"{r['proposals']:>7}{r['on_error_cells']:>7}{r['gate_recall']:>8.3f}"
              f"{r['gate_precision']:>7.3f} | {r['exact']:>6}{r['contains_gold']:>6}"
              f"{r['wrong']:>6}{r['acc_exact']:>7.3f}{r['acc_lenient']:>7.3f} | "
              f"{r['harm_empty']:>6}{r['harm_value']:>6}{r['malformed']:>6}")
    print("\naccX = exact match on error cells (the paper's protocol, D10)")
    print("accL = lenient: exact OR gold appears inside the value (how much is the metric)")
    print("h_emp = wrote into a correctly-EMPTY cell; h_val = overwrote real content")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

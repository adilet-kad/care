"""Calibration of each cleaner's confidence: mean confidence against accuracy, per log.

Reports the pooled gap (mean confidence - accuracy) and ECE over equal-width bins;
bins with fewer than min_bin records are excluded from ECE. A proposal is correct if
it matches the gold value, or, on a clean cell, if it preserves the current value.

    python audit_calibration.py
    -> experiments/calibration.csv, experiments/calibration_bins.csv
"""

from __future__ import annotations

import csv
import glob
import re
import os
import sys

OUT = "experiments"
DATASETS = ("hospital", "beers", "flights", "rayyan", "tax")

# file prefix -> display name
PROPOSER = {
    "baran_raha": "Baran (Raha)",
    "baran": "Baran",
    "bclean_keep": "BClean (+keep)",
    "bclean_plus": "BClean+",
    "bclean": "BClean",
    "gidcl_scored": "GIDCL",          # GIDCL's own repair-selection score (seed 0)
    "holoclean_keep": "HoloClean (+keep)",
    "zeroec": "ZeroEC",               # seed-agreement confidence over three runs
    "holoclean": "HoloClean",
}
# skip GIDCL's retrained-detector seed variants
_SEED_VARIANT = re.compile(r"_s\d+_")


def _proposer_of(stem):
    for p in sorted(PROPOSER, key=len, reverse=True):   # longest prefix wins
        if stem.startswith(p):
            return PROPOSER[p]
    return None


def _dataset_of(stem):
    for d in DATASETS:
        if d in stem:
            return d
    return None


def records(path, ds, gold, cur, colmap):
    """(confidence, correct) for every non-empty proposal that can be scored."""
    out = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            v = (r.get("value") or "").strip().lower()
            if not v:
                continue                      # empty value: nothing to score
            ref = f"{r['row_id']}::{colmap.get(r['column'], r['column'])}"
            if ref in gold:
                ok = v == str(gold[ref]).lower().strip()
            elif ref in cur:
                ok = v == cur[ref]            # clean cell: correct iff preserved
            else:
                continue
            try:
                c = float(r["confidence"])
            except (KeyError, ValueError, TypeError):
                continue
            if 0.0 <= c <= 1.0:
                out.append((c, ok))
    return out


def calibrate(recs, nbins=10, min_bin=30):
    """Return (bins, ece, gap). bins = [(lo, hi, n, mean_conf, acc)]."""
    buckets = [[] for _ in range(nbins)]
    for c, ok in recs:
        i = min(int(c * nbins), nbins - 1)
        buckets[i].append((c, ok))
    bins, ece, mass = [], 0.0, 0
    for i, b in enumerate(buckets):
        lo, hi = i / nbins, (i + 1) / nbins
        if not b:
            continue
        mc = sum(c for c, _ in b) / len(b)
        ac = sum(o for _, o in b) / len(b)
        bins.append((lo, hi, len(b), mc, ac))
        if len(b) >= min_bin:
            ece += len(b) * abs(ac - mc)
            mass += len(b)
    ece = ece / mass if mass else float("nan")
    gap = (sum(c for c, _ in recs) / len(recs)) - (sum(o for _, o in recs) / len(recs))
    return bins, ece, gap


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_log import COL_MAPS
    from bench.datasets import load

    cache = {}

    def truth(ds):
        if ds not in cache:
            art, d, _ = load(ds)
            gold = {k: (str(v).lower().strip() if v is not None else v)
                    for k, v in d.gold.items()}
            cur = {f"{c.row_id}::{c.col}": (str(c.value).lower().strip()
                                            if c.value is not None else "")
                   for c in art.iter_cells()}
            cache[ds] = (gold, cur)
        return cache[ds]

    rows, binrows = [], []
    for path in sorted(glob.glob("csvs/*_pred.csv")):
        stem = os.path.basename(path)[:-len("_pred.csv")]
        if "_ni_" in stem:                       # Ni et al. variants have their own gold
            continue
        if _SEED_VARIANT.search(stem):
            continue
        if stem == "gidcl_scored_flights":   # retrained detector has F1 0 here
            continue
        prop, ds = _proposer_of(stem), _dataset_of(stem)
        if not prop or not ds:
            continue
        try:
            gold, cur = truth(ds)
        except Exception as e:                              # noqa: BLE001
            print(f"  skip {stem}: {e}")
            continue
        recs = records(path, ds, gold, cur, COL_MAPS.get(ds, {}))
        if not recs:
            continue
        bins, ece, gap = calibrate(recs)
        conf = [c for c, _ in recs]
        rows.append({
            "proposer": prop, "dataset": ds, "log": os.path.basename(path),
            "n": len(recs), "distinct_conf": len(set(conf)),
            "mean_conf": round(sum(conf) / len(conf), 4),
            "accuracy": round(sum(o for _, o in recs) / len(recs), 4),
            "gap": round(gap, 4), "ece": round(ece, 4),
        })
        for lo, hi, n, mc, ac in bins:
            binrows.append({"proposer": prop, "dataset": ds, "bin_lo": lo,
                            "bin_hi": hi, "n": n, "mean_conf": round(mc, 4),
                            "accuracy": round(ac, 4)})

    os.makedirs(OUT, exist_ok=True)
    for name, data in (("calibration.csv", rows), ("calibration_bins.csv", binrows)):
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
        print(f"wrote {OUT}/{name} ({len(data)} rows)")

    rows.sort(key=lambda r: -r["gap"])
    print(f"\n{'proposer':<16}{'dataset':<10}{'n':>8}{'conf':>8}{'acc':>8}{'gap':>8}{'ECE':>8}")
    for r in rows:
        print(f"{r['proposer']:<16}{r['dataset']:<10}{r['n']:>8}{r['mean_conf']:>8.3f}"
              f"{r['accuracy']:>8.3f}{r['gap']:>+8.3f}{r['ece']:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

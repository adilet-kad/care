"""AUROC of GIDCL's repair-selection score for separating correct from wrong repairs.

Per (dataset, seed): AUROC (ties count one half), mean score per class, and number of
distinct scores, over every repair GIDCL writes. Scores come from
integrations/gidcl/score_gidcl.py. Flights is omitted: its retrained detector has F1 0.

    python audit_gidcl_confidence.py      -> experiments/gidcl_confidence.csv
"""
from __future__ import annotations

import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
csv.field_size_limit(sys.maxsize)
DATASETS = ["hospital", "beers", "rayyan", "tax"]
OUT = os.path.join(ROOT, "experiments", "gidcl_confidence.csv")
_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _canon(v):
    s = (str(v) if v is not None else "").strip().lower()
    if s in ("", "none", "nan", "null", "n/a", "na", "?", "empty"):
        return ""
    if _NUM.match(s):
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    return s


def _auroc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    s, i, n = 0.0, 0, len(allv)
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        s += sum(avg for k in range(i, j) if allv[k][1])
        i = j
    return (s - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    from bench.datasets import load
    rows = []
    for ds in DATASETS:
        art, defects, _ = load(ds)
        gold = {k: _canon(v) for k, v in defects.gold.items()}
        cur = {f"{c.row_id}::{c.col}": _canon(c.value) for c in art.iter_cells()}
        for seed in range(0, 10):
            tag = "gidcl_scored" if seed == 0 else f"gidcl_scored_s{seed}"
            path = os.path.join(ROOT, "csvs", f"{tag}_{ds}_mapped.csv")
            if not os.path.exists(path):          # tax has no mapped log
                path = path.replace("_mapped.csv", "_pred.csv")
            if not os.path.exists(path):
                continue
            ok, bad = [], []
            for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
                if not (r.get("value") or "").strip() or r.get("confidence") in ("", None):
                    continue
                ref = f"{r['row_id']}::{r['column']}"
                truth = gold.get(ref, cur.get(ref))
                if truth is None:
                    continue
                (ok if _canon(r["value"]) == truth else bad).append(float(r["confidence"]))
            if not ok or not bad:
                continue
            rows.append({"dataset": ds, "seed": seed, "n": len(ok) + len(bad),
                         "n_wrong": len(bad), "auroc": round(_auroc(ok, bad), 4),
                         "mean_conf_correct": round(sum(ok) / len(ok), 4),
                         "mean_conf_wrong": round(sum(bad) / len(bad), 4),
                         "distinct_conf": len(set(ok + bad))})
            print(f"  {ds:9s} seed {seed}: AUROC {rows[-1]['auroc']:.3f}  "
                  f"(n={rows[-1]['n']}, wrong={len(bad)})")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

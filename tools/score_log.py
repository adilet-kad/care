"""Repair precision, recall and F1 of a mapped cleaner log, exact and format-tolerant.

Exact: a repair is correct iff it equals the normalised clean value. Format-tolerant
also treats cells that differ only in numeric format ('0.0' vs '0') as clean.

    python tools/score_log.py tax csvs/gidcl_tax_mapped.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv.field_size_limit(sys.maxsize)
_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _n(v):
    return (str(v) if v is not None else "").strip().lower()


def _canon(v):
    s = _n(v)
    if s in ("", "none"):
        return ""
    if _NUM.match(s):
        f = float(s)
        return str(int(f)) if f == int(f) else repr(f)
    return s


def _prf(fixed, proposals, errors):
    p = fixed / proposals if proposals else 0.0
    r = fixed / errors if errors else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("log")
    a = ap.parse_args()
    from bench.datasets import load
    art, defects, _ = load(a.dataset)
    gold = {k: _n(v) for k, v in defects.gold.items()}
    cur = {f"{c.row_id}::{c.col}": _n(c.value) for c in art.iter_cells()}
    with open(a.log, newline="", encoding="utf-8", errors="replace") as fh:
        log = [(f"{r['row_id']}::{r['column']}", _n(r.get("value")))
               for r in csv.DictReader(fh) if _n(r.get("value"))]
    fixed = sum(1 for ref, v in log if ref in gold and v == gold[ref])
    tol = {r for r, g in gold.items() if _canon(cur.get(r, "")) != _canon(g)}
    fixed_t = sum(1 for ref, v in log if ref in tol and _canon(v) == _canon(gold[ref]))
    P, R, F = _prf(fixed, len(log), len(gold))
    Pt, Rt, Ft = _prf(fixed_t, len(log), len(tol))
    print(f"{a.dataset:8s} proposals={len(log)}")
    print(f"  exact           errors={len(gold):6d} fixed={fixed:6d}  P={P:.3f} R={R:.3f} F1={F:.3f}")
    print(f"  format-tolerant errors={len(tol):6d} fixed={fixed_t:6d}  P={Pt:.3f} R={Rt:.3f} F1={Ft:.3f}")


if __name__ == "__main__":
    main()

"""audit_variants.py -- the Jellyfish interface and scale variants (Section 8.4).

Two questions a reader of Table 1 asks about its LLM rows: would a model with an
abstention do less damage, and would a larger model of the same family be certifiable?
Both were run under the published protocol otherwise unchanged (same correction prompt,
same five-sample self-consistency, same datasets):

  jellyfish        8B,  published protocol: correct every cell            (Table 1)
  jellyfish_vc     8B,  verify-correct: the model's own error-detection question gates
                        correction; cells it does not flag are logged abstentions
  jellyfish_13b    13B, published protocol
  jellyfish_vc13b  13B, verify-correct

For each (variant, dataset) this script reads the mapped log and the gold/current values
and writes, per row: proposals (non-empty values after schema mapping), error cells
touched, errors fixed, accuracy on touched error cells, pi_0 over touched error cells,
correct cells rewritten ("broken"), and the net effect fixes - breaks. It then joins the
CARE sweep at alpha = 0.2 (group-conditional control, oracle detection) from
experiments/results_<dir>/, so the certifier's verdict on each log sits beside it.

    python audit_variants.py                     -> experiments/variants.csv

`make_figures.tab7` draws Table 4 from the same logs; this CSV carries the fuller set of
per-variant numbers (proposals, recall of the model's own detector, A(0.2)) that the
text of Section 8.4 quotes.

Abstention rows (empty value) are neither proposals nor touches, matching
bench.baran_proposer and audit_proposers.
"""
from __future__ import annotations

import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATASETS = ["hospital", "beers", "flights", "rayyan"]
VARIANTS = [
    # (tag, label, results dir under experiments/)
    ("jellyfish",       "8B",                 "results_detected"),
    ("jellyfish_vc",    "8B, verify-correct", "results_jellyfish_vc"),
    ("jellyfish_13b",   "13B",                "results_jelly13b"),
    ("jellyfish_vc13b", "13B, verify-correct", "results_jellyfish_vc13b"),
]
EMPTY = 0.999


def _norm(v):
    return str(v).lower().strip() if v is not None else ""


def _sweep(rdir, ds, alpha=0.2):
    """(A, realised error) under mondrian control at alpha, or None if no sweep."""
    p = os.path.join(ROOT, "experiments", rdir, f"{ds}_pareto.csv")
    if not os.path.exists(p):
        return None
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        if (r.get("baseline") == "CARE" and r.get("strata") == "mondrian"
                and abs(float(r["alpha"]) - alpha) < 1e-9):
            h = float(r["human_cost"])
            return (1 - h, None if h >= EMPTY else float(r["realized_error"]))
    return None


def main():
    from bench.datasets import load
    out_rows = []
    for ds in DATASETS:
        art, defects, _ = load(ds)
        gold = {k: _norm(v) for k, v in defects.gold.items()}
        cur = {f"{c.row_id}::{c.col}": _norm(c.value) for c in art.iter_cells()}
        for tag, label, rdir in VARIANTS:
            path = os.path.join(ROOT, "csvs", f"{tag}_{ds}_mapped.csv")
            if not os.path.exists(path):
                print(f"  missing {path}; skipped")
                continue
            props = err_t = err_f = clean_t = clean_b = 0
            per, ok = collections.Counter(), collections.Counter()
            for r in csv.DictReader(open(path, newline="", encoding="utf-8", errors="replace")):
                v = _norm(r.get("value"))
                if not v:
                    continue
                ref = f"{r['row_id']}::{r['column']}"
                props += 1
                if ref in gold:
                    err_t += 1
                    hit = v == gold[ref]
                    err_f += hit
                    per[r["column"]] += 1
                    ok[r["column"]] += hit
                elif ref in cur:
                    clean_t += 1
                    clean_b += v != cur[ref]
            pi0 = sum(per[c] for c in per if ok[c] == per[c]) / err_t if err_t else 0.0
            sw = _sweep(rdir, ds)
            out_rows.append({
                "dataset": ds, "variant": tag, "label": label, "proposals": props,
                "errors_total": len(gold), "errors_touched": err_t, "errors_fixed": err_f,
                "acc_on_touched": f"{err_f / err_t:.4f}" if err_t else "",
                "pi0": f"{pi0:.4f}", "clean_touched": clean_t, "clean_broken": clean_b,
                "net": err_f - clean_b,
                "A02": f"{sw[0]:.4f}" if sw else "",
                "R02": f"{sw[1]:.4f}" if sw and sw[1] is not None else "",
            })
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    with open(os.path.join(ROOT, "experiments", "variants.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"  experiments/variants.csv ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()

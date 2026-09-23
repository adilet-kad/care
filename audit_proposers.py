"""audit_proposers.py -- measure what each proposer actually did, before CARE sees it.

Writes experiments/proposer_accuracy.csv. Every column is a direct measurement on the
proposer's own log against the benchmark's ground truth; nothing here involves CARE.

Why this file exists
--------------------
CARE's job is to decide which of a proposer's repairs are safe to apply. Its behaviour
is therefore only interpretable relative to how good the proposer was in the first
place. Several (proposer, dataset) pairs produce a selective error of exactly 0.000,
and a reader is entitled to ask whether that is a bug. It is not: it is what happens
when the proposer is exactly right on the cells CARE certifies. This audit makes that
checkable by reporting, per pair:

  acc_on_errors   accuracy on cells the benchmark marks as erroneous (the cells that
                  matter -- proposing the status quo on a clean cell is not a repair)
  n_perfect_cols  number of columns where the proposer is right on every error cell
  frac_in_perfect fraction of proposed error-cell repairs living in those columns
                  -- this is the automation ceiling a column-stratified certifier can
                  reach while remaining exactly correct
  clean_touched   clean cells the proposer also rewrote (collateral damage risk)
  clean_broken    of those, how many it changed to something else

A pair with acc_on_errors == 1.0 is DEGENERATE: the proposer solves the benchmark, so
CARE certifying everything is the correct-but-uninformative answer. Such pairs are
reported as sanity checks, never as evidence that CARE adds value.
"""
from __future__ import annotations

import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASETS = ["hospital", "beers", "flights", "rayyan", "tax"]
PROPOSERS = [
    ("Baran",       "csvs/baran_%s_mapped.csv"),
    ("BClean",      "csvs/bclean_%s_mapped.csv"),
    ("Jellyfish",   "csvs/jellyfish_%s_mapped.csv"),
    ("RetClean",    "csvs/retclean_%s_mapped.csv"),   # hospital == Jellyfish (D29); flights is independent
    ("GPT-4o-mini", "csvs/gpt4o_%s_mapped.csv"),
    # HoloClean runs on hospital, beers and flights (the corpora with denial
    # constraints); beers is degenerate and dropped from Table 1 (make_figures).
    ("HoloClean",   "csvs/holoclean_%s_mapped.csv"),
]

# ROW SHARDS. A proposer too expensive to run over a whole table can be run over a
# contiguous row range instead. That is NOT the same as a truncated log, and the
# distinction decides how the numbers may be read:
#
#   truncated log  -- the run stopped early (rate limits, a crash). Which cells are
#                     missing is determined by when the failure happened, so the
#                     surviving set is biased in a way we cannot characterise.
#   row shard      -- the range was chosen BEFORE the run and processed exhaustively.
#                     Exchangeability holds within the shard, so it is a smaller
#                     benchmark rather than a biased sample of a larger one.
#
# Coverage is therefore computed against the SHARD's error cells, not the whole
# table's, and the regime is tagged 'shard' rather than 'partial'. The same range must
# be passed to bench.run_study via --row-range or the un-sharded rows are scored as
# escalations. Verify against the run log: it prints the queue reduction.
SHARDS = {
    ("Jellyfish", "tax"): {"path": "csvs/jellyfish_tax_mapped.csv",
                           "row_range": (0, 25000)},
}
OUT = "experiments/proposer_accuracy.csv"


def _norm(v):
    return str(v).lower().strip() if v is not None else ""


def main():
    from bench.datasets import load

    cache = {}

    def truth(ds):
        if ds not in cache:
            art, defects, _ = load(ds)
            cache[ds] = ({k: _norm(v) for k, v in defects.gold.items()},
                         {f"{c.row_id}::{c.col}": _norm(c.value)
                          for c in art.iter_cells()})
        return cache[ds]

    os.makedirs("experiments", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["proposer", "dataset", "n_gold_errors", "n_proposed_on_errors",
                    "coverage_of_errors", "acc_on_errors", "n_cols", "n_perfect_cols",
                    "cells_in_perfect_cols", "frac_in_perfect_cols",
                    "clean_touched", "clean_broken", "regime"])
        for prop, pat in PROPOSERS:
            for ds in DATASETS:
                shard = SHARDS.get((prop, ds))
                path = shard["path"] if shard else pat % ds
                if not os.path.exists(path):
                    continue
                gold, cur = truth(ds)
                rr = shard["row_range"] if shard else None
                if rr:
                    lo, hi = rr

                    def _in(ref, lo=lo, hi=hi):
                        try:
                            return lo <= int(ref.split("::")[0]) < hi
                        except (TypeError, ValueError):
                            return False
                    # Score against the shard's own population, so 'coverage' answers
                    # "did the proposer answer everything it was asked?" rather than
                    # "how much of the table did we run?" -- which is a budget fact,
                    # not a property of the proposer.
                    gold = {k: v for k, v in gold.items() if _in(k)}
                per, ok = collections.Counter(), collections.Counter()
                n = clean_touched = clean_broken = 0
                cols = set()
                for r in csv.DictReader(open(path, newline="", encoding="utf-8",
                                             errors="replace")):
                    ref = f"{r['row_id']}::{r['column']}"
                    v = _norm(r.get("value"))
                    cols.add(r["column"])
                    if ref in gold:
                        n += 1
                        hit = v == gold[ref]
                        per[r["column"]] += 1
                        ok[r["column"]] += hit
                    elif ref in cur:
                        clean_touched += 1
                        clean_broken += v != cur[ref]
                if not n:
                    continue
                acc = sum(ok.values()) / n
                perfect = [c for c in per if ok[c] == per[c]]
                in_perfect = sum(per[c] for c in perfect)
                cov = n / max(1, len(gold))
                if acc >= 0.9999:
                    regime = "degenerate"        # proposer solves the benchmark
                elif acc < 0.05:
                    regime = "hopeless"          # nothing is certifiable
                else:
                    regime = "informative"
                if cov < 0.95:
                    regime += "/partial"
                if rr:
                    regime += f"/shard[{rr[0]},{rr[1]})"
                w.writerow([prop, ds, len(gold), n, f"{cov:.4f}", f"{acc:.4f}",
                            len(cols), len(perfect), in_perfect,
                            f"{in_perfect / n:.4f}", clean_touched, clean_broken,
                            regime])
                extra = ""
                if clean_touched:
                    # The damage a proposer does to cells that were ALREADY correct is
                    # invisible to repair-F1, which only scores error cells. On tax it
                    # is the dominant effect and the reason the ungoverned baseline is
                    # unusable, so surface it here rather than leaving it in the CSV.
                    per_fix = (clean_broken / (n * acc)) if n * acc >= 1 else float("inf")
                    extra = (f"  clean_broken={clean_broken:,}/{clean_touched:,}"
                             f" ({clean_broken/clean_touched*100:.1f}%)"
                             + (f", {per_fix:,.0f} broken per cell fixed"
                                if per_fix != float("inf") else ", fixed nothing"))
                print(f"{prop:<12}{ds:<10} acc={acc:.4f} cov={cov:.3f} "
                      f"perfect_cols={len(perfect)}/{len(cols)} "
                      f"frac_in_perfect={in_perfect / n:.4f}  [{regime}]{extra}")
    print(f"\nwrote {OUT}")
    print()
    audit_detector()
    print()
    audit_fp_handling()




def audit_detector(path="experiments/detector_quality.csv"):
    """Precision/recall of the real detector whose log we use as a non-oracle work
    queue. Reported because the non-oracle automation numbers are only interpretable
    relative to how good the detector was: false positives inject clean cells into the
    queue (which a detector-free proposer then rewrites), and missed errors shrink the
    per-stratum calibration mass the certifier needs."""
    import glob
    from bench.datasets import load
    rows = []
    for p in sorted(glob.glob("csvs/raha_*_detected.csv")):
        ds = os.path.basename(p).split("_")[1]
        try:
            art, defects, _ = load(ds)
        except Exception:
            continue
        gold = set(defects.gold)
        known = {f"{c.row_id}::{c.col}" for c in art.iter_cells()}
        det = {f"{r['row_id']}::{r['column']}"
               for r in csv.DictReader(open(p, newline=""))} & known
        tp = len(det & gold)
        prec = tp / max(1, len(det))
        rec = tp / max(1, len(gold))
        rows.append([ds, len(det), len(gold), tp, f"{prec:.4f}", f"{rec:.4f}",
                     f"{2*prec*rec/max(1e-9, prec+rec):.4f}"])
        print(f"  raha/{ds:<9} detected={len(det):>5} gold={len(gold):>5} "
              f"prec={prec:.3f} rec={rec:.3f}")
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "n_detected", "n_gold", "tp", "precision",
                        "recall", "f1"])
            w.writerows(rows)
        print(f"wrote {path}")

def audit_fp_handling(path="experiments/fp_handling.csv"):
    """What each proposer does with the detector's FALSE POSITIVES.

    This is the quantity that decides robustness to detection error, and it is not
    visible in any accuracy metric: a false positive is a cell that was already
    correct, so repair-F1 never scores it. A proposer asked "fix this cell" that has
    no way to answer "there is nothing to fix" will invent a change, and every such
    change converts a clean cell into an error inside a stratum that would otherwise
    have been certifiable.
    """
    from bench.datasets import load
    LOGS = [("Baran", "csvs/baran_raha_%s_mapped.csv"),
            ("Jellyfish", "csvs/jellyfish_%s_mapped.csv"),
            ("BClean", "csvs/bclean_keep_%s_mapped.csv")]
    rows = []
    for ds in DATASETS:
        det_p = f"csvs/raha_{ds}_detected.csv"
        if not os.path.exists(det_p):
            continue
        art, defects, _ = load(ds)
        gold = set(defects.gold)
        cur = {f"{c.row_id}::{c.col}": _norm(c.value) for c in art.iter_cells()}
        det = {f"{r['row_id']}::{r['column']}"
               for r in csv.DictReader(open(det_p, newline=""))} & set(cur)
        fps = det - gold                      # flagged but actually correct
        for prop, pat in LOGS:
            p2 = pat % ds
            if not os.path.exists(p2):
                continue
            touched = changed = 0
            for r in csv.DictReader(open(p2, newline="", encoding="utf-8",
                                         errors="replace")):
                ref = f"{r['row_id']}::{r['column']}"
                if ref not in fps:
                    continue
                v = _norm(r.get("value"))
                if not v:
                    continue                  # abstention: the safe answer
                touched += 1
                changed += v != cur[ref]
            if not touched:
                continue
            rows.append([prop, ds, len(det), len(fps), touched, changed,
                         f"{changed/touched:.4f}"])
            print(f"  {prop:<10}{ds:<10} FPs={len(fps):>5}  touched={touched:>5}  "
                  f"CHANGED={changed:>5} ({changed/touched:.2f})")
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["proposer", "dataset", "n_detected", "n_false_positives",
                        "fp_touched", "fp_changed", "fp_change_rate"])
            w.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

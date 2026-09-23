"""export_holoclean.py -- run HoloClean over a dataset and export a CARE-ready
offline-repair log.

Why HoloClean is worth adding as a fifth proposer: it is the probabilistic
data-repair system this literature is built on, and unlike every other proposer in
the study it emits a genuine posterior probability per repair rather than a
heuristic score. That makes it the sharpest available test of the paper's central
claim. The natural objection is "HoloClean already gives me a probability, so why do
I need a certifier?" -- and the answer is that a marginal posterior from
a factor graph is not a distribution-free bound on the error rate of the set you
choose to auto-apply. Running CARE on top of HoloClean's own probabilities either
demonstrates that gap empirically or fails to, and both outcomes are informative.
Report the calibration curve of `prob` against realised correctness either way.

Same zero-coupling pattern as the other integrations: HoloClean runs natively in
its own repo and environment, and this script writes a plain CSV that CARE reads
through the offline-log proposer (`holoclean:<log>` backend spec). CARE never
imports HoloClean's code, and HoloClean never learns CARE exists.

    row_id, column, value, confidence, source

Environment. HoloClean is old and pinned: Python 3.6/3.7, PostgreSQL >= 9.4, and a
torch build from that era. Do NOT try to run it in CARE's venv -- it will not
resolve. Create its own, follow the upstream README to start Postgres and create
the `holo` database and user, then run this script from the HoloClean repo root.

    git clone https://github.com/HoloClean/holoclean
    # ... upstream env setup, start postgres, create db ...
    python export_holoclean.py hospital \\
      --data testdata/hospital.csv --dcs testdata/hospital_constraints.txt

Then, back in CARE:

    python prepare_log.py  hospital csvs/holoclean_hospital_pred.csv
    python precheck_log.py hospital csvs/holoclean_hospital_mapped.csv
    python -m bench.run_study --dataset hospital \\
      --backends "holoclean:csvs/holoclean_hospital_mapped.csv" \\
      --experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 \\
      --scoring errors --out experiments/results_holoclean

Where the numbers come from, verified against this checkout rather than assumed:

* `inf_values_dom` holds (`_tid_`, `attribute`, `rv_value`) and carries **no
  probability**. The probability lives in `inf_values_idx` as `prob`, keyed by
  `_vid_`, alongside `inferred_val`. So the export joins `cell_domain` (which maps
  `_vid_` to `_tid_`, `attribute` and `init_value`) against `inf_values_idx`.
  Reading `inf_values_dom` instead -- the obvious guess -- silently yields a log
  with no confidence column, which reduces CARE to all-or-nothing per stratum and
  throws away the entire reason for running HoloClean.
* `_tid_` is 0-based row position. CARE's `row_id` is the 1-based `index` column of
  its own dirty CSV, so `row_id = _tid_ + 1`.
* Row alignment was checked, not hoped for: HoloClean's `testdata/hospital.csv` and
  CARE's `data/hospital/dirty.csv` are the same 1000 rows in the same order
  (6000 fields compared across six columns, zero mismatches). You can therefore run
  HoloClean on its own testdata copy unchanged. If you ever point it at a different
  file, re-check that before trusting a sweep.
* CARE writes the literal string `empty` where HoloClean writes its `NULL_REPR`
  (`_nan_`) for a missing value. Both are normalised here, otherwise every imputed
  null would be miscounted as a change to a non-empty value.

`--probe` prints the tables and columns actually present after inference. Use it if
you are on a different HoloClean revision than this one.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

SOURCE = "holoclean"
# Keeps get their OWN provenance tag, and this is load-bearing rather than cosmetic.
# A "change" row's confidence scores the repair HoloClean chose; a "keep" row's scores
# the absence of one. The two are not on a comparable scale -- measured on hospital,
# changes have median 0.530 and keeps 0.918 -- so pooling them into one stratum lets
# the ranking invert (deviation D23, the Simpson's paradox BClean hit first).
# CARE's guard is `strata=mondrian_src`, which bench.experiments.pareto auto-enables
# when a log carries MORE THAN ONE source. Emitting a single tag for both decision
# types therefore silently disables that guard, which is why this is not just a
# label. BClean's exporter has always tagged them separately; this matches it.
SOURCE_KEEP = "holoclean_keep"

# HoloClean's null marker; CARE's benchmark CSVs use the literal string "empty".
NULL_TOKENS = {"", "_nan_", "empty", "nan", "null", "none"}


def _norm(v):
    """Compare values the way both systems mean them, not the way they spell them."""
    s = str(v).strip().lower() if v is not None else ""
    return "" if s in NULL_TOKENS else s


# --------------------------------------------------------------------------
# HoloClean session
# --------------------------------------------------------------------------
def build_session(args):
    import holoclean
    from detect import NullDetector, ViolationDetector
    from repair.featurize import (ConstraintFeaturizer, FreqFeaturizer,
                                  InitAttrFeaturizer, LangModelFeaturizer,
                                  OccurAttrFeaturizer)

    hc = holoclean.HoloClean(
        db_name=args.db, domain_thresh_1=0, domain_thresh_2=0,
        weak_label_thresh=0.99, max_domain=10000, cor_strength=0.6,
        nb_cor_strength=0.8, epochs=args.epochs, weight_decay=0.01,
        learning_rate=0.001, threads=1, batch_size=1, verbose=args.verbose,
        timeout=3 * 60000, feature_norm=False, weight_norm=False, print_fw=False,
    ).session

    name = os.path.splitext(os.path.basename(args.data))[0]
    hc.load_data(name, args.data)
    hc.load_dcs(args.dcs)
    hc.ds.set_constraints(hc.get_dcs())

    hc.detect_errors([NullDetector(), ViolationDetector()])
    hc.setup_domain()
    featurizers = [InitAttrFeaturizer(), OccurAttrFeaturizer(),
                   FreqFeaturizer(), ConstraintFeaturizer()]
    if args.lang_model:
        featurizers.append(LangModelFeaturizer())
    hc.repair_errors(featurizers)
    return hc


# --------------------------------------------------------------------------
# Result extraction
# --------------------------------------------------------------------------
def _query(hc, sql):
    return hc.ds.engine.execute_query(sql)


def probe(hc):
    """Print what is actually in the database, so a wrong assumption is visible."""
    rows = _query(hc, """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema='public' ORDER BY table_name
    """)
    names = [r[0] for r in rows]
    print("\ntables present:")
    for n in names:
        cols = _query(hc, f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='{n}' ORDER BY ordinal_position
        """)
        print(f"  {n:<28} {[c[0] for c in cols]}")
    print("\nExpected on this revision: `inf_values_idx` with (_vid_, inferred_val, "
          "prob), and `cell_domain` with (_vid_, _tid_, attribute, init_value). "
          "Note that `inf_values_dom` has the value but NOT the probability.")


EXTRACT_SQL = """
    SELECT c._tid_, c.attribute, i.inferred_val, i.prob, c.init_value
    FROM cell_domain AS c, inf_values_idx AS i
    WHERE c._vid_ = i._vid_
"""


def extract(hc):
    """Return [(tid, attribute, inferred_value, prob, init_value)] per inferred cell.

    The join is the whole point: `inf_values_idx` has the probability but is keyed
    by `_vid_`, and only `cell_domain` knows which (tuple, attribute) a `_vid_` is.
    """
    rows = _query(hc, EXTRACT_SQL)
    if not rows:
        raise SystemExit(
            "no rows from cell_domain x inf_values_idx -- did repair_errors() run?\n"
            "Use --probe to see what is actually in the database."
        )
    return [(int(r[0]), r[1], r[2],
             float(r[3]) if r[3] is not None else None, r[4]) for r in rows]


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", help="CARE dataset name, used only to name the output")
    ap.add_argument("--data", default="testdata/hospital.csv", help="path to the dirty CSV")
    ap.add_argument("--dcs", default="testdata/hospital_constraints.txt",
                    help="path to the denial-constraints file")
    ap.add_argument("--db", default="holo")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lang-model", action="store_true",
                    help="add LangModelFeaturizer (slow; needs its pretrained weights)")
    ap.add_argument("--row-offset", type=int, default=1,
                    help="added to HoloClean's 0-based _tid_ to get CARE's 1-based "
                         "row_id. Changing this silently misaligns EVERY cell, so "
                         "verify against the dirty CSV before trusting a sweep.")
    ap.add_argument("--emit-keep", action="store_true",
                    help="also emit cells HoloClean examined but left unchanged. Writes "
                         "to holoclean_keep_<ds>_pred.csv so it cannot clobber the "
                         "changes-only log. Emits keeps as "
                         "explicit 'keep' rows. Without this the log has no row for a "
                         "detector's false positive, so it cannot be scored against a "
                         "detector queue (see the BClean exporter for the same issue).")
    ap.add_argument("--probe", action="store_true",
                    help="print the database schema after inference and exit")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hc = build_session(args)
    if args.probe:
        probe(hc)
        return 0

    preds = extract(hc)
    print(f"  [extract] {len(preds)} inferred cells")

    # Read the input purely to check alignment. A mismatch here means the log would
    # be written against the wrong rows, which no downstream check would catch.
    with open(args.data, newline="") as fh:
        dirty = list(csv.DictReader(fh))
    max_tid = max(p[0] for p in preds)
    if max_tid + args.row_offset > len(dirty):
        raise SystemExit(
            f"_tid_ {max_tid} + offset {args.row_offset} exceeds the {len(dirty)} rows "
            f"in {args.data}. The log would be misaligned; fix --row-offset or --data."
        )

    # The keep log gets its OWN filename. Writing both variants to one name silently
    # destroys whichever was produced first, and the two are not interchangeable: the
    # changes-only log is the repair proposal set, the keep log additionally records
    # every cell HoloClean examined and declined to alter. `prepare_log.py` splits the
    # tool name on the dataset, so "holoclean_keep" survives as a distinct backend tag.
    stem = "holoclean_keep" if args.emit_keep else "holoclean"
    out = args.out or f"{stem}_{args.dataset}_pred.csv"
    n_change = n_keep = 0
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["row_id", "column", "value", "confidence", "source"])
        for tid, attr, value, prob, init in preds:
            changed = _norm(value) != _norm(init)
            if not changed and not args.emit_keep:
                n_keep += 1
                continue
            # A keep is recorded as the CURRENT value, not as an abstention: an empty
            # value means "I have nothing to say about this cell" and CARE escalates
            # it, which is a different statement from "I examined this and it is fine".
            wr.writerow([tid + args.row_offset, attr, value,
                         "" if prob is None else f"{prob:.6f}",
                         SOURCE if changed else SOURCE_KEEP])
            n_change += changed
            n_keep += (not changed)

    print(f"\nwrote {out}")
    print(f"  changes: {n_change}   keeps: {n_keep}")
    if args.emit_keep and n_change and n_keep:
        print(f"  provenance: source={SOURCE} on changes, {SOURCE_KEEP} on keeps -- two "
              f"decision types whose confidence scales are NOT comparable.")
        print("  CARE will auto-enable strata=mondrian_src (column x source) because the "
              "log now carries >1 source. Do NOT read a pooled top-k curve from it.")
    probs = [p[3] for p in preds if p[3] is not None]
    if not probs:
        print("  WARNING: no probabilities extracted. Without a confidence column CARE "
              "can only take a stratum all-or-nothing, which wastes HoloClean's main "
              "advantage over the other proposers.")
    else:
        lo, hi = min(probs), max(probs)
        print(f"  confidence: n={len(probs)} min={lo:.4f} max={hi:.4f} "
              f"mean={sum(probs) / len(probs):.4f}")
        if hi - lo < 1e-6:
            print("  WARNING: the probability is constant, so it cannot rank repairs "
                  "and CARE will only be able to take a stratum whole or not at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

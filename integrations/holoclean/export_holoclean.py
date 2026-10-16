"""Run HoloClean and export its repairs as a CARE log
(row_id, column, value, confidence, source).

Confidence is HoloClean's marginal probability `prob` from inf_values_idx, joined to
cell_domain on _vid_ (inf_values_dom has no probability). row_id = _tid_ + 1, and
HoloClean's null `_nan_` and CARE's `empty` are both treated as null. HoloClean needs
its own environment (Python 3.6/3.7, PostgreSQL); run from the HoloClean repository root:

    python export_holoclean.py hospital \\
      --data testdata/hospital.csv --dcs testdata/hospital_constraints.txt
    python export_holoclean.py beers --data beers_holoclean.csv \\
      --dcs constraints/beers_constraints.txt --emit-keep

--probe prints the database tables after inference.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

SOURCE = "holoclean"
# Keep rows get their own source tag, so CARE stratifies by source (strata=mondrian_src);
# change and keep confidences are not on a comparable scale.
SOURCE_KEEP = "holoclean_keep"

# HoloClean's null marker; CARE's benchmark CSVs use the literal string "empty".
NULL_TOKENS = {"", "_nan_", "empty", "nan", "null", "none"}


def _norm(v):
    """Normalise a value for comparison, mapping null tokens to ''."""
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
    """Print the database tables and their columns."""
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

    inf_values_idx holds prob keyed by _vid_; cell_domain maps _vid_ to (tid, attribute).
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

    # check row alignment against the input
    with open(args.data, newline="") as fh:
        dirty = list(csv.DictReader(fh))
    max_tid = max(p[0] for p in preds)
    if max_tid + args.row_offset > len(dirty):
        raise SystemExit(
            f"_tid_ {max_tid} + offset {args.row_offset} exceeds the {len(dirty)} rows "
            f"in {args.data}. The log would be misaligned; fix --row-offset or --data."
        )

    # the keep log gets its own file name and backend tag
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
            # a keep records the current value; an empty value would mean "no proposal"
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

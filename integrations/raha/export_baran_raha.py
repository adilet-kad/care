"""export_baran_raha.py -- Baran corrections driven by REAL (Raha) detection.

WHY THIS EXISTS
---------------
Every Baran number in the paper uses Baran's published protocol, in which the error
locations are handed to it (`correction.py`'s own `__main__` does
`data.detected_cells = dict(data.get_actual_errors_dictionary())` -- the oracle diff).
That is the right protocol for reproducing Baran, but it is not a deployment setting,
and it is the strongest assumption behind Baran's numbers.

We already measured the cost of dropping it for the LLM proposer, whose log covers every
cell and could therefore simply be re-scored against a different work queue. Baran's log
cannot: it records only the cells Baran CHANGED, so it holds no opinion about a
detector's false positives. Scoring that silence would measure the protocol rather than
the proposer. The only honest way to close the gap is to RE-RUN Baran with the detector's
cells as its input, which is what this script does.

This is not a hack on Baran. Raha (detector) and Baran (corrector) are the same project,
and Raha -> Baran is its intended pipeline; `detected_cells` is the documented seam.

WHAT IS AND IS NOT NON-ORACLE HERE
----------------------------------
NOT oracle any more:  the work queue. Cells come from raha.detection's real ensemble.
STILL oracle:         Baran's 20 labelled tuples (`LABELING_BUDGET = 20`), which use
                      ground truth. That is Baran's published active-learning protocol
                      and we keep it so the comparison to our oracle-detection runs
                      changes exactly ONE variable. Both facts belong together:
                      "non-oracle detection" here means detection only.

VALUES IN `detected_cells` ARE UNUSED -- verified, not assumed
-------------------------------------------------------------
`correction.py` touches `d.detected_cells` at lines 377, 397, 399, 462-463, 608, 663:
every one is a membership test or a len(), and 463 overwrites the value with
IGNORE_SIGN. No code path reads a value. So seeding it with a placeholder is exactly
equivalent to seeding it with the oracle's clean values -- and, unlike the latter, it
cannot leak ground truth into the corrector. We use a placeholder.

DETECTION IS REUSED, NOT RE-RUN (by default)
--------------------------------------------
We load ../CARE/csvs/raha_<ds>_detected.csv, the same file the LLM non-oracle sweep
used. That makes the two comparable: same detector, same cells, only the proposer
differs. `--rerun-detection` re-runs raha.detection instead, which will NOT reproduce
the earlier file exactly (Raha's active-learning sampling is stochastic).

USAGE (from the raha repo root, in raha's own venv)
---------------------------------------------------
    python export_baran_raha.py hospital
    python export_baran_raha.py hospital beers flights rayyan

Output: ../CARE/csvs/baran_raha_<ds>_pred.csv, or baran_<ds>_pred.csv with --oracle
        (row_id,column,value,confidence,source)

Then, in CARE:
    python prepare_log.py  <ds> csvs/baran_raha_<ds>_pred.csv
    python precheck_log.py <ds> csvs/baran_raha_<ds>_pred.csv --scope touched
    python -m bench.run_study --dataset <ds> \
        --backends "baran_raha:csvs/baran_raha_<ds>_mapped.csv" \
        --experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 \
        --scoring detected --fast-verify on \
        --detection log:csvs/raha_<ds>_detected.csv \
        --out experiments/results_raha_baran

Note the sweep ALSO passes --detection: the log and the work queue must be the same
cells, or CARE will queue cells Baran was never asked about.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import raha.correction
import raha.dataset
# Imported at module level ON PURPOSE. A conditional `import raha.detection` inside
# export_one() would bind the name `raha` as a FUNCTION-LOCAL, shadowing the two
# module-level imports above for the whole function -- so `raha.dataset.Dataset(...)`
# would raise UnboundLocalError even on the path that never runs the import.
import raha.detection

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, os.pardir, "CARE", "csvs")
DEFAULT_DET_DIR = os.path.join(HERE, os.pardir, "CARE", "csvs")

# Must stay in sync with export_detection.py::COL_MAPS and CARE's
# prepare_log.py::COL_MAPS. Same raha.dataset column ordering feeds all three.
COL_MAPS = {
    "hospital": {
        "phone": "PhoneNumber", "state": "State", "emergency_service": "EmergencyService",
        "zip": "ZipCode", "name": "HospitalName", "measure_name": "MeasureName",
        "city": "City", "state_average": "Stateavg", "type": "HospitalType",
        "owner": "HospitalOwner", "sample": "Sample", "address_1": "Address1",
        "provider_number": "ProviderNumber", "county": "CountyName",
        "condition": "Condition", "measure_code": "MeasureCode", "score": "Score",
    },
}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _has_index_col(clean_header, clean_rows):
    """Mirrors bench.datasets.load AND export_detection.py EXACTLY. Getting this wrong
    shifts every column by one and silently produces a plausible, wrong log."""
    first_name = clean_header[0].strip().lower()
    looks_named_index = first_name in {"index", "tid", "id", "row_id", "key", ""}
    first_vals = [r[0] for r in clean_rows if r]
    looks_int_index = bool(first_vals) and all(
        v.strip().lstrip("-").isdigit() for v in first_vals
    )
    return looks_named_index or looks_int_index


def export_one(dataset, args):
    ds_dir = os.path.join(HERE, "datasets", dataset)
    dirty_path = os.path.join(ds_dir, "dirty.csv")
    clean_path = os.path.join(ds_dir, "clean.csv")
    if not (os.path.exists(dirty_path) and os.path.exists(clean_path)):
        raise SystemExit(f"missing {dirty_path} / {clean_path}")

    dirty_header, dirty_rows = _read_csv(dirty_path)
    clean_header, clean_rows = _read_csv(clean_path)
    has_index = _has_index_col(clean_header, clean_rows)

    def row_id(i):
        return dirty_rows[i][0] if has_index else str(i)

    dd = {"name": dataset, "path": dirty_path, "clean_path": clean_path}
    data = raha.dataset.Dataset(dd)
    raw_cols = data.dataframe.columns.tolist()
    col_map = COL_MAPS.get(dataset, {})
    # CARE column name -> raha dataframe column index
    care_to_j = {col_map.get(name, name): j for j, name in enumerate(raw_cols)}
    rowid_to_i = {row_id(i): i for i in range(len(dirty_rows))}

    print(f"\n========== BARAN + RAHA DETECTION: {dataset} ==========")

    # ---- work queue -------------------------------------------------------
    if args.oracle:
        # Baran's own published protocol: the work queue IS the gold diff. This is
        # the documented regeneration path for the oracle logs in csvs/.
        #
        # The gold VALUES are blanked even though Baran's published call passes
        # them through. correction.py never reads a detected cell's value (see the
        # module docstring: verified at lines 377, 397, 399, 462-463, 608, 663), so
        # this is behaviourally identical -- but "identical because we checked" is
        # a worse guarantee than "cannot leak because the value is not there", and
        # seeding a corrector with the answers is the one mistake that would
        # invalidate every number in the paper.
        detected = {k: "" for k in data.get_actual_errors_dictionary()}
        print(f"ORACLE queue: {len(detected)} cells (Baran's published protocol; "
              f"gold values blanked, keys only)")
    elif args.rerun_detection:
        app = raha.detection.Detection()
        app.ERROR_DETECTION_ALGORITHMS = args.algorithms
        app.VERBOSE = False
        detected = dict(app.run(dd))
        print(f"re-ran raha.detection: {len(detected)} cells "
              f"(will NOT match the cached file; Raha's sampling is stochastic)")
    else:
        det_path = args.detection or os.path.join(
            DEFAULT_DET_DIR, f"raha_{dataset}_detected.csv")
        if not os.path.exists(det_path):
            raise SystemExit(
                f"no detection log at {det_path}. Generate it first with\n"
                f"    python export_detection.py {dataset}\n"
                f"or pass --rerun-detection to run detection inline.")
        detected, dropped_row, dropped_col = {}, 0, 0
        with open(det_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                i = rowid_to_i.get(r["row_id"])
                j = care_to_j.get(r["column"])
                if i is None:
                    dropped_row += 1
                elif j is None:
                    dropped_col += 1
                else:
                    # Value is never read by correction.py (see module docstring);
                    # a placeholder avoids seeding the corrector with gold values.
                    detected[(i, j)] = ""
        print(f"loaded {det_path}")
        print(f"  work queue: {len(detected)} cells")
        if dropped_row or dropped_col:
            # Loud, because a silent inversion failure looks exactly like a detector
            # that found fewer errors, and would quietly change every number below.
            print(f"  !! DROPPED {dropped_row} rows + {dropped_col} columns that could "
                  f"not be mapped back to raha's frame. This is a BUG, not a detector "
                  f"property -- check COL_MAPS and the index-column logic before "
                  f"using this log.", file=sys.stderr)
            if not args.allow_drops:
                raise SystemExit("refusing to continue; pass --allow-drops to override.")

    if not detected:
        raise SystemExit("empty work queue")

    # How different is this from the oracle queue? Report it here so the log carries
    # its own provenance rather than relying on the reader to recompute it.
    actual = data.get_actual_errors_dictionary()
    det_set, act_set = set(detected), set(actual)
    tp = len(det_set & act_set)
    prec = tp / max(1, len(det_set))
    rec = tp / max(1, len(act_set))
    print(f"  vs oracle: precision={prec:.3f} recall={rec:.3f} "
          f"(TP={tp}, FP={len(det_set - act_set)}, FN={len(act_set - det_set)})")
    print(f"  -> Baran will now be asked about {len(det_set - act_set)} cells that are "
          f"NOT errors; what it does to them is the point of this run.")

    # ---- run Baran --------------------------------------------------------
    data.detected_cells = dict(detected)
    app = raha.correction.Correction()
    app.VERBOSE = args.verbose
    if args.labeling_budget is not None:
        app.LABELING_BUDGET = args.labeling_budget
    print(f"  LABELING_BUDGET={app.LABELING_BUDGET} (ground-truth labelled tuples -- "
          f"still oracle, by Baran's published protocol)")
    corrections = app.run(data)          # {(i, j): corrected_value}
    print(f"  Baran returned {len(corrections)} corrections")

    # ---- export -----------------------------------------------------------
    stem = "baran" if args.oracle else "baran_raha"
    out_rows = []
    for (i, j), value in corrections.items():
        if has_index and j == 0:
            continue                      # the key column is not a CARE "column"
        v = "" if value is None else str(value).strip()
        if not v:
            continue                      # empty -> abstention, CARE escalates
        out_rows.append((row_id(i), col_map.get(raw_cols[j], raw_cols[j]), v,
                         "1.0000", stem))

    os.makedirs(args.out, exist_ok=True)
    # Distinct filenames: the oracle-queue log and the detector-queue log are
    # different measurements of the same system and must never overwrite each other.
    out_path = os.path.join(args.out, f"{stem}_{dataset}_pred.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "column", "value", "confidence", "source"])
        w.writerows(out_rows)

    # accuracy split by whether the queued cell was a real error, which is the
    # comparison this whole run exists to make
    on_err = on_err_ok = on_fp = on_fp_broken = 0
    for (i, j), value in corrections.items():
        if has_index and j == 0:
            continue
        v = "" if value is None else str(value).strip()
        if not v:
            continue
        if (i, j) in actual:
            on_err += 1
            on_err_ok += str(actual[(i, j)]).strip().lower() == v.lower()
        elif (i, j) in det_set:
            on_fp += 1
            cur = str(data.dataframe.iloc[i, j]).strip().lower()
            on_fp_broken += cur != v.lower()
    print(f"WROTE: {out_path}  ({len(out_rows)} repairs)")
    print(f"  on true errors      : {on_err_ok}/{on_err} correct "
          f"({on_err_ok/max(1,on_err):.3f})")
    print(f"  on detector FALSE POSITIVES: {on_fp} touched, {on_fp_broken} CHANGED "
          f"(each change corrupts an already-correct cell)")
    print("=" * (34 + len(dataset)))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--oracle", action="store_true",
                    help="use the gold diff as the work queue (Baran's published "
                         "protocol) and write baran_<ds>_pred.csv, rather than a "
                         "detector's cells")
    ap.add_argument("--detection", default=None,
                    help="path to a row_id,column detection CSV (default: "
                         "../CARE/csvs/raha_<ds>_detected.csv)")
    ap.add_argument("--rerun-detection", action="store_true",
                    help="run raha.detection inline instead of loading the cached log. "
                         "Will not reproduce it exactly (stochastic sampling), so the "
                         "comparison with the LLM non-oracle sweep stops being "
                         "controlled -- prefer the cached file.")
    ap.add_argument("--algorithms", nargs="+", default=["OD", "PVD", "RVD", "KBVD"],
                    help="only with --rerun-detection")
    ap.add_argument("--labeling-budget", type=int, default=None,
                    help="override Baran's LABELING_BUDGET (default: its own 20). "
                         "Changing this changes the protocol.")
    ap.add_argument("--allow-drops", action="store_true",
                    help="continue even if detection refs fail to map back (unsafe)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    for ds in args.datasets:
        export_one(ds, args)


if __name__ == "__main__":
    main()

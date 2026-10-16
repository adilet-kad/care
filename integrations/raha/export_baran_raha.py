"""Run Baran with Raha's detected cells as its work queue and export a CARE log
(row_id, column, value, confidence, source).

By default the queue is ../CARE/csvs/raha_<ds>_detected.csv (from export_detection.py);
--rerun-detection runs raha.detection instead, and --oracle uses the gold diff (Baran's
published protocol). Baran's 20 labelled tuples still use ground truth. correction.py
never reads values in detected_cells, so they are set to "".

Run from the raha repository root, in raha's environment:
    python export_baran_raha.py hospital beers flights rayyan
    python export_baran_raha.py hospital --oracle

Output: ../CARE/csvs/baran_raha_<ds>_pred.csv, or baran_<ds>_pred.csv with --oracle.
Sweep it in CARE with --detection log:csvs/raha_<ds>_detected.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import raha.correction
import raha.dataset
# module-level: an import inside export_one() would make `raha` a local name
import raha.detection

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, os.pardir, "CARE", "csvs")
DEFAULT_DET_DIR = os.path.join(HERE, os.pardir, "CARE", "csvs")

# raha column name -> CARE column name; same as export_detection.py and prepare_log.py
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
    """Same index-column detection as bench.datasets.load and export_detection.py."""
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

    print(f"\n== Baran on Raha-detected cells: {dataset}")

    # ---- work queue -------------------------------------------------------
    if args.oracle:
        # Baran's published protocol: the queue is the gold diff, values blanked
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
                    # value is never read by correction.py
                    detected[(i, j)] = ""
        print(f"loaded {det_path}")
        print(f"  work queue: {len(detected)} cells")
        if dropped_row or dropped_col:
            # unmapped cells indicate a COL_MAPS or index bug
            print(f"  warning: dropped {dropped_row} rows + {dropped_col} columns that could "
                  f"not be mapped back to raha's frame. This is a BUG, not a detector "
                  f"property -- check COL_MAPS and the index-column logic before "
                  f"using this log.", file=sys.stderr)
            if not args.allow_drops:
                raise SystemExit("refusing to continue; pass --allow-drops to override.")

    if not detected:
        raise SystemExit("empty work queue")

    # detection quality against the oracle queue
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
            continue                      # empty -> no proposal
        out_rows.append((row_id(i), col_map.get(raw_cols[j], raw_cols[j]), v,
                         "1.0000", stem))

    os.makedirs(args.out, exist_ok=True)
    # oracle-queue and detector-queue logs use distinct file names
    out_path = os.path.join(args.out, f"{stem}_{dataset}_pred.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "column", "value", "confidence", "source"])
        w.writerows(out_rows)

    # accuracy on true errors vs on detector false positives
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
    print(f"  on detector false positives: {on_fp} touched, {on_fp_broken} changed "
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

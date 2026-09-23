"""export_detection.py -- run REAL Raha error detection and export a CARE-ready
detected-cells log, for the T2 non-oracle-detection experiment.

Deliberate architectural choice (same as the Baran correction integration):
Raha runs natively in ITS OWN repo/environment here, and we export a plain CSV
log that CARE reads through `bench.study.detect(..., mode="log:<path>")`. CARE
never imports raha's code -- this keeps the "governs any proposer/detector with
zero coupling" claim literal in the codebase.

What this produces is genuinely different from the oracle detection already used
everywhere else in CARE: raha.detection.Detection().run() runs Raha's real
ensemble of error-detection strategies (outlier detection, pattern violation,
rule violation, knowledge-base violation) plus its active-learning classifier --
NOT the ground-truth diff. (Careful: raha/correction.py's own __main__ example
uses `data.detected_cells = dict(data.get_actual_errors_dictionary())`, which
IS the oracle diff -- that is fine for generating Baran's corrections (CARE's
headline results are explicitly oracle-detection), but it is NOT what we want
here. This script calls raha.detection.Detection().run() directly, which is the
real, non-oracle detector.)

Usage (run from the raha repo root, in raha's own venv):
    python export_detection.py hospital
    python export_detection.py beers rayyan flights          # several datasets
    python export_detection.py tax --algorithms PVD RVD      # tax is 200K rows;
                                                               # OD/KBVD are slow
                                                               # at that scale --
                                                               # start without them

Output:
    ../CARE/csvs/raha_<dataset>_detected.csv   (row_id,column -- no value; this
                                                 is a WORK QUEUE, not a proposer)
and a precision/recall/F1 printout against the dataset's own clean.csv, so you
know the detection quality before it ever reaches CARE.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import raha.dataset
import raha.detection

# ---------------------------------------------------------------------------
# Column remap: raha's raw dataframe column name -> CARE's CSV header name.
# MUST stay in sync with CARE's prepare_log.py::COL_MAPS -- same underlying
# raha.dataset.Dataset column ordering feeds both the correction and detection
# exports, so a column that needed remapping for Baran needs it here too.
# ---------------------------------------------------------------------------
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

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, os.pardir, "CARE", "csvs")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _has_index_col(clean_header, clean_rows):
    """Mirror bench.datasets.load's index-column detection EXACTLY.

    Getting this wrong shifts every column by one."""
    first_name = clean_header[0].strip().lower()
    looks_named_index = first_name in {"index", "tid", "id", "row_id", "key", ""}
    first_vals = [r[0] for r in clean_rows if r]
    looks_int_index = bool(first_vals) and all(
        v.strip().lstrip("-").isdigit() for v in first_vals
    )
    return looks_named_index or looks_int_index


def export_one(dataset, *, out_dir, algorithms):
    ds_dir = os.path.join(HERE, "datasets", dataset)
    dirty_path = os.path.join(ds_dir, "dirty.csv")
    clean_path = os.path.join(ds_dir, "clean.csv")
    if not (os.path.exists(dirty_path) and os.path.exists(clean_path)):
        raise SystemExit(f"missing {dirty_path} / {clean_path} -- is '{dataset}' bundled in datasets/?")

    dirty_header, dirty_rows = _read_csv(dirty_path)
    clean_header, clean_rows = _read_csv(clean_path)
    has_index = _has_index_col(clean_header, clean_rows)

    def row_id(i):
        return dirty_rows[i][0] if has_index else str(i)

    col_map = COL_MAPS.get(dataset, {})

    def care_col(raw_name):
        return col_map.get(raw_name, raw_name)

    print(f"\n========== RAHA DETECTION: {dataset} ==========")
    print(f"rows: {len(dirty_rows)}  has_index_col: {has_index}  algorithms: {algorithms}")

    dd = {"name": dataset, "path": dirty_path, "clean_path": clean_path}
    app = raha.detection.Detection()
    app.ERROR_DETECTION_ALGORITHMS = algorithms
    app.VERBOSE = False
    detected_cells = app.run(dd)   # {(i, j): value} -- REAL, non-oracle detection

    d = raha.dataset.Dataset(dd)
    raw_cols = d.dataframe.columns.tolist()

    out_rows = []
    skipped_index_col = 0
    for (i, j) in detected_cells:
        if has_index and j == 0:
            skipped_index_col += 1   # the key column itself isn't a CARE "column"
            continue
        col = care_col(raw_cols[j])
        out_rows.append((row_id(i), col))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"raha_{dataset}_detected.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "column"])
        w.writerows(out_rows)

    # Precision/recall/F1 against ground truth, so detection quality is known
    # BEFORE this log reaches CARE. A suspiciously high number and a suspiciously
    # low one are both worth catching here rather than after a full sweep.
    actual_errors = d.get_actual_errors_dictionary()   # {(i,j): clean_value}, oracle diff
    detected_set = set(detected_cells)
    actual_set = set(actual_errors)
    tp = len(detected_set & actual_set)
    precision = tp / len(detected_set) if detected_set else 0.0
    recall = tp / len(actual_set) if actual_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"detected cells: {len(detected_cells)}  (skipped {skipped_index_col} index-column hits)")
    print(f"true errors:    {len(actual_set)}")
    print(f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    if tp == 0:
        print("!! ZERO true positives -- something is wrong (algorithms misconfigured, "
              "or this dataset's errors aren't the type these strategies catch). "
              "Do NOT feed this log to CARE as-is.")
    print(f"WROTE: {out_path}")
    print(f"Use in CARE:  --detection \"log:csvs/raha_{dataset}_detected.csv\"")
    print("=" * (26 + len(dataset)) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", help="one or more of: beers hospital rayyan flights tax movies_1")
    ap.add_argument("--out", default=DEFAULT_OUT_DIR, help="output directory (default: ../CARE/csvs)")
    ap.add_argument("--algorithms", nargs="+", default=["OD", "PVD", "RVD", "KBVD"],
                    help="raha ERROR_DETECTION_ALGORITHMS subset (drop KBVD/OD if slow or erroring)")
    args = ap.parse_args()

    for ds in args.datasets:
        if ds == "tax" and "OD" in args.algorithms:
            print("!! tax is 200K rows -- OD (dboost) over the full grid can be very slow. "
                  "If this hangs, Ctrl-C and retry with --algorithms PVD RVD", file=sys.stderr)
        export_one(ds, out_dir=args.out, algorithms=args.algorithms)


if __name__ == "__main__":
    main()

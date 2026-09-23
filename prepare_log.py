"""prepare_log.py -- validate/prepare ANY external cleaner's repair log for CARE.

Generic replacement for the tool-specific prep scripts. Works for Baran, RetClean,
Jellyfish, or any system emitting the offline-log contract:

    row_id, column, value, [confidence], [source]

What it does
------------
1. Remaps the cleaner's column names to CARE's header (via ``COL_MAPS`` below).
2. DROPS rows whose mapped column doesn't exist in CARE's schema (e.g. hospital's
   address_2/address_3), so junk never enters the work queue -- and reports them.
3. Carries `confidence` and `source` through (the graded signal CARE ranks on and the
   provenance the trust gate reads). Losing these silently is a real failure mode.
4. Reports alignment against CARE's true error cells + the confidence distribution,
   and gives a proceed / do-not-proceed verdict.
5. Writes `csvs/<tool>_<dataset>_mapped.csv` -- the tool name is inferred from the
   input filename (or set with --tool), so different cleaners never overwrite each
   other's outputs.

Usage (from the CARE repo root):
    python prepare_log.py <dataset> <path/to/log.csv> [--tool NAME]

Examples:
    python prepare_log.py hospital csvs/jellyfish_hospital_pred.csv
    python prepare_log.py flights  csvs/retclean_flights_pred.csv
    python prepare_log.py beers    csvs/baran_beers_pred.csv --tool baran
"""

import argparse
import collections
import csv
import os
import re

# ---------------------------------------------------------------------------
# Per-dataset column remaps: the cleaner's (Raha-side) column name -> CARE's
# header name. Only needed when a cleaner's dataframe columns differ from CARE's
# CSV header; datasets whose names already match (beers, flights, ...) need NO
# entry. precheck_log.py and audit_calibration.py import this same table, and the
# exporters under integrations/ keep their copies in sync with it.
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

OUT_DIR = "csvs"
_KNOWN_TOOLS = ("jellyfish", "retclean", "baran", "holoclean", "gpt4o", "bclean")


def _infer_tool(path, dataset=None):
    """Tool name for the output filename: everything before `_<dataset>_`.

    Prefix matching against _KNOWN_TOOLS alone is WRONG for variant logs. On
    `baran_raha_hospital_pred.csv` it returns "baran", so the output would be
    `csvs/baran_hospital_mapped.csv` -- silently overwriting the oracle-detection
    Baran log that the headline results depend on. Splitting on the dataset name
    instead yields "baran_raha" and keeps the two side by side.
    """
    base = os.path.basename(path).lower()
    if dataset:
        m = re.match(rf"(.+?)_{re.escape(dataset.lower())}[_.]", base)
        if m:
            return m.group(1)
    for t in _KNOWN_TOOLS:
        if base.startswith(t):
            return t
    m = re.match(r"([a-z0-9]+)[_\-]", base)
    return m.group(1) if m else "external"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("log")
    ap.add_argument("--tool", default=None,
                    help="name used in the output filename (default: inferred)")
    ap.add_argument("--keep-unknown-columns", action="store_true",
                    help="keep rows whose column isn't in CARE's schema (default: drop)")
    ap.add_argument("--out", default=OUT_DIR,
                    help=f"output directory (default: {OUT_DIR}/). Use a separate "
                         "directory when preparing repeated draws of the same "
                         "(tool, dataset) pair, which otherwise overwrite each other")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        raise SystemExit(f"log not found: {args.log}")
    tool = args.tool or _infer_tool(args.log, args.dataset)

    try:
        from bench.datasets import load
        _art, defects, dcols = load(args.dataset)
    except Exception as e:
        raise SystemExit(f"could not load CARE dataset {args.dataset!r}: {e}\n"
                         f"Run from the CARE repo root so 'bench' is importable.")
    care_cols, care_refs = set(dcols), set(defects.gold)

    with open(args.log, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    for req in ("row_id", "column", "value"):
        if req not in fields:
            raise SystemExit(f"log missing required column {req!r}; has {fields}")
    has_conf = "confidence" in fields       # graded ranking signal -- must survive
    has_src = "source" in fields            # provenance -> trust gate

    user_map = COL_MAPS.get(args.dataset, {})
    mapped = lambda c: user_map.get(c, c)

    kept, dropped = [], collections.Counter()
    for r in rows:
        col = mapped(r["column"])
        if col not in care_cols and not args.keep_unknown_columns:
            dropped[col] += 1
            continue
        kept.append((r, col))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{tool}_{args.dataset}_mapped.csv")
    # csvs/ holds logs that cost GPU-hours or API spend to produce. Overwriting one
    # with a different run's output is unrecoverable and silent, so say what is being
    # replaced and with how many rows -- a large row-count change is the tell.
    if os.path.exists(out_path) and os.path.abspath(out_path) != os.path.abspath(args.log):
        try:
            with open(out_path, newline="", encoding="utf-8", errors="replace") as f:
                old_n = sum(1 for _ in f) - 1
        except OSError:
            old_n = -1
        print(f"  note: overwriting existing {out_path} ({old_n} rows) with "
              f"{len(kept)} rows from {args.log}")
    out_fields = (["row_id", "column", "value"]
                  + (["confidence"] if has_conf else [])
                  + (["source"] if has_src else []))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out_fields)
        for r, col in kept:
            row = [r["row_id"], col, r["value"]]
            if has_conf:
                row.append(r.get("confidence", ""))
            if has_src:
                row.append(r.get("source", ""))
            w.writerow(row)

    nonblank = [(r, c) for r, c in kept if (r.get("value") or "").strip()]
    refs = {f"{r['row_id']}::{c}" for r, c in nonblank}
    touched_cols = {c for _r, c in kept}
    scoped = {ref for ref in care_refs if ref.split("::", 1)[-1] in touched_cols}
    overlap = refs & care_refs
    cov = len(overlap) / max(1, len(scoped))

    print(f"\n========== PREP [{tool}]: {args.dataset} ==========")
    print(f"log rows: {len(rows)}   kept: {len(kept)}   "
          f"confidence: {'YES' if has_conf else 'no'}   source: {'YES' if has_src else 'no'}")
    if dropped:
        print(f"dropped {sum(dropped.values())} rows in columns not in CARE's schema: "
              f"{dict(dropped)}")
    print(f"columns touched: {sorted(touched_cols)}")
    print(f"alignment: {len(overlap)}/{len(scoped)} error cells in the touched columns "
          f"= {cov:.1%}  (dataset-wide error count {len(care_refs)})")

    if has_conf:
        vals = []
        for r, _c in nonblank:
            try:
                vals.append(float(r["confidence"]))
            except (TypeError, ValueError):
                pass
        if vals:
            vals.sort()
            uniq = len(set(vals))
            print(f"confidence: min/med/max = {vals[0]:.3f}/{vals[len(vals)//2]:.3f}/"
                  f"{vals[-1]:.3f}   distinct values = {uniq}")
            if uniq <= 2:
                print("  !! near-constant confidence -> the conformal controller cannot "
                      "RANK within a stratum; it degenerates to all-or-nothing.")
    else:
        print("!! no confidence column -> constant s_hat -> all-or-nothing per stratum.")
    if has_src:
        c = collections.Counter((r.get("source") or "").strip() for r, _ in nonblank)
        print("provenance split: " + ", ".join(f"{k or '(blank)'}={v}" for k, v in c.most_common()))

    if cov == 0:
        print("!! ZERO overlap -> row_id base or column format wrong. Do NOT proceed.")
    elif cov < 0.3:
        print("!! LOW overlap (<30%) -> the cleaner abstained on most error cells, or a "
              "keying mismatch. Investigate before the sweep.")
    else:
        print("OK -> healthy overlap. Proceed.")
    print(f"WROTE: {out_path}")
    print(f'Use in run_study:  --backends "{tool}:{out_path}"')
    print("=" * (26 + len(args.dataset)) + "\n")


if __name__ == "__main__":
    main()

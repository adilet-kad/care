"""export_bclean.py -- run BClean (ICDE'24) and export a CARE-ready repair log.

Same zero-coupling pattern as the Baran / Raha / RetClean / Jellyfish integrations:
BClean runs natively here, and we export a plain CSV that CARE reads through its
offline-log proposer (`bclean:<log>` backend). CARE never imports BClean's code.

WHY BCLEAN IS IN THE PANEL
--------------------------
It is a genuinely THIRD paradigm alongside the classical-ML (Baran) and LLM
(Jellyfish/GPT/RetClean) proposers: Bayesian-network inference with user constraints.
It is peer-reviewed at a main A* database venue (ICDE 2024) and was already evaluated
on Hospital and Flights, so we are using it on its home turf -- unlike RetClean, whose
native task (imputation from a data lake) mismatched these corruption benchmarks.

FAIRNESS DECISIONS (state these in the paper)
---------------------------------------------
1. **We use the AUTHORS' user constraints verbatim** (`UC_json/<dataset>.json`).
   Authoring our own UCs would let us tune BClean's performance up or down, which
   would make the comparison meaningless. Consequence: BClean only repairs the columns
   its UC file declares -- hospital 15 of 17, beers 6 of 10, flights 6 of 6. Cells in
   undeclared columns are simply absent from the log (CARE escalates them), which is
   the honest representation of what BClean does.
2. **Every proposer sees identical input.** We feed BClean OUR dirty.csv (renamed to
   the column names its UC file expects) rather than its bundled copy, so row order
   and cell values are guaranteed identical to what Baran/Jellyfish/GPT saw. Verified:
   the bundled copies match ours row-for-row anyway, but constructing the input removes
   the assumption.
3. **No oracle leakage.** BClean's API takes `clean_df`, which we verified is used ONLY
   to compute `actual_error` for its own precision/recall reporting -- `Inference.Repair`
   passes it through untouched and `repair_line` never receives it. The repair decision
   sees dirty data only. (Contrast with the Baran logs, which DO use oracle detection.)

CONFIDENCE (--with-confidence, optional)
----------------------------------------
BClean internally computes, per candidate value, a score
    P(value | parents) * P(children | value) * penalty
sorts candidates by it, and takes the top one -- then DISCARDS the scores, keeping only
the winning value. That score is a natural Bayesian confidence, and CARE needs a graded
signal to rank repairs within a stratum (without one it degenerates to all-or-nothing;
this is why the no-confidence Baran logs certify whole columns or nothing).

With `--with-confidence` we capture it by patching BClean's `infer.py` **source at
import time, in memory**: the file on disk is never modified. The patch only ADDS a
recording statement; it does not change which candidate is selected, so BClean's repairs
are bit-identical with and without it. Confidence is
reported as the normalised margin  top1 / (top1 + top2), i.e. how decisively the
Bayesian network preferred its choice.

If the patch fails to apply, we fall back to constant confidence and say so loudly
rather than silently emitting a degraded signal.

USAGE (from the BClean repo root)
---------------------------------
    pip install -r requirements.txt
    python export_bclean.py hospital --with-confidence
    python export_bclean.py hospital beers flights --with-confidence --workers 32

Output: ../CARE/csvs/bclean_<dataset>_pred.csv  (row_id,column,value,confidence,source)
Then in CARE:
    python prepare_log.py  <ds> csvs/bclean_<ds>_pred.csv
    python precheck_log.py <ds> csvs/bclean_<ds>_pred.csv --scope touched
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, os.pardir, "CARE", "csvs")
_DATA_CANDIDATES = [
    os.path.join(HERE, os.pardir, "RetClean", "datasets"),
    os.path.join(HERE, os.pardir, "raha", "datasets"),
]

# Raha/CARE lowercase name -> the column name BClean's UC_json expects.
# Kept in sync with CARE/prepare_log.py::COL_MAPS; only hospital differs in naming.
COL_MAPS = {
    "hospital": {
        "provider_number": "ProviderNumber", "name": "HospitalName",
        "address_1": "Address1", "address_2": "Address2", "address_3": "Address3",
        "city": "City", "state": "State", "zip": "ZipCode", "county": "CountyName",
        "phone": "PhoneNumber", "type": "HospitalType", "owner": "HospitalOwner",
        "emergency_service": "EmergencyService", "condition": "Condition",
        "measure_code": "MeasureCode", "measure_name": "MeasureName",
        "score": "Score", "sample": "Sample", "state_average": "Stateavg",
    },
}


# ------------------------------------------------------------------ patching

_PATCH_ANCHOR = """			total.sort(key = lambda x: x[1], reverse = True)"""
_PATCH_REPLACEMENT = """			total.sort(key = lambda x: x[1], reverse = True)
			# [CARE instrumentation] record the scores BClean already computed.
			# Adds a side-channel only; the selection loop below is untouched.
			try:
				_t = [float(s) for _v, s in total[:2]]
				_care_scores[(line, val)] = (_t + [0.0])[:2]
			except Exception:
				pass"""


def _load_bclean(with_confidence):
    """Import BClean, optionally with in-memory instrumentation of infer.py.

    Returns (BayesianClean, Dataset, UC, care_scores_dict_or_None).
    The on-disk repo is never modified.
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    care_scores = {}
    if with_confidence:
        infer_path = os.path.join(HERE, "src", "infer.py")
        src = open(infer_path, encoding="utf-8").read()
        if _PATCH_ANCHOR not in src:
            print("  !! could not find the instrumentation anchor in src/infer.py; "
                  "BClean's source may have changed. Falling back to CONSTANT "
                  "confidence (CARE will only be able to certify whole strata).",
                  file=sys.stderr)
            care_scores = None
        else:
            patched = src.replace(_PATCH_ANCHOR, _PATCH_REPLACEMENT, 1)
            mod = types.ModuleType("src.infer")
            mod.__file__ = infer_path
            mod.__dict__["_care_scores"] = care_scores
            # make the package-relative imports inside infer.py resolve
            import importlib
            pkg = importlib.import_module("src") if os.path.exists(
                os.path.join(HERE, "src", "__init__.py")) else None
            if pkg is not None:
                sys.modules.setdefault("src", pkg)
            exec(compile(patched, infer_path, "exec"), mod.__dict__)
            sys.modules["src.infer"] = mod

    from BClean import BayesianClean          # noqa: E402
    from dataset import Dataset               # noqa: E402
    from src.UC import UC                     # noqa: E402
    return BayesianClean, Dataset, UC, care_scores


# ------------------------------------------------------------------ data prep

def _find_dataset_dir(dataset, override=None):
    cands = ([os.path.join(override, dataset)] if override else []) + \
            [os.path.join(d, dataset) for d in _DATA_CANDIDATES]
    for c in cands:
        if os.path.exists(os.path.join(c, "dirty.csv")):
            return c
    raise SystemExit(f"could not find datasets/{dataset}/dirty.csv in:\n  " +
                     "\n  ".join(cands) + "\nPass --data-dir.")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _has_index_col(header, rows):
    """Mirror bench.datasets.load's index detection EXACTLY.

    A wrong row-key convention shifts every column by one and silently misaligns
    every gold comparison, so this must not drift from CARE's copy."""
    first = header[0].strip().lower()
    if first in {"index", "tid", "id", "row_id", "key", ""}:
        return True
    vals = [r[0] for r in rows if r]
    return bool(vals) and all(v.strip().lstrip("-").isdigit() for v in vals)


def _norm(name):
    """Normalise a column name for matching across dirty/clean files.

    NECESSARY because the benchmark's dirty and clean CSVs do NOT use identical
    headers (verified):
        beers    dirty: beer_name, brewery_name   clean: beer-name, brewery-name
        hospital dirty: provider_number, name...  clean: ProviderNumber, HospitalName...
        flights  identical
    Selecting columns independently per file therefore yields DIFFERENT column sets
    (beers clean lost 'brewery-name' -> BClean crashed with KeyError). We instead fix
    the selection from the dirty header and locate each column in clean by this
    normalised key.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _build_bclean_input(dataset, ds_dir, uc_cols, tmp_dir, limit=None):
    """Write dirty/clean CSVs restricted+renamed to what BClean's UC file expects.

    The DIRTY header is authoritative: it fixes which columns are kept and what they
    are called. The clean file is matched to that selection by normalised name, so a
    header mismatch is impossible by construction (and is reported loudly if a column
    genuinely cannot be located rather than silently dropped).

    Returns (dirty_path, clean_path, row_ids).
    """
    os.makedirs(tmp_dir, exist_ok=True)
    cmap = COL_MAPS.get(dataset, {})

    d_header, d_rows = _read_csv(os.path.join(ds_dir, "dirty.csv"))
    c_header, c_rows = _read_csv(os.path.join(ds_dir, "clean.csv"))
    if limit:
        d_rows, c_rows = d_rows[:limit], c_rows[:limit]
    if len(d_rows) != len(c_rows):
        raise SystemExit(f"{dataset}: dirty has {len(d_rows)} rows, clean has "
                         f"{len(c_rows)} -- refusing to proceed (row alignment).")

    has_idx = _has_index_col(d_header, d_rows)
    col_start = 1 if has_idx else 0
    row_ids = [(r[0] if has_idx else str(i)) for i, r in enumerate(d_rows)]

    # 1) selection fixed by the DIRTY header
    keep = []
    for j in range(col_start, len(d_header)):
        out_name = cmap.get(d_header[j], d_header[j])
        if out_name in uc_cols:
            keep.append((j, out_name))
    if not keep:
        raise SystemExit(f"none of the UC columns {sorted(uc_cols)} matched {dataset}'s "
                         f"dirty header {d_header}. Check COL_MAPS.")

    # 2) locate the SAME columns in clean, by normalised name
    c_by_norm = {}
    for j, h in enumerate(c_header):
        c_by_norm.setdefault(_norm(cmap.get(h, h)), j)
        c_by_norm.setdefault(_norm(h), j)
    c_keep, missing = [], []
    for j, out_name in keep:
        cj = c_by_norm.get(_norm(out_name), c_by_norm.get(_norm(d_header[j])))
        (c_keep.append(cj) if cj is not None else missing.append(out_name))
    if missing:
        raise SystemExit(f"{dataset}: columns {missing} exist in dirty.csv but could not "
                         f"be located in clean.csv (header {c_header}).")

    for which, header_rows, cols in (("dirty", d_rows, [j for j, _ in keep]),
                                     ("clean", c_rows, c_keep)):
        p = os.path.join(tmp_dir, f"{dataset}_{which}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([name for _j, name in keep])       # identical header for both
            for r in header_rows:
                w.writerow([(r[j] if j < len(r) else "") for j in cols])

    return (os.path.join(tmp_dir, f"{dataset}_dirty.csv"),
            os.path.join(tmp_dir, f"{dataset}_clean.csv"), row_ids)


# ------------------------------------------------------------------ main

def export_one(dataset, args, BayesianClean, Dataset, UC, care_scores):
    uc_json = os.path.join(HERE, "UC_json", f"{dataset}.json")
    if not os.path.exists(uc_json):
        raise SystemExit(f"no author-provided UC file for {dataset}: {uc_json}\n"
                         f"We deliberately do NOT author our own UCs (that would let us "
                         f"tune BClean). Datasets without a shipped UC are out of scope.")
    uc_cols = set(json.load(open(uc_json)).keys())
    ds_dir = _find_dataset_dir(dataset, args.data_dir)
    tmp = os.path.join(HERE, "_care_input")
    dirty_p, clean_p, row_ids = _build_bclean_input(dataset, ds_dir, uc_cols, tmp,
                                                     limit=args.limit)

    print(f"\n========== BCLEAN: {dataset} ==========")
    print(f"rows: {len(row_ids)}   UC columns ({len(uc_cols)}): {sorted(uc_cols)}")
    print(f"confidence: {'bayesian-margin' if care_scores is not None else 'CONSTANT (patch unavailable)'}")

    loader = Dataset()
    dirty_df = loader.get_data(path=dirty_p)
    clean_df = loader.get_data(path=clean_p)

    uc = UC(dirty_df)
    uc.build_from_json(uc_json)
    attr = uc.get_uc()
    dirty_df = loader.get_real_data(dirty_df, attr_type=attr)
    clean_df = loader.get_real_data(clean_df, attr_type=attr)

    if care_scores is not None:
        care_scores.clear()

    model = BayesianClean(
        dirty_df=dirty_df, clean_df=clean_df,
        model_path=None, model_save_path=None,
        attr_type=attr, fix_edge=[],
        model_choice=args.model_choice,
        infer_strategy=args.infer_strategy,
        tuple_prun=args.tuple_prun,
        maxiter=args.maxiter,
        num_worker=args.workers,
        chunksize=args.chunksize,
    )

    repair_err = model.repair_list[3]          # {(line, col): new_value}

    def _conf(line, col):
        conf = args.default_confidence
        if care_scores is not None:
            sc = care_scores.get((line, col))
            if sc:
                top1, top2 = (list(sc) + [0.0, 0.0])[:2]
                tot = top1 + top2
                conf = (top1 / tot) if tot > 0 else args.default_confidence
        return conf

    log_rows = []
    for (line, col), value in repair_err.items():
        v = "" if value is None else str(value).strip()
        if not v or v == "A Null Cell":
            continue                            # BClean's null sentinel -> abstain
        log_rows.append((row_ids[line], col, v, f"{_conf(line, col):.4f}", "bclean"))

    if args.emit_keep:
        # FULL-COVERAGE MODE, for the non-oracle-detection comparison.
        #
        # BClean has no detection input: it infers over the whole table and
        # repair_list[3] holds only the cells it decided to CHANGE. That makes its log
        # impossible to re-slice onto a different work queue -- for a detector's false
        # positive, BClean has no row, and absence is ambiguous between "I say keep
        # this value" and "I never considered it".
        #
        # Within the UC-declared columns BClean DID consider every cell, so there the
        # ambiguity is resolvable: absence means "keep". Emitting that explicitly turns
        # the log into a full-coverage proposal set over the declared columns, directly
        # comparable to the LLM proposers', and lets the same log be scored against
        # oracle or detector queues.
        #
        # Restricted to declared columns on purpose: outside them BClean genuinely has
        # no opinion, and inventing one would manufacture correctness on cells it never
        # examined.
        changed = set(repair_err)
        added = 0
        for line in range(len(row_ids)):
            for col in dirty_df.columns:
                if col not in uc_cols or (line, col) in changed:
                    continue
                cur = dirty_df.iloc[line][col]
                cur = "" if cur is None else str(cur).strip()
                if not cur or cur == "A Null Cell":
                    continue
                # SEMANTICS: care_scores holds the margin of the best ALTERNATIVE
                # BClean scored for this cell. For a cell it chose NOT to change, a
                # high margin means it came close to overwriting -- i.e. LOW confidence
                # that the current value should stand. So a keep's confidence is the
                # complement. Measured on flights: as-is the keep rows rank INVERSELY
                # (top-10% 0.813 vs 0.873 pool); complemented they rank correctly
                # (0.936 vs 0.873).
                log_rows.append((row_ids[line], col, cur,
                                 f"{1.0 - _conf(line, col):.4f}", "bclean_keep"))
                added += 1
        print(f"  --emit-keep: added {added} 'keep current value' proposals over the "
              f"{len(uc_cols)} UC-declared columns (source=bclean_keep)")
        print("  !! The log now mixes TWO decision types whose confidence scales are "
              "NOT comparable: 'change' scores the chosen repair, 'keep' scores the "
              "absence of one. Pooled in a single stratum they produce a Simpson's "
              "paradox -- each source ranks correctly alone, the pool ranks inversely."
              "\n     Stratify by source as well as column when sweeping this log, or "
              "take strata all-or-nothing. Do NOT read a pooled top-k curve from it.",
              file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    sfx = f"_first{args.limit}" if args.limit else ""
    # --emit-keep produces a DIFFERENT log (full coverage over UC columns, ~5x the
    # rows), so it must not land on the oracle run's filename. Writing both to
    # bclean_<ds>_pred.csv would silently replace the log every published BClean
    # number derives from.
    stem = "bclean_keep" if args.emit_keep else "bclean"
    out_path = os.path.join(args.out, f"{stem}_{dataset}_pred{sfx}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "column", "value", "confidence", "source"])
        w.writerows(log_rows)

    print(f"  BClean proposed {len(log_rows)} repairs "
          f"(it reports {len(model.actual_error)} actual errors in the UC columns)")
    if log_rows:
        cs = sorted(float(r[3]) for r in log_rows)
        uniq = len(set(cs))
        print(f"  confidence min/med/max = {cs[0]:.3f}/{cs[len(cs)//2]:.3f}/{cs[-1]:.3f}"
              f"   distinct={uniq}")
        if uniq <= 2:
            print("  !! near-constant confidence -> CARE cannot rank within a stratum.")
    print(f"WROTE: {out_path}")
    print(f"Next (in CARE):  python prepare_log.py {dataset} csvs/bclean_{dataset}_pred.csv")
    print("=" * (22 + len(dataset)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--with-confidence", action="store_true",
                    help="capture BClean's internal Bayesian candidate scores as a "
                         "graded confidence (in-memory patch; disk untouched)")
    ap.add_argument("--model-choice", default="bdeu")
    ap.add_argument("--infer-strategy", default="PIPD")
    ap.add_argument("--tuple-prun", type=float, default=1.0)
    ap.add_argument("--maxiter", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunksize", type=int, default=10)
    ap.add_argument("--default-confidence", type=float, default=0.5)
    ap.add_argument("--emit-keep", action="store_true",
                    help="also emit a 'keep the current value' proposal for every cell "
                         "in a UC-declared column that BClean did NOT change "
                         "(source=bclean_keep). BClean takes no detection input, so by "
                         "default its log contains only changes and cannot be scored "
                         "against a detector's work queue -- absence is ambiguous "
                         "between 'keep this' and 'never considered'. Within the "
                         "declared columns BClean did consider every cell, so making "
                         "the 'keep' explicit is a faithful reading, and it makes the "
                         "log full-coverage like the LLM proposers'. Required for the "
                         "non-oracle-detection comparison; it also enlarges the log "
                         "substantially, so keep the default off for the oracle runs "
                         "that all published numbers use.")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N rows. ESSENTIAL for tax: BClean does "
                         "per-row inference over each column's value DOMAIN, and tax "
                         "has 200K rows with a ~30K-distinct 'zip' column, so a full "
                         "run can be extremely slow or exhaust memory. Start small.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    BayesianClean, Dataset, UC, care_scores = _load_bclean(args.with_confidence)
    if not args.with_confidence:
        care_scores = None
    for ds in args.datasets:
        export_one(ds, args, BayesianClean, Dataset, UC, care_scores)


if __name__ == "__main__":
    main()

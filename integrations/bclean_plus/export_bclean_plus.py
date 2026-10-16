"""Run BClean+ on CARE's tables and export its repairs as a CARE log
(row_id, column, value, confidence, source).

BClean+ is BClean with user constraints generated automatically (flan-t5-xxl
column typing plus a pattern library). The input is CARE's dirty.csv, renamed to
the UC column names. --uc best uses the authors' UC_json/<dataset>_schema_best.json
(hospital, beers, flights); --uc generated uses UC_json/<dataset>_schema.json from
generate_UC.py (run_generate_uc.sh). fix_edge is always empty. clean_df is used only
for BClean+'s own precision/recall report.

With --with-confidence, src/infer.py is patched in memory to record the top two
candidate scores in each worker and merge them in the parent; confidence is the
margin top1 / (top1 + top2). Repairs are unchanged by the patch.

Run from the BClean+ repository root (run_bclean_plus.sh wraps these):
    python export_bclean_plus.py hospital beers flights --with-confidence --data-dir ../CARE/data
    python export_bclean_plus.py rayyan tax --uc generated --with-confidence --data-dir ../CARE/data

Output: ../CARE/csvs/bclean_plus_<dataset>_pred.csv
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
    os.path.join(HERE, os.pardir, "CARE", "data"),
    os.path.join(HERE, os.pardir, "raha", "datasets"),
]

# Raha/CARE lowercase name -> the column name BClean+'s UC_json expects.
# Same as prepare_log.py::COL_MAPS; only hospital differs.
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

_PATCHES = [
    # 1. record the two best candidate scores BClean+ already computed
    ("""\t\t\ttotal.sort(key = lambda x: x[1], reverse = True)""",
     """\t\t\ttotal.sort(key = lambda x: x[1], reverse = True)
\t\t\t# [CARE instrumentation] record the scores BClean+ already computed.
\t\t\t# Adds a side-channel only; the selection loop below is untouched.
\t\t\ttry:
\t\t\t\t_t = [float(s) for _v, s in total[:2]]
\t\t\t\t_care_local[(line, val)] = (_t + [0.0])[:2]
\t\t\texcept Exception:
\t\t\t\tpass"""),
    # 2. a per-call buffer, created where repair_line creates its own repair dict
    ("""\t\trepair_err = {}\n\t\tdata_line, line, model_all""",
     """\t\trepair_err = {}\n\t\t_care_local = {}\n\t\tdata_line, line, model_all"""),
    # 3. hand the buffer back with the worker's result
    ("""\t\treturn {"id": line, "data": data_line, "repairdict": repair_err}""",
     """\t\treturn {"id": line, "data": data_line, "repairdict": repair_err, "carescores": _care_local}"""),
    # 4. merge it in the parent, next to the repairs
    ("""\t\t\t\tself.repair_err = {**self.repair_err, **r["repairdict"]}""",
     """\t\t\t\tself.repair_err = {**self.repair_err, **r["repairdict"]}
\t\t\t\t_care_scores.update(r.get("carescores", {}))"""),
]


def _load_bclean(with_confidence):
    """Import BClean+, optionally with in-memory instrumentation of infer.py.

    Returns (BayesianClean, Dataset, UC, care_scores_dict_or_None).
    The on-disk repo is never modified.
    """
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    care_scores = {}
    if with_confidence:
        infer_path = os.path.join(HERE, "src", "infer.py")
        src = open(infer_path, encoding="utf-8").read()
        missing = [a[:50] for a, _b in _PATCHES if src.count(a) != 1]
        if missing:
            print(f"  warning: instrumentation anchors not found exactly once in src/infer.py: "
                  f"{missing}. BClean+'s source may have changed. Falling back to "
                  f"CONSTANT confidence (CARE will only be able to certify whole strata).",
                  file=sys.stderr)
            care_scores = None
        else:
            patched = src
            for a, b in _PATCHES:
                patched = patched.replace(a, b, 1)
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
    """Same index-column detection as bench.datasets.load; keep in sync."""
    first = header[0].strip().lower()
    if first in {"index", "tid", "id", "row_id", "key", ""}:
        return True
    vals = [r[0] for r in rows if r]
    return bool(vals) and all(v.strip().lstrip("-").isdigit() for v in vals)


def _norm(name):
    """Normalise a column name; dirty and clean headers differ (e.g. beer_name vs beer-name)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _build_bclean_input(dataset, ds_dir, uc_cols, tmp_dir, limit=None):
    """Write dirty/clean CSVs restricted to and renamed for BClean+'s UC columns.

    Columns are selected from the dirty header and matched in clean by normalised name.
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

    # columns selected from the dirty header
    keep = []
    for j in range(col_start, len(d_header)):
        out_name = cmap.get(d_header[j], d_header[j])
        if out_name in uc_cols:
            keep.append((j, out_name))
    if not keep:
        raise SystemExit(f"none of the UC columns {sorted(uc_cols)} matched {dataset}'s "
                         f"dirty header {d_header}. Check COL_MAPS.")

    # same columns in clean, by normalised name
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

# Authors' settings from example/<dataset>.py; rayyan and tax use hospital's.
EXAMPLE_SETTINGS = {
    "hospital": dict(chunksize=10, model_update=False),
    "beers":    dict(chunksize=50, model_update=True),
    "flights":  dict(chunksize=20, model_update=False),
}

def export_one(dataset, args, BayesianClean, Dataset, UC, care_scores):
    name = {"best": f"{dataset}_schema_best.json",
            "generated": f"{dataset}_schema.json"}.get(args.uc, args.uc)
    uc_json = name if os.path.isabs(name) else os.path.join(HERE, "UC_json", name)
    if not os.path.exists(uc_json):
        raise SystemExit(f"no BClean+ UC file for {dataset}: {uc_json}\n"
                         f"Use --uc generated after running `bash run_bclean_plus.sh uc`. "
                         f"We never author UCs ourselves (that would tune BClean+).")
    uc_cols = set(json.load(open(uc_json)).keys())
    ds_dir = _find_dataset_dir(dataset, args.data_dir)
    tmp = os.path.join(HERE, "_care_input")
    dirty_p, clean_p, row_ids = _build_bclean_input(dataset, ds_dir, uc_cols, tmp,
                                                     limit=args.limit)

    print(f"\n========== BCLEAN+: {dataset}  (UC: {os.path.basename(uc_json)}) ==========")
    print(f"rows: {len(row_ids)}   UC columns ({len(uc_cols)}): {sorted(uc_cols)}")
    print(f"confidence: {'bayesian-margin' if care_scores is not None else 'CONSTANT (patch unavailable)'}")

    loader = Dataset()
    dirty_df = loader.get_data(path=dirty_p)
    clean_df = loader.get_data(path=clean_p)

    settings = EXAMPLE_SETTINGS.get(dataset, EXAMPLE_SETTINGS["hospital"])
    print(f"settings: model_choice={args.model_choice} infer={args.infer_strategy} "
          f"chunksize={args.chunksize or settings['chunksize']} "
          f"model_update={settings['model_update']} fix_edge=[]")
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
        chunksize=(args.chunksize or settings["chunksize"]),
        model_update=settings["model_update"],
    )
    repair_err = model.clean()["repair_err"]   # {(line, col): new_value}

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
            continue                            # BClean+'s null sentinel -> skip
        log_rows.append((row_ids[line], col, v, f"{_conf(line, col):.4f}", "bclean_plus"))

    if args.emit_keep:
        # BClean+ logs only changed cells. Within UC-declared columns it considered
        # every cell, so absence means "keep"; emit those rows explicitly so the log
        # covers every declared cell and can be sliced onto any detection queue.
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
                # the margin scores the best alternative, so a keep uses its complement
                log_rows.append((row_ids[line], col, cur,
                                 f"{1.0 - _conf(line, col):.4f}", "bclean_plus_keep"))
                added += 1
        print(f"  --emit-keep: added {added} 'keep current value' proposals over the "
              f"{len(uc_cols)} UC-declared columns (source=bclean_plus_keep)")
        print("  warning: the log now mixes two decision types whose confidence scales are "
              "NOT comparable: 'change' scores the chosen repair, 'keep' scores the "
              "absence of one. Pooled in a single stratum they produce a Simpson's "
              "paradox -- each source ranks correctly alone, the pool ranks inversely."
              "\n     Stratify by source as well as column when sweeping this log, or "
              "take strata all-or-nothing. Do NOT read a pooled top-k curve from it.",
              file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    sfx = f"_first{args.limit}" if args.limit else ""
    # --emit-keep writes to a separate file name
    stem = "bclean_plus_keep" if args.emit_keep else "bclean_plus"
    out_path = os.path.join(args.out, f"{stem}_{dataset}_pred{sfx}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "column", "value", "confidence", "source"])
        w.writerows(log_rows)

    print(f"  BClean+ proposed {len(log_rows)} repairs "
          f"(it reports {len(model.actual_error)} actual errors in the UC columns)")
    if log_rows:
        cs = sorted(float(r[3]) for r in log_rows)
        uniq = len(set(cs))
        print(f"  confidence min/med/max = {cs[0]:.3f}/{cs[len(cs)//2]:.3f}/{cs[-1]:.3f}"
              f"   distinct={uniq}")
        if uniq <= 2:
            print("  warning: near-constant confidence -> CARE cannot rank within a stratum. "
                  "If --with-confidence was set, the score side channel did not arrive.")
    print(f"WROTE: {out_path}")
    print(f"Next (in CARE):  python prepare_log.py {dataset} csvs/{stem}_{dataset}_pred.csv --tool {stem}")
    print("=" * (22 + len(dataset)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--with-confidence", action="store_true",
                    help="capture BClean+'s internal Bayesian candidate scores as a "
                         "graded confidence (in-memory patch; disk untouched)")
    ap.add_argument("--uc", default="best",
                    help="best = the authors' released UC_json/<ds>_schema_best.json; "
                         "generated = UC_json/<ds>_schema.json from generate_UC.py; or a path")
    ap.add_argument("--model-choice", default="bdeu",
                    help="structure search; example/*.py uses bdeu for every dataset")
    ap.add_argument("--infer-strategy", default="PI",
                    help="example/*.py uses PI (partition inference) for every dataset")
    ap.add_argument("--tuple-prun", type=float, default=1.0)
    ap.add_argument("--maxiter", type=int, default=1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunksize", type=int, default=None,
                    help="default: the authors' per-dataset value from example/*.py")
    ap.add_argument("--default-confidence", type=float, default=0.5)
    ap.add_argument("--emit-keep", action="store_true",
                    help="also emit a 'keep the current value' proposal for every cell "
                         "in a UC-declared column that BClean+ did NOT change "
                         "(source=bclean_plus_keep). BClean+ takes no detection input, so by "
                         "default its log contains only changes and cannot be scored "
                         "against a detector's work queue -- absence is ambiguous "
                         "between 'keep this' and 'never considered'. Within the "
                         "declared columns BClean+ did consider every cell, so making "
                         "the 'keep' explicit is a faithful reading, and it makes the "
                         "log full-coverage like the LLM proposers'. Required for the "
                         "non-oracle-detection comparison; it also enlarges the log "
                         "substantially, so keep the default off for the oracle runs "
                         "that all published numbers use.")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N rows. ESSENTIAL for tax: BClean+ does "
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

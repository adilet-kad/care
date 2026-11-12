"""Error Drop Rate (Ni et al., PVLDB 2024) of apply-all and CARE on the same cells.

EDR = (errors_before - errors_after) / errors_before, over the test split of each seed.
Apply-all writes every proposal; CARE writes only auto-applied cells, the rest keep
their dirty value. Among CARE's changes, fixes - breaks >= (1 - 2*alpha) * n_changes
whenever the risk bound holds; violations are counted per seed. `edr_care_table`
divides by all gold errors in the table.

    python audit_error_drop.py                # oracle detection
    python audit_error_drop.py --raha         # add Raha-detection rows
    python audit_error_drop.py --skip-tax     # skip the two slow tax configurations
    -> experiments/error_drop.csv  (+ paper/tables/extra_error_drop.tex with --tex)
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

# (tag, dataset, log, scoring, detection, row_range), oracle detection; degenerate
# pairs (Baran/flights, BClean/rayyan+tax, HoloClean/beers) are excluded.
TABLE1 = [
    ("baran", "hospital", "csvs/baran_hospital_mapped.csv", "errors", "oracle", None),
    ("baran", "beers", "csvs/baran_beers_mapped.csv", "errors", "oracle", None),
    ("baran", "rayyan", "csvs/baran_rayyan_mapped.csv", "errors", "oracle", None),
    ("baran", "tax", "csvs/baran_tax_mapped.csv", "errors", "oracle", None),
    ("bclean", "hospital", "csvs/bclean_hospital_mapped.csv", "errors", "oracle", None),
    ("bclean", "beers", "csvs/bclean_beers_mapped.csv", "errors", "oracle", None),
    ("bclean", "flights", "csvs/bclean_flights_mapped.csv", "errors", "oracle", None),
    ("holoclean", "hospital", "csvs/holoclean_hospital_mapped.csv", "errors", "oracle", None),
    ("holoclean", "flights", "csvs/holoclean_flights_mapped.csv", "errors", "oracle", None),
    ("bclean_plus", "hospital", "csvs/bclean_plus_hospital_mapped.csv", "errors", "oracle", None),
    ("bclean_plus", "beers", "csvs/bclean_plus_beers_mapped.csv", "errors", "oracle", None),
    ("bclean_plus", "flights", "csvs/bclean_plus_flights_mapped.csv", "errors", "oracle", None),
    ("gidcl", "hospital", "csvs/gidcl_hospital_mapped.csv", "errors", "oracle", None),
    ("gidcl", "beers", "csvs/gidcl_beers_mapped.csv", "errors", "oracle", None),
    ("gidcl", "flights", "csvs/gidcl_flights_mapped.csv", "errors", "oracle", None),
    ("gidcl", "rayyan", "csvs/gidcl_rayyan_mapped.csv", "errors", "oracle", None),
    ("gidcl", "tax", "csvs/gidcl_tax_mapped.csv", "errors", "oracle", None),
    ("zeroec", "hospital", "csvs/zeroec_hospital_mapped.csv", "errors", "oracle", None),
    ("zeroec", "beers", "csvs/zeroec_beers_mapped.csv", "errors", "oracle", None),
    ("zeroec", "flights", "csvs/zeroec_flights_mapped.csv", "errors", "oracle", None),
    ("zeroec", "rayyan", "csvs/zeroec_rayyan_mapped.csv", "errors", "oracle", None),
    ("zeroec", "tax", "csvs/zeroec_tax_mapped.csv", "errors", "oracle", None),
    # GIDCL with its own detector: writes to clean cells are scored too
    ("gidcl_owndet", "hospital", "csvs/gidcl_hospital_mapped.csv", "detected", "log:csvs/gidcl_hospital_detected.csv", None),
    ("gidcl_owndet", "beers", "csvs/gidcl_beers_mapped.csv", "detected", "log:csvs/gidcl_beers_detected.csv", None),
    ("gidcl_owndet", "flights", "csvs/gidcl_flights_mapped.csv", "detected", "log:csvs/gidcl_flights_detected.csv", None),
    ("gidcl_owndet", "rayyan", "csvs/gidcl_rayyan_mapped.csv", "detected", "log:csvs/gidcl_rayyan_detected.csv", None),
    ("gidcl_owndet", "tax", "csvs/gidcl_tax_mapped.csv", "detected", "log:csvs/gidcl_tax_detected.csv", None),
]

# Raha detection (false positives included); keep logs cover cells left unchanged.
TABLE3 = [
    ("baran_raha", "hospital", "csvs/baran_raha_hospital_mapped.csv", "detected", "log:csvs/raha_hospital_detected.csv", None),
    ("baran_raha", "beers", "csvs/baran_raha_beers_mapped.csv", "detected", "log:csvs/raha_beers_detected.csv", None),
    ("baran_raha", "flights", "csvs/baran_raha_flights_mapped.csv", "detected", "log:csvs/raha_flights_detected.csv", None),
    ("baran_raha", "rayyan", "csvs/baran_raha_rayyan_mapped.csv", "detected", "log:csvs/raha_rayyan_detected.csv", None),
    ("baran", "tax", "csvs/baran_tax_mapped.csv", "detected", "log:csvs/raha_tax_detected.csv", None),
    ("bclean_keep", "hospital", "csvs/bclean_keep_hospital_mapped.csv", "detected", "log:csvs/raha_hospital_detected.csv", None),
    ("bclean_keep", "beers", "csvs/bclean_keep_beers_mapped.csv", "detected", "log:csvs/raha_beers_detected.csv", None),
    ("bclean_keep", "flights", "csvs/bclean_keep_flights_mapped.csv", "detected", "log:csvs/raha_flights_detected.csv", None),
    ("holoclean_keep", "hospital", "csvs/holoclean_keep_hospital_mapped.csv", "detected", "log:csvs/raha_hospital_detected.csv", None),
    ("holoclean_keep", "flights", "csvs/holoclean_keep_flights_mapped.csv", "detected", "log:csvs/raha_flights_detected.csv", None),
]

PRETTY = {"baran": "Baran", "baran_raha": "Baran", "bclean": "BClean", "bclean_keep": "BClean",
          "bclean_plus": "BClean+", "gidcl": "GIDCL", "gidcl_owndet": "GIDCL (own detector)", "zeroec": "ZeroEC",
          "holoclean": "HoloClean", "holoclean_keep": "HoloClean"}


def _norm(v):
    return str(v).lower().strip() if v is not None else None


def _strata(art, defects, repairs):
    """Mondrian strata as in bench.experiments.pareto, crossed with source if >1 source."""
    from care.conformal import by_product, by_source
    from bench.experiments import _mondrian_strata_fn
    sf = _mondrian_strata_fn(art, defects)
    srcs = {(getattr(r, "evidence", None) or ["?"])[0] for r in repairs.values()}
    if len(srcs) > 1:
        sf = by_product(sf, by_source({}))
        return sf, "mondrian_src"
    return sf, "mondrian"


def _count_errors(refs, values, truth):
    return sum(1 for r in refs if _norm(values[r]) != _norm(truth[r]))


def run_config(tag, ds, log, scoring, detection, row_range, *, alphas, delta, seeds):
    from bench.study import evaluate_split, load_and_propose
    from bench.experiments import _detected_truth

    art, defects, repairs, _ = load_and_propose(
        ds, f"{tag}:{log}", detection=detection, fast_verify=True, row_range=row_range)
    truth = _detected_truth(art, defects) if scoring == "detected" else dict(defects.gold)
    truth = {k: _norm(v) for k, v in truth.items()}
    dirty = {f"{c.row_id}::{c.col}": _norm(c.value) for c in art.iter_cells()}
    n_gold_table = len(defects.gold)
    sf, sname = _strata(art, defects, repairs)

    out = []
    for a in alphas:
        per = []
        for s in range(seeds):
            scores, dec = evaluate_split(repairs, truth, alpha=a, delta=delta, seed=s,
                                         strata_fn=sf, return_decisions=True)
            test = dec["test_refs"]
            auto = dec["auto"]                     # ref -> value CARE wrote
            before = _count_errors(test, dirty, truth)

            # apply-all: every proposal in the test split is written
            all_vals = {r: repairs[r].proposed_value for r in test}
            after_all = _count_errors(test, all_vals, truth)

            # CARE: auto-applied cells take the proposal, everything else stays dirty
            care_vals = {r: (auto[r] if r in auto else dirty[r]) for r in test}
            after_care = _count_errors(test, care_vals, truth)

            # counts among CARE's auto-applied changes
            changes = [r for r in auto if _norm(auto[r]) != dirty[r]]
            fixes = sum(1 for r in changes if dirty[r] != truth[r] and _norm(auto[r]) == truth[r])
            breaks = sum(1 for r in changes if dirty[r] == truth[r])
            still_wrong = len(changes) - fixes - breaks
            bound = (1.0 - 2.0 * a) * len(changes)
            per.append({
                "before": before, "after_all": after_all, "after_care": after_care,
                "edr_all": (before - after_all) / before if before else None,
                "edr_care": (before - after_care) / before if before else None,
                "edr_care_table": (fixes - breaks) / n_gold_table if n_gold_table else None,
                "n_test": len(test), "n_auto": len(auto), "n_changes": len(changes),
                "fixes": fixes, "breaks": breaks, "still_wrong": still_wrong,
                "bound_ok": (fixes - breaks) + 1e-9 >= bound,
                "realized_error": scores["CARE"].realized_error,
                "covered_seed": scores["CARE"].realized_error <= a + 1e-12,
            })

        def m(k):
            xs = [p[k] for p in per if p[k] is not None]
            return round(statistics.mean(xs), 4) if xs else None

        def mn(k):
            xs = [p[k] for p in per if p[k] is not None]
            return round(min(xs), 4) if xs else None

        rec = {
            "proposer": PRETTY.get(tag, tag), "dataset": ds, "detection": detection.split(":")[0],
            "scoring": scoring, "strata": sname, "alpha": a, "delta": delta, "seeds": seeds,
            "errors_before": m("before"), "n_test": m("n_test"),
            "edr_apply_all": m("edr_all"), "edr_apply_all_min": mn("edr_all"),
            "apply_all_negative_seeds": sum(1 for p in per if p["edr_all"] is not None and p["edr_all"] < 0),
            "edr_care": m("edr_care"), "edr_care_min": mn("edr_care"),
            "care_negative_seeds": sum(1 for p in per if p["edr_care"] is not None and p["edr_care"] < 0),
            "edr_care_table": m("edr_care_table"),
            "n_auto": m("n_auto"), "n_changes": m("n_changes"),
            "fixes": m("fixes"), "breaks": m("breaks"), "still_wrong": m("still_wrong"),
            "bound_violations": sum(1 for p in per if not p["bound_ok"]),
            "bound_violations_while_covered": sum(1 for p in per if (not p["bound_ok"]) and p["covered_seed"]),
        }
        out.append(rec)
        fmt = lambda x: "   -  " if x is None else f"{x:+.3f}"
        print(f"  a={a:<5} EDR apply-all {fmt(rec['edr_apply_all'])} "
              f"(neg {rec['apply_all_negative_seeds']}/{seeds}) | "
              f"CARE {fmt(rec['edr_care'])} (neg {rec['care_negative_seeds']}/{seeds}) | "
              f"auto {rec['n_auto']:.0f} changes {rec['n_changes']:.0f} "
              f"fixes {rec['fixes']:.0f} breaks {rec['breaks']:.0f} | "
              f"bound viol {rec['bound_violations']}", flush=True)
    return out


def write_tex(rows, path, alpha=0.2):
    """Compact LaTeX table: one line per pair at one alpha, oracle protocol."""
    sel = [r for r in rows if abs(r["alpha"] - alpha) < 1e-9 and r["detection"] == "oracle"]
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Proposer & Dataset & errors & EDR apply-all & EDR \textsc{Care} & changes/fixes/breaks \\",
         r"\midrule"]
    def f(x):
        return "---" if x is None else f"${x:+.2f}$"
    for r in sel:
        L.append(f"{r['proposer']} & \\textsc{{{r['dataset']}}} & {r['errors_before']:.0f} & "
                 f"{f(r['edr_apply_all'])} & {f(r['edr_care'])} & "
                 f"{r['n_changes']:.0f}/{r['fixes']:.0f}/{r['breaks']:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
    print(f"wrote {path}")


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--raha", action="store_true", help="also run the rows under the Raha detector")
    ap.add_argument("--only-raha", action="store_true")
    ap.add_argument("--skip-tax", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument("--baran-draw", default=None,
                    help="directory holding baran_<ds>_mapped.csv for another draw")
    ap.add_argument("--out", default="experiments/error_drop.csv")
    ap.add_argument("--tex", default=None, help="e.g. paper/tables/extra_error_drop.tex")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    configs = ([] if args.only_raha else list(TABLE1)) + (list(TABLE3) if (args.raha or args.only_raha) else [])
    rows = []
    for tag, ds, log, scoring, detection, rr in configs:
        if args.only and ds not in args.only:
            continue
        if args.skip_tax and ds == "tax":
            continue
        if args.baran_draw and tag == "baran":
            log = os.path.join(args.baran_draw, os.path.basename(log))
        if not os.path.exists(log):
            print(f"  [skip] {tag}/{ds}: missing {log}")
            continue
        det = detection.split(":")[0]
        print(f"\n===== {PRETTY.get(tag, tag)} / {ds}  [{scoring}, {det}] =====", flush=True)
        try:
            rows.extend(run_config(tag, ds, log, scoring, detection, rr,
                                   alphas=args.alphas, delta=args.delta, seeds=args.seeds))
        except SystemExit as e:
            print(f"  [skip] {tag}/{ds}: {e}")

    if not rows:
        print("nothing ran")
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")

    # summary
    pairs = {(r["proposer"], r["dataset"], r["detection"]) for r in rows}
    neg_all = {(r["proposer"], r["dataset"], r["detection"]) for r in rows
               if r["edr_apply_all"] is not None and r["edr_apply_all"] < 0}
    neg_care = {(r["proposer"], r["dataset"], r["detection"]) for r in rows if r["care_negative_seeds"]}
    bv = sum(r["bound_violations_while_covered"] for r in rows)
    print(f"pairs: {len(pairs)} | apply-all EDR < 0 on {len(neg_all)} pairs | "
          f"CARE EDR < 0 on any seed: {len(neg_care)} pairs | "
          f"bound (*) violated while covered: {bv} seed-runs")
    if neg_all:
        print("  apply-all negative on: " + ", ".join(f"{p}/{d}[{t}]" for p, d, t in sorted(neg_all)))
    if args.tex:
        write_tex(rows, args.tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

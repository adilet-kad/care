"""Ceiling on A(alpha) per proposer, dataset and partition, computed before calibration.

pi_alpha is the share of repairs in strata with error rate <= alpha; pi_alpha_floor
drops strata too small to certify at alpha (zero-error Clopper-Pearson bound at
delta/|Lambda|, scaled by --cal-frac). Partitions: marginal, mondrian (the one the
certifier uses), column. The output is sealed with a SHA-256 digest that
audit_sweep.py checks.

    python audit_alpha_ceiling.py --logs "csvs/baran_*_ni_*_mapped.csv"
    python audit_alpha_ceiling.py --logs "csvs/*_ni_*_mapped.csv" --per-stratum
    -> experiments/sweep_predictions.csv  (+ .sha256)
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import hashlib
import math
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

ALPHAS = (0.05, 0.1, 0.2)


def _norm(v):
    return str(v).strip().lower() if v is not None else ""


def tag_for(alpha) -> str:
    """Column suffix: 0.05 -> '05', 0.1 -> '1', 0.2 -> '2'."""
    return str(alpha).replace("0.", "", 1)


def split_log_name(path, known):
    """Split a log path into (proposer tag, dataset) by longest matching dataset name."""
    stem = os.path.basename(path)
    for suf in ("_mapped.csv", "_pred.csv", ".csv"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    for ds in sorted(known, key=len, reverse=True):
        if stem.endswith("_" + ds):
            return stem[: -len(ds) - 1], ds
    return None, None


def draw_of(path) -> int:
    """Draw index from a numbered subdirectory in the path; 0 if there is none."""
    for part in os.path.normpath(path).split(os.sep):
        if part.isdigit():
            return int(part)
    return 0


def n_min_cells(alpha, delta, m) -> int:
    """Smallest calibration count whose zero-error Clopper-Pearson bound is <= alpha."""
    from care.conformal.bounds import clopper_pearson_upper

    n = 1
    while clopper_pearson_upper(0, n, delta / m) > alpha:
        n += 1
        if n > 100_000:
            raise SystemExit(f"no n certifies alpha={alpha} at delta/{m}")
    return n


def partitions_for(art, gold):
    """Marginal, mondrian and column partitions; mondrian mirrors bench.experiments."""
    from care.conformal import by_cardinality, by_column, marginal_strata
    from care.core.artifact import CellKey

    counts = collections.Counter(CellKey.parse(r).col for r in gold)
    median = sorted(counts.values())[len(counts) // 2] if counts else 0
    mondrian = by_column() if median >= 60 else by_cardinality(art)
    base = "column" if median >= 60 else "cardinality"
    return [("marginal", marginal_strata, "marginal"),
            ("mondrian", mondrian, base),
            ("column", by_column(), "column")]


def read_log(path, gold):
    """ref -> correct, restricted to error cells."""
    out = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames or "row_id" not in rdr.fieldnames:
            raise SystemExit(f"{path} is not a CARE log (needs row_id,column,value)")
        for r in rdr:
            ref = f"{r['row_id']}::{r['column']}"
            if ref in gold:
                out[ref] = _norm(r.get("value")) == gold[ref]
    return out


def analyse(hits, sf, sf_base, alphas, delta, m, cal_frac):
    """One prediction row for one partition, plus its per-stratum detail."""
    per, ok = collections.Counter(), collections.Counter()
    for ref, good in hits.items():
        st = sf(ref)
        per[st] += 1
        ok[st] += good
    n = sum(per.values())
    if not n:
        return None, []

    rec = {"n": n, "acc": round(sum(ok.values()) / n, 4), "n_strata": len(per),
           "stratifier": sf_base,
           "pi0": round(sum(per[s] for s in per if ok[s] == per[s]) / n, 4)}
    for a in alphas:
        need = math.ceil(n_min_cells(a, delta, m) / cal_frac)
        good = [s for s in per if (per[s] - ok[s]) / per[s] <= a]
        big = [s for s in good if per[s] >= need]
        t = tag_for(a)
        rec[f"pi{t}"] = round(sum(per[s] for s in good) / n, 4)
        rec[f"pi{t}_floor"] = round(sum(per[s] for s in big) / n, 4)
        rec[f"strata{t}"] = len(big)
        rec[f"need{t}"] = need

    detail = [{"stratum": s, "repairs": per[s], "correct": ok[s],
               "error_rate": round((per[s] - ok[s]) / per[s], 4)}
              for s in sorted(per, key=lambda s: -per[s])]
    return rec, detail


def main() -> int:
    from bench.datasets import DATASETS, load
    from care.conformal.rcps import _DEFAULT_GRID, _candidate_grid

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", nargs="+", default=["csvs/*_ni_*_mapped.csv"],
                    help="glob(s) over mapped proposer logs")
    ap.add_argument("--alphas", nargs="*", type=float, default=list(ALPHAS))
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--cal-frac", type=float, default=0.4,
                    help="calibration share of each stratum; must match evaluate_split")
    ap.add_argument("--out", default="experiments/sweep_predictions.csv")
    ap.add_argument("--per-stratum", action="store_true",
                    help="also write <out>.strata.csv with every stratum's error rate")
    args = ap.parse_args()
    os.chdir(ROOT)

    m = len(_candidate_grid(_DEFAULT_GRID))
    paths = sorted({p for g in args.logs for p in glob.glob(g)})
    if not paths:
        raise SystemExit(f"no logs matched {args.logs}")

    print(f"|Lambda| = {m} (from the library grid), delta = {args.delta}, "
          f"cal_frac = {args.cal_frac}")
    print("floor, proposed cells needed per stratum: " + "  ".join(
        f"alpha={a}: {math.ceil(n_min_cells(a, args.delta, m) / args.cal_frac)}"
        for a in args.alphas))

    manifest = {}
    mpath = os.path.join("data", "ni_variants.csv")
    if os.path.exists(mpath):
        with open(mpath, newline="") as fh:
            manifest = {r["dataset"]: r for r in csv.DictReader(fh)}

    cache, rows, detail_rows = {}, [], []
    for p in paths:
        tag, ds = split_log_name(p, DATASETS)
        if not ds:
            print(f"  [skip] {p}: no registered dataset matches the file name")
            continue
        if ds not in cache:
            art, defects, _ = load(ds)
            cache[ds] = (art, {k: _norm(v) for k, v in defects.gold.items()})
        art, gold = cache[ds]
        hits = read_log(p, gold)
        if not hits:
            print(f"  [skip] {p}: no repairs on error cells")
            continue
        mrow = manifest.get(ds, {})
        for pname, sf, sf_base in partitions_for(art, gold):
            rec, detail = analyse(hits, sf, sf_base, args.alphas, args.delta, m,
                                  args.cal_frac)
            # coverage lets audit_sweep.py report automation over the whole error queue
            rows.append({"proposer": tag, "dataset": ds, "partition": pname,
                         "draw": draw_of(p),
                         "base": mrow.get("base", ds), "kind": mrow.get("kind", "-"),
                         "level": mrow.get("level", ""),
                         "cell_error_rate": mrow.get("cell_error_rate", ""),
                         "error_cells": len(gold),
                         "coverage": round(rec["n"] / len(gold), 4) if gold else 0.0,
                         "log": p, **rec})
            if pname != "marginal":
                for d in detail:
                    detail_rows.append({"proposer": tag, "dataset": ds,
                                        "partition": pname, "draw": draw_of(p), **d})

    if not rows:
        raise SystemExit("no logs produced a prediction")

    _porder = {"marginal": 0, "mondrian": 1, "column": 2}
    rows.sort(key=lambda r: (r["proposer"], r["base"], r["kind"],
                             str(r["level"]).zfill(3), _porder[r["partition"]], r["draw"]))
    fields = list(rows[0])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # the digest covers the prediction rows only, not the timestamp
    body = "\n".join(",".join(str(r[f]) for f in fields) for r in rows)
    digest = hashlib.sha256(body.encode()).hexdigest()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(args.out + ".sha256", "w") as fh:
        fh.write(f"{digest}  {os.path.basename(args.out)}\n"
                 f"written_at {stamp}\n"
                 f"grid {m}\ndelta {args.delta}\ncal_frac {args.cal_frac}\n"
                 f"rows {len(rows)}\n")
    if args.per_stratum and detail_rows:
        dpath = args.out.replace(".csv", "") + ".strata.csv"
        with open(dpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(detail_rows[0]))
            w.writeheader()
            w.writerows(detail_rows)
        print(f"wrote {dpath} ({len(detail_rows)} strata)")

    t2 = tag_for(0.2)
    print(f"\n{'proposer':<11}{'dataset':<21}{'part':<9}{'dr':>3}{'rate':>7}{'n':>7}"
          f"{'cov':>6}{'acc':>7}{'pi0':>7}{'pi.2':>7}{'pi.2fl':>8}{'strata':>8}")
    for r in rows:
        rate = float(r["cell_error_rate"]) if r["cell_error_rate"] else float("nan")
        print(f"{r['proposer']:<11}{r['dataset']:<21}{r['partition']:<9}{r['draw']:>3}"
              f"{rate:>7.3f}{r['n']:>7}{r['coverage']:>6.2f}{r['acc']:>7.3f}"
              f"{r['pi0']:>7.3f}{r[f'pi{t2}']:>7.3f}{r[f'pi{t2}_floor']:>8.3f}"
              f"{r[f'strata{t2}']:>4}/{r['n_strata']:<3}")

    ndraw = collections.Counter((r["proposer"], r["dataset"]) for r in rows
                                if r["partition"] == "mondrian")
    single = [k for k, v in ndraw.items() if v == 1]
    if single:
        print(f"\n{len(single)} of {len(ndraw)} (proposer, variant) pairs have ONE draw. "
              f"Baran is unseeded:\na single draw is a draw, not a measurement. See "
              f"docs/REPRODUCE.md section 3 for the three-draw protocol.")

    covs = {(r["proposer"], r["base"], r["kind"]): [] for r in rows}
    for r in rows:
        if r["partition"] == "mondrian":
            covs[(r["proposer"], r["base"], r["kind"])].append(r["coverage"])
    swings = {k: (min(v), max(v)) for k, v in covs.items() if len(v) > 1
              and max(v) - min(v) > 0.15}
    if swings:
        print("\ncoverage changes along these series (see A_queue in sweep.csv):")
        for (pr, base, kind), (lo, hi) in sorted(swings.items()):
            print(f"  {pr}/{base} {kind}: proposals cover {lo:.2f} to {hi:.2f} of the "
                  f"error queue, so A(alpha)'s denominator is not fixed")
    print(f"\nwrote {args.out} ({len(rows)} predictions)")
    print(f"sealed sha256 {digest[:16]}...  at {stamp}")
    print("Commit this file before running the sweeps. audit_sweep.py verifies the seal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

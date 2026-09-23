"""audit_alpha_ceiling.py -- write the ceiling down BEFORE spending any calibration.

The paper's mechanism claim is that certifiable automation is set by how a proposer's
accuracy is distributed across strata, not by its mean. The sharp form of that claim is
a prediction: pi_alpha, the share of a proposer's repairs that fall in strata whose
error rate is at most alpha, is the ceiling on A(alpha), and the finite-sample floor
knocks out strata too thin to certify however accurate they are. pi_0 is the alpha = 0
case; the paper reports pi_0 in Table 1 and the alpha-indexed ceiling via audit_sweep.py.

A prediction that is computed after the measurement is not a prediction. This script
computes the ceiling from the proposer's log and the gold labels alone -- no
calibration split, no threshold, no conformal machinery -- and seals the result with a
SHA-256 digest over the rows. Run it before `bench.run_study`, commit the output, and
`audit_sweep.py` will refuse to join a predictions file whose digest does not match its
contents. That is what lets the paper say the ceiling was written down first.

WHAT IS COMPUTED, AND WHY EACH PIECE IS THERE

  acc          accuracy on error cells the proposer touched -- the aggregate number the
               paper argues is not predictive. Reported so the sweep can show it is not.
  pi_0         share of repairs in strata the proposer gets entirely right.
  pi_alpha     share of repairs in strata with error rate <= alpha. The raw ceiling.
  pi_alpha_fl  the same, after dropping strata too small to certify at alpha even with
               zero observed errors. THIS is the quantity A(alpha) should track: a
               perfect stratum holding 12 repairs certifies nothing at alpha = 0.05,
               so counting it in the ceiling guarantees an apparent shortfall that has
               nothing to do with the proposer.

The floor is not the paper's approximation. For a stratum with zero errors in n
calibration cells the exact Clopper-Pearson bound at level delta/|Lambda| is
1 - (delta/|Lambda|)^(1/n); n_min is the smallest n where that is <= alpha, found by
search, and the requirement is converted to PROPOSED cells by dividing by `--cal-frac`
(0.4, the split `evaluate_split` uses). |Lambda| is read from the library
(`care.conformal.rcps._DEFAULT_GRID`), never hardcoded -- the grid is 21 candidates,
which gives n_min = 105 / 51 / 24 at alpha = 0.05 / 0.1 / 0.2 and delta = 0.1.

ONE ROW PER PARTITION, BECAUSE A CEILING IS ONLY DEFINED AGAINST A PARTITION

The output is long: one row per (proposer, dataset, partition), for the three
partitions the sweep reports.

  marginal   one stratum. pi_alpha is 1 if the proposer's overall error is at most
             alpha and 0 otherwise -- the degenerate ceiling the paper argues is
             uninformative, kept so the sweep can show it.
  mondrian   the partition `bench.experiments` actually calibrates on: `by_column()`
             when the median column holds at least 60 error cells, `by_cardinality`
             otherwise. THIS row is the prediction A(alpha) is compared against.
  column     always per-column, whether or not the certifier uses columns. This is
             the convention Table 1's pi_0 follows, so it is reported as a diagnostic
             -- and on the small corpora it differs from `mondrian`, which matters:
             hospital's median column holds far fewer than 60 error cells, so its
             certifier partitions into ~3 cardinality buckets, not 17 columns.

Predicting a ceiling over columns while the certifier partitions by cardinality
bucket would compare two different partitions and blame the difference on the method,
so `audit_sweep.py` joins each measured `strata` value to the row of the same name.

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
    """0.05 -> '05', 0.1 -> '1', 0.2 -> '2' -- the column suffix used everywhere."""
    return str(alpha).replace("0.", "", 1)


def split_log_name(path, known):
    """('baran', 'beers_ni_inner30') from 'csvs/baran_beers_ni_inner30_mapped.csv'.

    Both halves contain underscores, so the split is by longest matching dataset name
    rather than by position. `baran_raha_hospital_ni_orig` resolves to the tag
    `baran_raha` and not to a dataset called `raha_hospital_ni_orig`.
    """
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
    """Draw index from a path like `csvs/baran_var_beers_ni_inner30/2/..._mapped.csv`.

    Baran's 20-tuple active-learning sample is unseeded, so one log is one draw from a
    distribution rather than a measurement, and a curve built from single draws would
    inherit that spread without showing it. Draws live in numbered subdirectories;
    the shipped log (no numeric component in its path) is draw 0. Everything downstream
    keys on this, so `audit_sweep.py --median` can collapse draws per point.
    """
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
        if n > 100_000:                       # unreachable for sane alpha; fail loudly
            raise SystemExit(f"no n certifies alpha={alpha} at delta/{m}")
    return n


def partitions_for(art, gold):
    """The three partitions the sweep reports, in the order they are written.

    `mondrian` mirrors `bench.experiments` exactly -- change one and the other must
    change with it, or the prediction stops describing the certifier.
    """
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
    """ref -> correct, restricted to error cells (the population Table 1 scores)."""
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
            # A(alpha) is measured over the cells the proposer PROPOSED on --
            # `evaluate_split` sets refs = [r for r in gold if r in repairs] -- so a
            # proposer that abstains more as the table degrades shrinks its own
            # denominator. Baran's coverage on these variants runs from 0.59 down to
            # 0.18, which is enough to move an automation curve on its own. Carry the
            # coverage so `audit_sweep.py` can also report automation over the FIXED
            # denominator of the whole error queue.
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
                    # `draw` is part of the key: without it, three draws of one variant
                    # write three rows per stratum that look like one dataset's strata
                    # listed three times, and any aggregation over the file silently
                    # triple-counts.
                    detail_rows.append({"proposer": tag, "dataset": ds,
                                        "partition": pname, "draw": draw_of(p), **d})

    if not rows:
        raise SystemExit("no logs produced a prediction")

    _porder = {"marginal": 0, "mondrian": 1, "column": 2}
    rows.sort(key=lambda r: (r["proposer"], r["base"], r["kind"],
                             str(r["level"]).zfill(3), _porder[r["partition"]], r["draw"]))
    fields = list(rows[0])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # The seal covers the prediction rows only -- not the timestamp line, which is
    # written after it. Anyone can recompute it; audit_sweep.py does.
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
        print("\nCOVERAGE MOVES ALONG THESE SERIES -- read A_queue in sweep.csv too:")
        for (pr, base, kind), (lo, hi) in sorted(swings.items()):
            print(f"  {pr}/{base} {kind}: proposals cover {lo:.2f} to {hi:.2f} of the "
                  f"error queue, so A(alpha)'s denominator is not fixed")
    print(f"\nwrote {args.out} ({len(rows)} predictions)")
    print(f"sealed sha256 {digest[:16]}...  at {stamp}")
    print("Commit this file BEFORE running the sweeps. audit_sweep.py verifies the seal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

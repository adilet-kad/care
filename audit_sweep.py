"""audit_sweep.py -- join the sealed ceiling predictions to what the sweeps measured.

`audit_alpha_ceiling.py` wrote pi_alpha before any calibration was spent. The sweeps
then measured A(alpha). This script puts them side by side, one row per
(proposer, dataset, partition, alpha), and answers the three questions the error-rate
experiment exists to answer:

  1. Did the budget hold everywhere?  `covered` must be true on every row. A single
     false is a bug in the import or the certifier and invalidates the rest.
  2. Does A(alpha) track the floor-adjusted ceiling as the proposer degrades?
     Reported as the signed gap A - pi_alpha_floor, its mean and its extremes.
  3. Is the collapse to the empty set predicted before it happens? Reported as the
     first variant (by measured cell error rate) where pi_alpha_floor hits zero
     against the first where A hits zero.

THE GAP HAS A SIGN, AND BOTH SIGNS MEAN SOMETHING

  A < pi_alpha_floor  -- the usual case. The certifier could not reach the ceiling:
       the stratum was thin, the bound was conservative, or the score put some correct
       repairs below the threshold. The size of this shortfall is the price of the
       finite-sample correction.

  A > pi_alpha_floor  -- NOT a violation, and the script does not report it as one.
       pi_alpha is the ceiling on what a rule that accepts or rejects WHOLE STRATA can
       automate. CARE's threshold selects a subset of a stratum, so whenever the
       confidence score ranks correctness inside a stratum, it can certify part of a
       stratum whose overall error exceeds alpha and legitimately beat the ceiling.
       HoloClean on flights does exactly this under marginal control: pooled error is
       0.258, so the marginal pi_0.2 is 0, yet A(0.2) is 0.74 on the high-confidence
       tail. Exceedances are therefore counted and reported separately, and they are
       the direct measurement of how much within-stratum ranking a proposer's score
       carries -- the same quantity Section 8.3 argues is usually absent.

So the honest claim the joined file supports is "pi_alpha is the ceiling attainable
without within-stratum ranking", and the exceedance count is the evidence for how
often that qualifier binds.

    python audit_sweep.py                                  # all results_sweep_* dirs
    python audit_sweep.py --results experiments/results_sweep_baran_var
    python audit_sweep.py --no-verify-seal                 # only for a re-run in place
    -> experiments/sweep.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import hashlib
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

MEASURED_PARTITIONS = ("marginal", "mondrian")   # the strata the pareto files carry
# Two ratios over a few thousand cells that select the SAME cells still differ in the
# fourth decimal. Anything inside this is a tie, not a gap in either direction.
TIE = 1e-3


def tag_for(alpha) -> str:
    return str(alpha).replace("0.", "", 1)


def verify_seal(path) -> str:
    """Recompute the digest over the prediction rows and compare to the sidecar.

    The point of pre-registration is that the predictions cannot be quietly edited
    after the measurement arrives. Without this check the seal is decoration.
    """
    side = path + ".sha256"
    if not os.path.exists(side):
        return f"no {os.path.basename(side)} -- predictions are UNSEALED"
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        fields = rdr.fieldnames or []
        body = "\n".join(",".join(str(r[f]) for f in fields) for r in rdr)
    got = hashlib.sha256(body.encode()).hexdigest()
    want = open(side).readline().split()[0]
    if got != want:
        return (f"SEAL BROKEN: {path} hashes to {got[:16]}... but the sidecar records "
                f"{want[:16]}...\nThe predictions were edited after they were sealed. "
                f"Regenerate both, or explain the edit.")
    return ""


def load_predictions(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_results(dirs):
    """(log_path, dataset, strata, alpha) -> row, from every *_pareto.csv found."""
    out = {}
    for d in dirs:
        # Recursive: draws live in per-variant, per-draw subdirectories so that two
        # draws of the same variant cannot overwrite each other's pareto file.
        for p in sorted(glob.glob(os.path.join(d, "**", "*_pareto.csv"), recursive=True)):
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r["baseline"] != "CARE":
                        continue
                    backend = r.get("backend", "")
                    log = backend.split(":", 1)[1] if ":" in backend else backend
                    key = (log, r["dataset"], r["strata"], round(float(r["alpha"]), 6))
                    out[key] = {**r, "_file": p}
    return out


def write_medians(rows, out):
    """Collapse draws to a median per (proposer, dataset, partition, alpha).

    Baran selects its 20 labelled tuples by unseeded active learning, so each log is a
    draw. Table 1 already reports medians over six draws for this reason; a curve built
    from single draws would carry less evidence than the table it sits beside. The
    median rather than the mean because certification is close to all-or-nothing per
    stratum -- the distribution over draws is bimodal, and a mean between two modes
    describes no run that happened.

    Also carries min and max, which is what the figure's range bars and the caption's
    spread statement need. Returns the path written, or None if every point is a single
    draw (in which case a median file would only invite over-reading).
    """
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["proposer"], r["dataset"], r["partition"], r["alpha"])].append(r)
    if max(len(v) for v in groups.values()) < 2:
        return None

    out_rows = []
    for k, sub in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2],
                                                         kv[0][3])):
        first = sub[0]
        def med(f):
            return round(statistics.median(float(r[f]) for r in sub), 4)
        A = [float(r["A"]) for r in sub]
        out_rows.append({
            "proposer": k[0], "base": first["base"], "kind": first["kind"],
            "level": first["level"], "dataset": k[1],
            "cell_error_rate": first["cell_error_rate"], "partition": k[2],
            "stratifier": first["stratifier"], "alpha": k[3], "draws": len(sub),
            "n": med("n"), "coverage": med("coverage"), "acc": med("acc"),
            "pi0": med("pi0"), "pi_alpha": med("pi_alpha"),
            "pi_alpha_floor": med("pi_alpha_floor"),
            "A": med("A"), "A_min": round(min(A), 4), "A_max": round(max(A), 4),
            "A_queue": med("A_queue"), "R": med("R"),
            # A median that hides a violation would be the worst possible summary, so
            # coverage is reported as the WORST draw, not the typical one.
            "covered": all(str(r["covered"]).lower() == "true" for r in sub),
            "R_max": round(max(float(r["R"]) for r in sub), 4),
            "gap": med("gap"),
            "empty": all(r["empty"] for r in sub),
            "predicted_empty": all(r["predicted_empty"] for r in sub),
        })
    path = out.replace(".csv", "") + "_median.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--predictions", default="experiments/sweep_predictions.csv")
    ap.add_argument("--results", nargs="*", default=None,
                    help="result directories; default every experiments/results_sweep_*")
    ap.add_argument("--out", default="experiments/sweep.csv")
    ap.add_argument("--no-verify-seal", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    if not os.path.exists(args.predictions):
        raise SystemExit(f"no predictions at {args.predictions}; "
                         f"run audit_alpha_ceiling.py first")
    if not args.no_verify_seal:
        problem = verify_seal(args.predictions)
        if problem:
            raise SystemExit(problem)

    dirs = args.results or sorted(glob.glob("experiments/results_sweep_*"))
    if not dirs:
        raise SystemExit("no experiments/results_sweep_* directories; run the sweeps first")
    preds = load_predictions(args.predictions)
    res = load_results(dirs)
    if not res:
        raise SystemExit(f"no CARE rows in any *_pareto.csv under {dirs}")

    alphas = sorted({round(float(k[3]), 6) for k in res})
    rows, missing = [], 0
    for pr in preds:
        if pr["partition"] not in MEASURED_PARTITIONS:
            continue                       # `column` is a diagnostic, never measured
        for a in alphas:
            key = (pr["log"], pr["dataset"], pr["partition"], a)
            m = res.get(key)
            if m is None:
                missing += 1
                continue
            t = tag_for(a)
            pi = float(pr.get(f"pi{t}", "nan"))
            pif = float(pr.get(f"pi{t}_floor", "nan"))
            A = 1.0 - float(m["human_cost"])
            # A is the auto-applied share of the cells the proposer PROPOSED on.
            # A_queue rebases it on the whole error queue, which is the denominator
            # that stays fixed as the injected rate changes. Quote A against the
            # ceiling (they share a denominator) and A_queue when comparing points
            # along a curve whose coverage moves.
            cov = float(pr["coverage"]) if pr.get("coverage") not in ("", None) else 1.0
            rows.append({
                "proposer": pr["proposer"], "base": pr["base"], "kind": pr["kind"],
                "level": pr["level"], "dataset": pr["dataset"],
                "cell_error_rate": pr["cell_error_rate"],
                "partition": pr["partition"], "stratifier": pr.get("stratifier", ""),
                "draw": int(pr.get("draw", 0) or 0), "alpha": a,
                "n": pr["n"], "error_cells": pr.get("error_cells", ""),
                "coverage": cov, "acc": pr["acc"], "pi0": pr["pi0"],
                "pi_alpha": round(pi, 4), "pi_alpha_floor": round(pif, 4),
                "n_strata": pr["n_strata"], "strata_certifiable": pr.get(f"strata{t}", ""),
                "A": round(A, 4), "A_queue": round(A * cov, 4),
                "R": round(float(m["realized_error"]), 4),
                "covered": m["covered"],
                "gap": round(A - pif, 4),
                "empty": A <= 1e-9,
                "predicted_empty": pif <= 1e-9,
            })

    if not rows:
        raise SystemExit("predictions and results share no (log, dataset, strata, alpha) key.\n"
                         "The usual cause is sweeping a different mapped log than the one "
                         "audit_alpha_ceiling.py read; the join is on the log path in the "
                         "pareto file's `backend` column.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    med_path = write_medians(rows, args.out)
    if med_path:
        print(f"\nwrote {med_path} -- Baran is unseeded, so every plotted point is the "
              f"MEDIAN\nover draws with a min-max range; make_figures.fig9 prefers this "
              f"file.")

    # ---- 1. coverage ------------------------------------------------------
    viol = [r for r in rows if str(r["covered"]).lower() != "true"]
    print(f"\n1. COVERAGE  {len(rows) - len(viol)}/{len(rows)} rows covered")
    for r in viol:
        print(f"   !! {r['proposer']}/{r['dataset']} {r['partition']} a={r['alpha']}: "
              f"R={r['R']} > alpha, A={r['A']}")
    if not viol:
        print("   no budget violation at any error rate, partition or alpha.")

    # ---- 2. tracking ------------------------------------------------------
    print("\n2. TRACKING  A(alpha) against the floor-adjusted ceiling")
    print(f"   {'partition':<10}{'alpha':>6}{'rows':>6}{'mean gap':>10}"
          f"{'worst short':>13}{'exceed':>8}{'max exceed':>12}")
    for part in MEASURED_PARTITIONS:
        for a in alphas:
            sub = [r for r in rows if r["partition"] == part and r["alpha"] == a]
            if not sub:
                continue
            gaps = [r["gap"] for r in sub]
            # TIE is 0.001, not 0: A and the ceiling are both ratios over a few
            # thousand cells, so they routinely differ in the fourth decimal when the
            # certified set is the same set. Counting those as exceedances would
            # manufacture a within-stratum-ranking finding out of float noise.
            over = [g for g in gaps if g > TIE]
            print(f"   {part:<10}{a:>6}{len(sub):>6}{statistics.mean(gaps):>10.3f}"
                  f"{min(gaps):>13.3f}{len(over):>4}/{len(sub):<3}"
                  f"{(max(over) if over else 0.0):>12.3f}")
    print("   negative gap = the certifier fell short of the ceiling (the usual case).")
    print("   exceed = A above the ceiling, i.e. the score ranked WITHIN a stratum;")
    print("   see the module docstring -- this is a measurement, not a violation.")

    # A proposer whose score is constant cannot rank within a stratum, so for it the
    # ceiling is not a bound but an equality, and any deviation is the calibration
    # sample's error rate differing from the stratum's. Worth stating separately:
    # it is the cleanest possible test of the identity, and Baran is exactly this case
    # (s_hat is 0.666667 on every repair in every Baran log, base and variant).
    flat = [r for r in rows if r["partition"] == "mondrian"]
    if flat:
        exact = sum(1 for r in flat if abs(r["gap"]) <= TIE)
        print(f"   A == ceiling exactly on {exact}/{len(flat)} mondrian rows "
              f"(expect ~all of them for a constant-confidence proposer).")

    # ---- 3. collapse ------------------------------------------------------
    print("\n3. COLLAPSE  first cell error rate at which each series certifies nothing")
    print(f"   {'proposer':<11}{'base':<10}{'kind':<7}{'part':<10}{'alpha':>6}"
          f"{'predicted':>11}{'observed':>10}")
    groups = collections.defaultdict(list)
    for r in rows:
        if r["cell_error_rate"] in ("", None):
            continue
        groups[(r["proposer"], r["base"], r["kind"], r["partition"], r["alpha"])].append(r)
    agree = total = 0
    for k, sub in sorted(groups.items()):
        sub.sort(key=lambda r: float(r["cell_error_rate"]))
        pe = next((float(r["cell_error_rate"]) for r in sub if r["predicted_empty"]), None)
        oe = next((float(r["cell_error_rate"]) for r in sub if r["empty"]), None)
        if pe is None and oe is None:
            continue
        total += 1
        agree += pe is not None and oe is not None and abs(pe - oe) < 1e-9
        print(f"   {k[0]:<11}{k[1]:<10}{k[2]:<7}{k[3]:<10}{k[4]:>6}"
              f"{('-' if pe is None else f'{pe:.3f}'):>11}"
              f"{('-' if oe is None else f'{oe:.3f}'):>10}")
    if total:
        print(f"   the predicted and observed collapse points agree on {agree}/{total} series.")

    # ---- 4. does the partition itself change along the curve? -------------
    # `mondrian` is by_column() when the median column holds >= 60 error cells and
    # by_cardinality otherwise, so injecting more errors can flip the certifier's
    # partition mid-sweep. A curve whose x-axis is the error rate and whose strata
    # silently change at some level is not one experiment; say so rather than let a
    # reader read the discontinuity as a proposer effect.
    flips = []
    series = collections.defaultdict(list)
    for r in rows:
        if r["partition"] == "mondrian":
            series[(r["proposer"], r["base"], r["kind"], r["alpha"])].append(r)
    for k, sub in sorted(series.items()):
        kinds = {r["stratifier"] for r in sub if r["stratifier"]}
        if len(kinds) > 1:
            flips.append((k, sorted(kinds)))
    if flips:
        print("\n4. PARTITION CHANGES ALONG THE CURVE (record or pin these)")
        for k, kinds in flips:
            print(f"   {k[0]}/{k[1]} {k[2]} alpha={k[3]}: mondrian is "
                  f"{' then '.join(kinds)} at different levels")
    else:
        print("\n4. PARTITION  mondrian resolves to the same partition at every level.")

    if missing:
        print(f"\n{missing} (prediction, alpha) pairs had no matching sweep row -- "
              f"those variants have not been swept yet.")
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 1 if viol else 0


if __name__ == "__main__":
    raise SystemExit(main())

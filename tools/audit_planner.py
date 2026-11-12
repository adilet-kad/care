"""Label planner: predicted vs procedure automation, review savings, break-even sizes.

Validation compares the planner's closed-form automation with evaluate_split on the
unranked pairs (Baran, GIDCL). Savings compare the 40% default, the pilot-based planner
and an oracle planner, scored with the exact hypergeometric admission probability.

    python tools/audit_planner.py              -> experiments/planner_{validation,savings,breakeven}.csv
    python tools/audit_planner.py --skip-real  # synthetic break-even only
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CARE_PROGRESS", "0")

from care.conformal.planner import (break_even_size, min_labels, p_certify_finite,  # noqa: E402
                                    plan)

DATASETS = ["hospital", "beers", "flights", "rayyan", "tax"]
PAIRS = [("Baran", "baran", "csvs/baran_%s_mapped.csv", "results_baran_errors"),
         ("GIDCL", "gidcl", "csvs/gidcl_%s_mapped.csv", "results_gidcl")]
ALPHAS = [0.05, 0.1, 0.2]
DELTA = 0.1
CAL_FRAC = 0.4
PILOT = 20
PILOT_DRAWS = 30


def _w(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def strata_counts(tag, log, ds):
    """{stratum: (m, wrong)} under the experiments' stratifier, and whether scores are constant per stratum."""
    from bench.experiments import _mondrian_strata_fn
    from bench.study import load_and_propose
    art, defects, repairs, _ = load_and_propose(ds, f"{tag}:{log}", detection="oracle",
                                                fast_verify=True)
    gold = {k: (str(v).lower().strip() if v is not None else v)
            for k, v in defects.gold.items()}
    sf = _mondrian_strata_fn(art, defects)
    m, wrong, scores = collections.Counter(), collections.Counter(), collections.defaultdict(set)
    for r in gold:
        if r not in repairs:
            continue
        rep = repairs[r]
        s = sf(rep)
        pv = str(rep.proposed_value).lower().strip() if rep.proposed_value is not None else None
        m[s] += 1
        wrong[s] += pv != gold[r]
        scores[s].add(round(float(rep.s_hat), 12))
    unranked = all(len(v) == 1 for v in scores.values())
    return {s: (m[s], wrong[s]) for s in m}, unranked, (repairs, gold, sf)


def predicted_automation(counts, alpha, seeds):
    """Expected automation of one split, and the s.d. of its mean over ``seeds``."""
    num = den = var = 0.0
    for m, w in counts.values():
        n = int(m * CAL_FRAC)
        pc = p_certify_finite(n, m, w, alpha, DELTA)
        num += (m - n) * pc
        var += (m - n) ** 2 * pc * (1 - pc)
        den += m - n
    if not den:
        return 0.0, 0.0
    return num / den, (var ** 0.5) / den / seeds ** 0.5


def certifier_automation(run, alpha, seeds):
    """Mean automation of the real procedure (evaluate_split) over ``seeds`` splits."""
    from bench.study import evaluate_split
    repairs, gold, sf = run
    return statistics.mean(
        1.0 - evaluate_split(repairs, gold, alpha=alpha, delta=DELTA, seed=s,
                             strata_fn=sf)["CARE"].human_cost for s in range(seeds))


def measured_automation(results_dir, ds, alpha):
    path = f"experiments/{results_dir}/{ds}_pareto.csv"
    if not os.path.exists(path):
        return None
    for r in csv.DictReader(open(path)):
        if r["baseline"] == "CARE" and r["strata"] == "mondrian" and \
                abs(float(r["alpha"]) - alpha) < 1e-9:
            return 1.0 - float(r["human_cost"])
    return None


def _score_units(units, members, kappa, alpha, extra_labels=0):
    """Expected saving of a plan on the real strata (hypergeometric)."""
    sv = -(kappa - 1.0) * extra_labels
    for u in units:
        if u.action != "calibrate":
            continue
        m = sum(members[s][0] for s in u.strata)
        w = sum(members[s][1] for s in u.strata)
        n = min(u.labels, m)
        sv += (m - n) * p_certify_finite(n, m, w, alpha, DELTA) - (kappa - 1.0) * n
    return sv


def savings(counts, alpha, kappa, rng):
    total = sum(m for m, _ in counts.values())
    # default: 40% of every stratum
    default = sum((m - int(m * CAL_FRAC)) * p_certify_finite(int(m * CAL_FRAC), m, w, alpha,
                                                             DELTA)
                  - (kappa - 1.0) * int(m * CAL_FRAC) for m, w in counts.values())
    default_labels = sum(int(m * CAL_FRAC) for m, _ in counts.values())
    # oracle: planner given the true rates
    o = plan({s: (m, w / m) for s, (m, w) in counts.items()}, alpha=alpha, delta=DELTA,
             kappa=kappa)
    oracle = _score_units(o.units, counts, kappa, alpha)
    # pilot: PILOT labels per stratum, disjoint from calibration, cost charged
    pil, pil_labels = [], []
    for _ in range(PILOT_DRAWS):
        beliefs, rest, spent = {}, {}, 0
        for s, (m, w) in counts.items():
            k = min(PILOT, m)
            drawn = sum(rng.sample([1] * w + [0] * (m - w), k))
            spent += k
            beliefs[s] = (m - k, (1 + drawn, 1 + k - drawn))
            rest[s] = (m - k, w - drawn)
        p = plan(beliefs, alpha=alpha, delta=DELTA, kappa=kappa)
        pil.append(_score_units(p.units, rest, kappa, alpha, extra_labels=spent))
        pil_labels.append(spent + p.labels)
    return {"queue": total,
            "default_saving": round(default / total, 4),
            "default_labels": default_labels,
            "planner_saving": round(statistics.mean(pil) / total, 4),
            "planner_labels": round(statistics.mean(pil_labels)),
            "oracle_saving": round(oracle / total, 4),
            "oracle_labels": o.labels}


def real(args):
    val, sav = [], []
    rng = random.Random(0)
    for name, tag, pattern, rdir in PAIRS:
        for ds in DATASETS:
            log = pattern % ds
            if not os.path.exists(log):
                print(f"  [skip] {name}/{ds}: no {log}")
                continue
            print(f"\n== {name} / {ds}", flush=True)
            counts, unranked, run = strata_counts(tag, log, ds)
            print(f"   {len(counts)} strata, {sum(m for m, _ in counts.values()):,} repairs, "
                  f"unranked={unranked}")
            for a in ALPHAS:
                pred, sd = predicted_automation(counts, a, args.seeds)
                meas = certifier_automation(run, a, args.seeds)
                pub = measured_automation(rdir, ds, a)
                z = abs(pred - meas) / sd if sd > 0 else (0.0 if abs(pred - meas) < 1e-9
                                                          else float("inf"))
                print(f"   alpha={a}: predicted A={pred:.3f} (s.d. of a {args.seeds}-seed mean "
                      f"{sd:.3f})  procedure A={meas:.3f}  published 20-seed A="
                      f"{'-' if pub is None else f'{pub:.3f}'}", flush=True)
                val.append({"proposer": name, "dataset": ds, "alpha": a,
                            "strata": len(counts), "unranked": unranked,
                            "predicted_A": round(pred, 4), "predicted_sd": round(sd, 4),
                            "certifier_A": round(meas, 4), "seeds": args.seeds,
                            "abs_diff": round(abs(pred - meas), 4), "z": round(z, 2),
                            "published_A_20seeds": None if pub is None else round(pub, 4)})
                for kappa in (1.0, 2.0):
                    s = savings(counts, a, kappa, rng)
                    print(f"     kappa={kappa:g}: saved default {s['default_saving']:+.3f} "
                          f"({s['default_labels']} labels) | planner {s['planner_saving']:+.3f}"
                          f" ({s['planner_labels']}) | oracle {s['oracle_saving']:+.3f} "
                          f"({s['oracle_labels']})", flush=True)
                    sav.append({"proposer": name, "dataset": ds, "alpha": a, "kappa": kappa,
                                **s})
    _w("experiments/planner_validation.csv", val)
    _w("experiments/planner_savings.csv", sav)
    ok = [r for r in val if r["unranked"]]
    if ok:
        d = [r["abs_diff"] for r in ok]
        print(f"\n  unranked pairs: {len(ok)} (pair, alpha) points, mean |pred - procedure| = "
              f"{statistics.mean(d):.4f}, max = {max(d):.4f}, "
              f"{sum(r['z'] <= 2 for r in ok)} within 2 s.d.")


def synthetic():
    rows = []
    print("\n== (a) break-even stratum size, kappa = 2")
    for a in ALPHAS:
        ps = [round(a * f, 4) for f in (0, .1, .2, .3, .4, .5, .6, .7, .8)]
        for p in ps:
            m = break_even_size(p, alpha=a, delta=DELTA, kappa=2.0)
            rows.append({"panel": "a", "alpha": a, "p": p, "x": None, "series": "breakeven",
                         "value": m})
            print(f"   alpha={a} p={p:.3f}: {m}")
        rows.append({"panel": "a", "alpha": a, "p": None, "x": None, "series": "min_labels",
                     "value": min_labels(a, DELTA)})
    print("\n== (b) saving vs schema width, 20,000 repairs, alpha=0.1, kappa=2")
    total, a, kappa = 20_000, 0.1, 2.0
    for W in (5, 10, 20, 50, 100, 200, 500):
        acc = collections.defaultdict(list)
        for seed in range(10):
            rng = random.Random(seed)
            # heterogeneous columns: most clean, a few bad (mean error ~4%)
            rates = [rng.choice([0.0, 0.01, 0.02, 0.03, 0.05, 0.15]) for _ in range(W)]
            sizes = [total // W] * W
            counts = {f"c{i}": (m, round(m * p)) for i, (m, p) in enumerate(zip(sizes, rates))}
            d = sum((m - int(m * CAL_FRAC)) * p_certify_finite(int(m * CAL_FRAC), m, w, a,
                                                               DELTA)
                    - (kappa - 1) * int(m * CAL_FRAC) for m, w in counts.values())
            acc["per_column_40"].append(d / total)
            mw = sum(w for _, w in counts.values())
            n = int(total * CAL_FRAC)
            acc["pooled_40"].append(((total - n) * p_certify_finite(n, total, mw, a, DELTA)
                                     - (kappa - 1) * n) / total)
            o = plan({s: (m, w / m) for s, (m, w) in counts.items()}, alpha=a, delta=DELTA,
                     kappa=kappa, pool=False)
            acc["per_column_best_n"].append(_score_units(o.units, counts, kappa, a) / total)
            o = plan({s: (m, w / m) for s, (m, w) in counts.items()}, alpha=a, delta=DELTA,
                     kappa=kappa)
            acc["planner"].append(_score_units(o.units, counts, kappa, a) / total)
        for k, v in acc.items():
            rows.append({"panel": "b", "alpha": a, "p": None, "x": W, "series": k,
                         "value": round(statistics.mean(v), 4)})
        print(f"   W={W:>3}: " + "  ".join(f"{k} {statistics.mean(v):+.3f}"
                                         for k, v in acc.items()))
    _w("experiments/planner_breakeven.csv", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--seeds", type=int, default=200,
                    help="splits of the real procedure per (pair, alpha)")
    a = ap.parse_args()
    if not a.skip_synthetic:
        synthetic()
    if not a.skip_real:
        real(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

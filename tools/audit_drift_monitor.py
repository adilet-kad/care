"""Corruption drift between calibration and deployment, and a per-batch audit monitor.

CARE calibrates on one Ni et al. variant of a table and is deployed on Baran's repairs
for another corruption level; the same-variant arm is the no-drift control. The monitor
audits K written repairs per stratum and commits the stratum only if the Clopper-Pearson
upper bound on the audited error is at most alpha (a lower-bound rule is also reported).

    python tools/audit_drift_monitor.py   -> experiments/drift_monitor.csv
"""
from __future__ import annotations

import collections
import csv
import math
import os
import random
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CARE_PROGRESS", "0")

# (table, variant calibrated on): the levels at which Baran admits a non-empty set.
SOURCES = [("beers", 30), ("beers", 70)]
TARGETS = [10, 30, 50, 70]
ALPHAS = [0.1, 0.2]
DELTA, SEEDS, K, MON_LEVEL = 0.1, 50, 30, 0.1
OUT = "experiments/drift_monitor.csv"


def _norm(v):
    return str(v).strip().lower() if v is not None else ""


def cp_lower(k, n, level):
    """One-sided Clopper-Pearson lower bound: the p at which P(X >= k) = level."""
    if k == 0:
        return 0.0
    from care.conformal.bounds import _binom_cdf_le
    lo, hi = 0.0, k / n
    for _ in range(60):
        mid = (lo + hi) / 2
        if 1.0 - _binom_cdf_le(k - 1, n, mid) < level:   # P(X >= k) too small -> p larger
            lo = mid
        else:
            hi = mid
    return lo


def load_variant(base, level, draw=None):
    from bench.experiments import _mondrian_strata_fn
    from bench.study import load_and_propose
    ds = f"{base}_ni_inner{level}"
    log = (f"csvs/baran_{ds}_mapped.csv" if draw is None
           else f"csvs/baran_var_ni/{ds}/{draw}/baran_{ds}_mapped.csv")
    art, defects, repairs, _ = load_and_propose(
        ds, f"baran:{log}", detection="oracle", fast_verify=True)
    gold = {k: _norm(v) for k, v in defects.gold.items()}
    refs = [r for r in gold if r in repairs]
    ok = {r: _norm(repairs[r].proposed_value) == gold[r] for r in refs}
    return art, defects, repairs, refs, ok, _mondrian_strata_fn(art, defects)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sources", nargs="*", default=None,
                    help="table:level pairs, e.g. flights:10 hospital:70 (default: beers 30/70)")
    ap.add_argument("--draw", type=int, default=None,
                    help="Baran draw under csvs/baran_var_ni/<variant>/<draw>/ (default: beers' top-level logs)")
    ap.add_argument("--alphas", nargs="*", type=float, default=ALPHAS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    sources = ([(x.split(":")[0], int(x.split(":")[1])) for x in a.sources]
               if a.sources else SOURCES)
    from care.conformal.bounds import clopper_pearson_upper
    from care.conformal.rcps import rcps_threshold
    rows = []
    for base, SOURCE in sources:
        s_art, s_def, s_rep, s_refs, s_ok, sf = load_variant(base, SOURCE, a.draw)
        for tgt in TARGETS:
            t = ((s_art, s_def, s_rep, s_refs, s_ok, sf) if tgt == SOURCE
                 else load_variant(base, tgt, a.draw))
            _, _, t_rep, t_refs, t_ok, _ = t
            for alpha in a.alphas:
                pre_A, pre_R, post_A, post_R, labels = [], [], [], [], []
                pre_v = post_v = lb_v = 0
                lb_A = []
                for seed in range(SEEDS):
                    rng = random.Random(seed)
                    by = collections.defaultdict(list)
                    for r in s_refs:
                        by[sf(r)].append(r)
                    thr, cal = {}, set()
                    for st, rs in sorted(by.items()):
                        rs = rs[:]
                        rng.shuffle(rs)
                        c = rs[:int(0.4 * len(rs))]
                        cal.update(c)
                        lam, _ = rcps_threshold([s_rep[r].s_hat for r in c],
                                                [s_ok[r] for r in c], alpha=alpha,
                                                delta=DELTA, bound="exact")
                        thr[st] = lam
                    # deploy on the target; on the control arm, only on the held-out part
                    test = [r for r in t_refs if not (tgt == SOURCE and r in cal)]
                    written = [r for r in test
                               if t_rep[r].s_hat >= thr.get(sf(r), math.inf)]
                    # monitor: audit K written repairs per stratum
                    by_w = collections.defaultdict(list)
                    for r in written:
                        by_w[sf(r)].append(r)
                    kept, kept_lb, spent = [], [], 0
                    for st, rs in by_w.items():
                        k = min(K, len(rs))
                        audit = rng.sample(rs, k)
                        spent += k
                        wrong = sum(not t_ok[r] for r in audit)
                        if clopper_pearson_upper(wrong, k, DELTA) <= alpha:
                            kept += rs
                        if cp_lower(wrong, k, MON_LEVEL) <= alpha:
                            kept_lb += rs
                    lb_A.append(len(kept_lb) / max(1, len(test)))
                    e_lb = (1 - sum(t_ok[r] for r in kept_lb) / len(kept_lb)) if kept_lb else 0.0
                    lb_v += bool(kept_lb) and e_lb > alpha + 1e-12
                    for lst, A, R, is_pre in ((written, pre_A, pre_R, True),
                                              (kept, post_A, post_R, False)):
                        A.append(len(lst) / max(1, len(test)))
                        e = (1 - sum(t_ok[r] for r in lst) / len(lst)) if lst else 0.0
                        R.append(e)
                        if lst and e > alpha + 1e-12:
                            if is_pre:
                                pre_v += 1
                            else:
                                post_v += 1
                    labels.append(spent)
                row = {"base": base, "draw": a.draw or "", "calibrated_on": SOURCE, "deployed_on": tgt,
                       "alpha": alpha, "seeds": SEEDS,
                       "written_before": round(statistics.mean(pre_A), 4),
                       "error_before": round(statistics.mean(pre_R), 4),
                       "over_alpha_before": pre_v,
                       "written_after": round(statistics.mean(post_A), 4),
                       "error_after": round(statistics.mean(post_R), 4),
                       "over_alpha_after": post_v,
                       "audit_labels": round(statistics.mean(labels)),
                       "written_after_lowerbound_rule": round(statistics.mean(lb_A), 4),
                       "over_alpha_after_lowerbound_rule": lb_v}
                rows.append(row)
                print(f"  {base} {SOURCE}->{tgt} alpha={alpha}: written {row['written_before']:.3f}"
                      f" err {row['error_before']:.3f} over {pre_v}/{SEEDS}  | monitored: "
                      f"written {row['written_after']:.3f} err {row['error_after']:.3f} "
                      f"over {post_v}/{SEEDS}  audit {row['audit_labels']}  | lower-bound rule over "
                      f"{lb_v}/{SEEDS}", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

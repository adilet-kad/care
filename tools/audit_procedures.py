"""Alternative threshold procedures on the scored pairs: automation and alpha exceedances.

Procedures: Bonferroni over the 21-point grid with Clopper-Pearson (default), fixed
sequence from lambda=1 and from a pre-registered lambda_0=0.5, a single test for
constant-score cleaners, SGR (Geifman and El-Yaniv, 2017), Bonferroni with a Hoeffding
bound, and the default at delta/m for joint validity over the m strata.

    python tools/audit_procedures.py         -> experiments/procedures.csv
    python tools/audit_procedures.py --tax   # adds Baran/tax (~16 GB RAM)
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CARE_PROGRESS", "0")

# (cleaner, tag, dataset, own-detector log or None); Baran uses its shipped draw.
PAIRS = [("Baran", "baran", d, None) for d in ("hospital", "beers", "rayyan", "tax")] + \
        [("BClean", "bclean", d, None) for d in ("hospital", "beers", "flights")] + \
        [("BClean+", "bclean_plus", d, None) for d in ("hospital", "beers", "flights")] + \
        [("GIDCL", "gidcl", d, f"csvs/gidcl_{d}_detected.csv")
         for d in ("hospital", "beers", "flights", "rayyan", "tax")] + \
        [("HoloClean", "holoclean", d, None) for d in ("hospital", "flights")] + \
        [("ZeroEC", "zeroec", d, None) for d in ("hospital", "beers", "flights", "rayyan")]
PROCS = {"bonferroni_cp": dict(procedure="bonferroni", bound="exact"),
         "single_test_cp": dict(procedure="bonferroni", bound="exact"),
         "fixed_seq_cp": dict(procedure="fixed_sequence", bound="exact"),
         "fixed_seq_cp_l0": dict(procedure="fixed_sequence", bound="exact"),
         "sgr_cp": dict(procedure="sgr", bound="exact"),
         "bonferroni_hoeff": dict(procedure="bonferroni", bound="hoeffding"),
         "bonferroni_cp_joint": dict(procedure="bonferroni", bound="exact")}
L0_GRID = [i / 20 for i in range(11)]      # fixed sequence started at lambda_0 = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--tax", action="store_true", help="include Baran/tax")
    ap.add_argument("--only", nargs="*", default=None, help="cleaner/dataset filters")
    ap.add_argument("--out", default="experiments/procedures.csv")
    a = ap.parse_args()
    from bench.experiments import _detected_truth, _mondrian_strata_fn
    from bench.study import evaluate_split, load_and_propose

    rows, srows = [], []
    for name, tag, ds, det in PAIRS:
        if ds == "tax" and name == "Baran" and not a.tax:
            continue
        if a.only and f"{name}/{ds}" not in a.only:
            continue
        log = f"csvs/{tag}_{ds}_mapped.csv"
        if not os.path.exists(log):
            print(f"  [skip] {name}/{ds}: no {log}")
            continue
        art, defects, repairs, _ = load_and_propose(
            ds, f"{tag}:{log}", detection=(f"log:{det}" if det else "oracle"), fast_verify=True)
        truth = _detected_truth(art, defects) if det else defects.gold
        gold = {k: (str(v).lower().strip() if v is not None else v) for k, v in truth.items()}
        sf = _mondrian_strata_fn(art, defects)
        # constant-score cleaners (Baran, GIDCL): the grid collapses to that one score
        svals = {r.s_hat for r in repairs.values()}
        const_grid = sorted(svals) if len(svals) == 1 else None
        n_strata = len({sf(r) for r in repairs.values()})
        for alpha in a.alphas:
            for pname, kw in PROCS.items():
                autos, viol, st_n, st_v, gaps = [], 0, 0, 0, []
                for s in range(a.seeds):
                    g = const_grid if (const_grid and pname in ("fixed_seq_cp", "fixed_seq_cp_l0", "single_test_cp")) else None
                    if pname == "fixed_seq_cp_l0" and g is None:
                        g = L0_GRID
                    d = a.delta / n_strata if pname == "bonferroni_cp_joint" else a.delta
                    sc, dec = evaluate_split(repairs, truth, alpha=alpha, delta=d, seed=s,
                                             strata_fn=sf, return_decisions=True, grid=g, **kw)
                    c = sc["CARE"]
                    auto = 1.0 - c.human_cost
                    autos.append(auto)
                    if dec["auto"]:
                        viol += c.realized_error > alpha + 1e-12
                        gaps.append(alpha - c.realized_error)
                    per = collections.defaultdict(lambda: [0, 0])
                    for ref, v in dec["auto"].items():
                        k = sf(repairs[ref])
                        per[k][0] += 1
                        per[k][1] += (v != gold[ref])
                    for k, (n, w) in per.items():
                        st_n += 1
                        st_v += (w / n) > alpha + 1e-12
                row = {"cleaner": name, "dataset": ds, "alpha": alpha, "procedure": pname,
                       "seeds": a.seeds, "automation": round(statistics.mean(autos), 4),
                       "split_violations": viol,
                       "written_strata": st_n, "stratum_violations": st_v,
                       "mean_slack": round(statistics.mean(gaps), 4) if gaps else ""}
                rows.append(row)
                print(f"  {name:9s} {ds:8s} a={alpha:<4} {pname:16s} A={row['automation']:.3f} "
                      f"viol={viol}/{a.seeds} strata {st_v}/{st_n} slack={row['mean_slack']}",
                      flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)")
    for p in PROCS:
        sub = [r for r in rows if r["procedure"] == p]
        print(f"  {p:16s} mean A={statistics.mean(r['automation'] for r in sub):.3f}  "
              f"split viol={sum(r['split_violations'] for r in sub)}/"
              f"{sum(r['seeds'] for r in sub)}  stratum viol="
              f"{sum(r['stratum_violations'] for r in sub)}/{sum(r['written_strata'] for r in sub)}")


if __name__ == "__main__":
    main()

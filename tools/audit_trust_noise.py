"""Mislabelled source trust: automation, realised error and contamination under noise rho.

10% of repairs are poisoned through an untrusted source; with probability rho a poisoned
repair is labelled trusted and a clean repair untrusted (``random``, or only correct ones
for ``correlated``). Exceedances are counted against alpha and alpha + eps_S(1 - alpha),
with the gate applied before (gate-first) or after (gate-after) calibration.

    python tools/audit_trust_noise.py   -> experiments/trust_noise.csv
"""
from __future__ import annotations

import csv
import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CARE_PROGRESS", "0")

CONFIGS = [("baran", "hospital"), ("baran", "beers"), ("baran", "rayyan"),
           ("bclean", "hospital"), ("bclean", "flights")]
RHOS = [0.0, 0.05, 0.1, 0.2, 0.4]
ALPHA, DELTA, POISON, SEEDS = 0.2, 0.1, 0.1, 20
OUT = "experiments/trust_noise.csv"


def main():
    from bench.experiments import _mondrian_strata_fn
    from bench.study import evaluate_split, load_and_propose

    rows = []
    for tag, ds in CONFIGS:
        art, defects, repairs, _ = load_and_propose(
            ds, f"{tag}:csvs/{tag}_{ds}_mapped.csv", detection="oracle", fast_verify=True)
        sf = _mondrian_strata_fn(art, defects)
        print(f"\n== {tag}/{ds}", flush=True)
        for mode in ("random", "correlated"):
            for rho in RHOS:
                for gate_first in (True, False):
                    A, R, E, over_a, over_b = [], [], [], 0, 0
                    for s in range(SEEDS):
                        sc = evaluate_split(
                            repairs, defects.gold, alpha=ALPHA, delta=DELTA, seed=s,
                            strata_fn=sf, poison_frac=POISON, trust_noise=rho,
                            trust_noise_mode=mode, gate_calibration=gate_first)["CARE"]
                        a = 1.0 - sc.human_cost
                        A.append(a)
                        R.append(sc.realized_error)
                        E.append(sc.applied_contamination)
                        if a > 0:
                            over_a += sc.realized_error > ALPHA + 1e-12
                            over_b += sc.realized_error > \
                                ALPHA + sc.applied_contamination * (1 - ALPHA) + 1e-12
                    row = {"cleaner": tag, "dataset": ds, "mode": mode, "rho": rho,
                           "order": "gate-first" if gate_first else "gate-after",
                           "automation": round(statistics.mean(A), 4),
                           "realized_error": round(statistics.mean(R), 4),
                           "worst_error": round(max(R), 4),
                           "eps_S": round(statistics.mean(E), 4),
                           "over_alpha": over_a, "over_thm2_bound": over_b, "seeds": SEEDS}
                    rows.append(row)
                    print(f"  {mode:>10} rho={rho:<4} {row['order']:>10}: A={row['automation']:.3f}"
                          f" R={row['realized_error']:.3f} (worst {row['worst_error']:.3f})"
                          f" eps={row['eps_S']:.3f}  >alpha {over_a}/{SEEDS}"
                          f"  >bound {over_b}/{SEEDS}", flush=True)
    os.makedirs("experiments", exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

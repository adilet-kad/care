"""Check that gating the calibration sample (gate-first) gives the same scores as gate-after.

Compares every score field of evaluate_split under both orders, on the poison
configurations and the oracle pareto configuration of every scored pair.

    python tools/verify_gate_first.py   # 10 seeds; tax skipped unless --tax
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CARE_PROGRESS", "0")

POISON = [("baran", "hospital"), ("baran", "beers"), ("baran", "rayyan"),
          ("bclean", "hospital"), ("bclean", "flights")]
PARETO = [("baran", d) for d in ("hospital", "beers", "flights", "rayyan", "tax")] + \
         [("bclean", d) for d in ("hospital", "beers", "flights", "rayyan")] + \
         [("bclean_plus", d) for d in ("hospital", "beers", "flights")] + \
         [("gidcl", d) for d in ("hospital", "beers", "flights", "rayyan", "tax")] + \
         [("holoclean", d) for d in ("hospital", "flights")]


def _fields(sc):
    return {k: tuple(sorted(vars(v).items())) for k, v in sc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tax", action="store_true")
    a = ap.parse_args()
    from bench.experiments import _mondrian_strata_fn
    from bench.study import evaluate_split, load_and_propose
    from care.conformal import marginal_strata

    checked = mism = 0
    for tag, ds in PARETO + [p for p in POISON if p not in PARETO]:
        if ds == "tax" and not a.tax:
            continue
        log = f"csvs/{tag}_{ds}_mapped.csv"
        if not os.path.exists(log):
            print(f"  [skip] {tag}/{ds}")
            continue
        art, defects, repairs, _ = load_and_propose(ds, f"{tag}:{log}", detection="oracle",
                                                    fast_verify=True)
        gold = defects.gold
        sf = _mondrian_strata_fn(art, defects)
        poison = (tag, ds) in POISON
        cfgs = [dict(strata_fn=f, alpha=al) for f in (marginal_strata, sf)
                for al in (0.05, 0.1, 0.2)]
        if poison:
            cfgs += [dict(strata_fn=sf, alpha=0.2, poison_frac=pf, trusted_poison=tp)
                     for pf in (0.05, 0.1, 0.2) for tp in (False, True)]
        n0 = mism
        for cfg in cfgs:
            for s in range(a.seeds):
                x = evaluate_split(repairs, gold, delta=0.1, seed=s, gate_calibration=False, **cfg)
                y = evaluate_split(repairs, gold, delta=0.1, seed=s, gate_calibration=True, **cfg)
                checked += 1
                if _fields(x) != _fields(y):
                    mism += 1
        print(f"  {tag}/{ds}: {len(cfgs) * a.seeds} splits, "
              f"{'IDENTICAL' if mism == n0 else f'{mism - n0} DIFFER'}", flush=True)
    print(f"\n{checked} splits compared, {mism} differ")
    return 1 if mism else 0


if __name__ == "__main__":
    raise SystemExit(main())

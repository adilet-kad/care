"""Run the certification sweeps for one dataset and write result CSVs.

    python -m bench.run_study --dataset hospital --backends baran:csvs/baran_hospital_mapped.csv
    python -m bench.run_study --dataset hospital \
        --backends baran_raha:csvs/baran_raha_hospital_mapped.csv \
        --detection log:csvs/raha_hospital_detected.csv --scoring detected
"""

from __future__ import annotations

import argparse
import csv
import os

from bench.experiments import pareto, poison_robustness
from bench.study import OFFLINE_LOG_BACKENDS, is_offline_log_backend


def _write(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def _slug(x: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in x).strip("-")[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backends", nargs="+", required=True, metavar="TAG:LOG",
                    help="one or more offline-log proposers as '<tag>:<log.csv>' "
                         f"(tag in {OFFLINE_LOG_BACKENDS}, or a '<tag>_<variant>' of "
                         "one); one CSV per backend when several are given")
    ap.add_argument("--experiment", default="all",
                    choices=["pareto", "poison", "all"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--detection", default="oracle",
                    help="'oracle', 'constraints', or 'log:<path>' to use a real "
                         "detector's exported row_id,column CSV (e.g. Raha) as the "
                         "work queue")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--row-range", type=int, nargs=2, metavar=("LO", "HI"), default=None,
                    help="restrict the queue to rows LO <= row_id < HI, for a cleaner run "
                         "on a row shard; set it from the shard bounds, not from the log")
    ap.add_argument("--scoring", default="errors", choices=["errors", "detected"],
                    help="'errors': score true error cells only; 'detected': score every "
                         "cell the cleaner wrote, so overwriting a correct cell counts")
    ap.add_argument("--poison-fracs", type=float, nargs="+", default=[0.0, 0.2, 0.4],
                    help="poisoned fractions for the poison experiment")
    ap.add_argument("--trusted-poison", action="store_true",
                    help="attribute poison to a trusted source (undetectable corruption); "
                         "default is an untrusted source")
    ap.add_argument("--fast-verify", default="auto", choices=["auto", "on", "off"],
                    help="incremental verification: 'auto' above 200K cells, 'on' always "
                         "(self-checked against the exact path), 'off' exact path only")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    backends = args.backends
    for b in backends:
        if not is_offline_log_backend(b):
            ap.error(f"--backends entry {b!r} is not a '<tag>:<log>' spec with a tag in "
                     f"{OFFLINE_LOG_BACKENDS}")
    sweep = len(backends) > 1
    kw = dict(delta=args.delta, seeds=args.seeds, detection=args.detection,
              max_cells=args.max_cells,
              row_range=tuple(args.row_range) if args.row_range else None,
              fast_verify={'auto': 'auto', 'on': True, 'off': False}[args.fast_verify])
    ds, out = args.dataset, args.out
    a1 = args.alphas[1] if len(args.alphas) > 1 else args.alphas[0]
    want = ["pareto", "poison"] if args.experiment == "all" else [args.experiment]

    for backend in backends:
        if sweep:
            print(f"\n===== backend: {backend} =====")
        client = backend
        sfx = f"_{_slug(backend)}" if sweep else ""

        def tag(rows):
            for r in rows:
                r["backend"] = backend
            return rows

        if "pareto" in want:
            print("[pareto]")
            rows = tag(pareto(ds, client, alphas=tuple(args.alphas),
                              scoring=args.scoring, **kw))
            for r in rows:
                print(f"  a={r['alpha']} {r['baseline']:8} {r.get('strata','-'):9} "
                      f"human_cost={r['human_cost']:.2f} realized_err={r['realized_error']:.3f} "
                      f"F1={r['repair_f1']:.2f} covered={r.get('covered','')} front={r['on_pareto_front']}")
            _write(rows, f"{out}/{ds}{sfx}_pareto.csv")
        if "poison" in want:
            mode = "trusted (undetectable)" if args.trusted_poison else "untrusted (detectable)"
            print(f"[poison_robustness]  mode: {mode}")
            rows = tag(poison_robustness(ds, client, alpha=a1, trusted_poison=args.trusted_poison,
                                         poison_fracs=tuple(args.poison_fracs),
                                         scoring=args.scoring, **kw))
            # warn when nothing is auto-applied at poison=0
            _base = [r for r in rows if r["baseline"] == "CARE" and r["poison_frac"] == 0.0]
            if _base and float(_base[0]["human_cost"]) >= 0.999:
                print("  [poison] warning: nothing is auto-applied at poison=0, so the gate "
                      "is never exercised and eps_S=0 is trivial; choose a --scoring/"
                      "--detection under which this cleaner certifies")
            for r in rows:
                extra = ""
                if r["baseline"] == "CARE":
                    extra = (f" eps_S={r['applied_contamination']:.3f} "
                             f"thm2_bound={r['thm2_bound']:.3f} ok={r['within_thm2_bound']}")
                print(f"  {r['baseline']:8} poison={r['poison_frac']:.3g} "
                      f"realized_err={r['realized_error']:.3f} [{r['err_lo']:.3f},{r['err_hi']:.3f}]{extra}")
            sfx2 = sfx + ("_trustedpoison" if args.trusted_poison else "")
            _write(rows, f"{out}/{ds}{sfx2}_poison.csv")


if __name__ == "__main__":
    main()
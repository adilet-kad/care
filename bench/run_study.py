"""bench.run_study -- CLI entrypoint for the real-world empirical study.

A run replays an external cleaner's offline repair log (``<tag>:<path>``, tag in
``bench.study.OFFLINE_LOG_BACKENDS``) through CARE's governance layer and writes
one CSV per experiment. The invocation reproduce.sh uses for the headline table:

  python -m bench.run_study --dataset hospital \
      --backends "baran:csvs/baran_hospital_mapped.csv" \
      --experiment all --alphas 0.05 0.1 0.2 --seeds 20 --scoring errors \
      --out experiments/results_baran_errors

  # deployment-realistic scoring under a real detector's exported cell log:
  python -m bench.run_study --dataset hospital \
      --backends "baran_raha:csvs/baran_raha_hospital_mapped.csv" \
      --detection log:csvs/raha_hospital_detected.csv --scoring detected \
      --out experiments/results_raha
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
                    help="restrict the work queue to rows LO <= row_id < HI. Use this "
                         "when a proposer was run on a ROW SHARD of a large table "
                         "(e.g. Jellyfish on tax rows 0-25000): without it the "
                         "un-sharded rows have no proposal and are silently counted as "
                         "escalations, which inflates human cost toward 1.0. "
                         "IMPORTANT: the range is part of the experimental design and "
                         "must be set from the shard bounds you actually ran -- never "
                         "from what the log happens to contain. Deriving the population "
                         "from the log would let a proposer choose its own evaluation "
                         "set (propose only on easy cells, look excellent). A row shard "
                         "is chosen a priori and is therefore safe; a difficulty-biased "
                         "subset is not.")
    ap.add_argument("--scoring", default="errors", choices=["errors", "detected"],
                    help="'errors' (default): score only true error cells -- the "
                         "standard repair-F1 population, what oracle detection means. "
                         "'detected': score EVERY proposed cell against a full truth "
                         "map, so preserving a clean cell counts and corrupting one is "
                         "penalised -- the deployment-realistic measure, and the only "
                         "one under which --detection actually affects scoring.")
    ap.add_argument("--poison-fracs", type=float, nargs="+", default=[0.0, 0.2, 0.4],
                    help="poison fractions to sweep in the poison experiment "
                         "(finer values e.g. 0.02 0.05 0.1 catch the eps_S>0 / "
                         "Theorem-2 degradation regime that 0.2/0.4 often escalate past).")
    ap.add_argument("--trusted-poison", action="store_true",
                    help="attribute poison to a HIGH-trust source so the escalation "
                         "gate cannot catch it (Theorem 2 / graceful-degradation "
                         "exhibit); default is low-trust poison (Theorem 1).")
    ap.add_argument("--fast-verify", default="auto", choices=["auto", "on", "off"],
                    help="constraint-verification fast path. 'auto' (default) engages "
                         "only above 200K cells. 'on' forces it on smaller tables, "
                         "which is safe BECAUSE IT SELF-CHECKS: it probes cells with "
                         "the exact verifier, confirms feasible/s_margin/delta are "
                         "constant under single-cell repair for this constraint set, "
                         "and falls back to the exact path with a warning if they are "
                         "not. Verified bit-identical on hospital (deviation D11). Use "
                         "'off' to re-derive that check.")
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
            mode = "trusted/undetectable (Thm 2)" if args.trusted_poison else "low-trust/detectable (Thm 1)"
            print(f"[poison_robustness]  mode: {mode}")
            rows = tag(poison_robustness(ds, client, alpha=a1, trusted_poison=args.trusted_poison,
                                         poison_fracs=tuple(args.poison_fracs),
                                         scoring=args.scoring, **kw))
            # VACUITY GUARD: a poison experiment on an empty auto-applied set proves
            # nothing -- eps_S is 0 at every level because nothing was ever applied.
            _base = [r for r in rows if r["baseline"] == "CARE" and r["poison_frac"] == 0.0]
            if _base and float(_base[0]["human_cost"]) >= 0.999:
                print("  [poison] !! VACUOUS RUN: human_cost=1.00 at poison=0, so nothing "
                      "is auto-applied and the gate is never exercised. eps_S=0 here is "
                      "trivial, NOT evidence for Theorem 1. Re-run with a --scoring / "
                      "--detection combination under which this proposer actually "
                      "certifies (check the pareto run first).")
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
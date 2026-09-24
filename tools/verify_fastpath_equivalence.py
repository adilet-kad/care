"""Prove the constraint-verification fast path is EXACT, not an approximation.

Section 4.1 of the paper claims that skipping per-candidate verification is safe under a
machine-checked precondition. This script is the evidence: it runs the same cells through
both the exact path and the fast path and asserts the outputs are bit-identical.

Why this exists as a standalone check rather than a unit test: the exact path costs
~85 ms per candidate on a 19K-cell table and ~25 s on a 3M-cell one, so running both to
completion is far too slow for CI. It is instead run deliberately, and its result is
recorded as deviation D11 in docs/REPRODUCE.md.

    python tools/verify_fastpath_equivalence.py            # from the repo root
    -> expects: IDENTICAL: True (differing cells: 0)

A non-zero difference means the fast path's precondition does not hold for the current
constraint set and every number produced with `--fast-verify on` must be re-derived with
`--fast-verify off`.
"""

from __future__ import annotations

import os
import sys

# Runnable from anywhere. bench.datasets resolves data/ relatively and will try to
# DOWNLOAD a dataset it cannot find locally, so chdir as well as extending sys.path --
# otherwise running this from another directory silently hits the network.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

os.environ["CARE_PROGRESS"] = "0"          # suppress progress bars; this is a diff, not a run

from bench.baran_proposer import BaranProposer          # noqa: E402
from bench.study import _registry, load, propose_repairs  # noqa: E402

DATASET = "hospital"     # small enough that the exact path finishes in seconds
LOG = "csvs/baran_hospital_mapped.csv"
N_CELLS = 150


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log = os.path.join(root, LOG)
    if not os.path.exists(log):
        raise SystemExit(f"missing proposer log: {log}\n"
                         f"Regenerate it first (docs/REPRODUCE.md section 3).")

    art, defects, _ = load(DATASET)
    reg = _registry()
    refs = list(defects.gold)[:N_CELLS]
    proposer = BaranProposer(log, source_id="baran")

    # max_workers=1 so any difference is attributable to the verification path rather
    # than to scheduling.
    slow = propose_repairs(art, refs, proposer, reg=reg, max_workers=1, fast_verify=False)
    fast = propose_repairs(art, refs, proposer, reg=reg, max_workers=1, fast_verify=True)

    differing = [
        r for r in set(slow) | set(fast)
        if r not in slow or r not in fast
        or (slow[r].proposed_value, slow[r].s_hat) != (fast[r].proposed_value, fast[r].s_hat)
    ]
    identical = not differing
    print(f"cells compared: {len(refs)}   repairs: exact={len(slow)} fast={len(fast)}")
    print(f"IDENTICAL: {identical} (differing cells: {len(differing)})")
    if differing:
        for r in differing[:5]:
            print(f"  {r}: exact={slow.get(r)} fast={fast.get(r)}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())

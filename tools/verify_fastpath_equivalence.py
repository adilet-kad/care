"""Check that the fast verification path gives the same repairs as the exact path.

Runs the same hospital cells through both paths (too slow for CI on large tables).

    python tools/verify_fastpath_equivalence.py   -> IDENTICAL: True (differing cells: 0)
"""

from __future__ import annotations

import os
import sys

# bench.datasets resolves data/ relative to the cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

os.environ["CARE_PROGRESS"] = "0"

from bench.log_proposer import LogProposer          # noqa: E402
from bench.study import _registry, load, propose_repairs  # noqa: E402

DATASET = "hospital"
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
    proposer = LogProposer(log, source_id="baran")

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

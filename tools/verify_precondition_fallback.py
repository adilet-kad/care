"""Check that the verification fast path detects when its constancy precondition fails.

Builds a 3-row table where the FD ZipCode -> City holds and one candidate breaks it,
then checks that the probe detects non-constancy and that the fast path matches the
exact path. Exit status 1 on failure.

    python tools/verify_precondition_fallback.py
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def build_registry(with_fd: bool):
    """The benchmark's registry, optionally with the cross-row FD predicate enabled."""
    from care.core import (ConstraintRegistry, DQRequirements, clear_registry,
                           register_constraint)
    from care.predicates import ALL_CONSTRAINTS

    clear_registry()
    for c in ALL_CONSTRAINTS:
        register_constraint(c)
    enabled = {"dq.completeness.floor"}
    if with_fd:
        enabled.add("dq.consistency.fd")
    return ConstraintRegistry.from_requirements(DQRequirements(enabled=enabled),
                                                strict=True)


def boundary_artifact():
    """Smallest table where the FD ZipCode -> City holds and one repair can break it."""
    from care.core import Cell, DataArtifact

    art = DataArtifact(artifact_id="boundary")
    for rid, zipc, city in (("r1", "35233", "birmingham"),
                            ("r2", "35233", "birmingham"),
                            ("r3", "35056", "cullman")):
        art.set_cell(Cell(row_id=rid, col="ZipCode", value=zipc,
                          is_required=True, dtype="string"))
        art.set_cell(Cell(row_id=rid, col="City", value=city,
                          is_required=True, dtype="string"))
    return art


class TwoValueProposer:
    """Proposes a fixed value per ref, so candidates can verify differently."""

    def __init__(self, values):
        self.values = values

    def propose(self, v, art=None, *, options=None):
        from care.core.repair import RepairCandidate
        out = []
        for ref in v.refs:
            if ref in self.values:
                out.append(RepairCandidate(
                    target_ref=ref, proposed_value=self.values[ref],
                    rationale="fixture", evidence=["reference"],
                    s_conf=0.9, s_agree=0.9, s_consistency=0.9))
        return out


def main() -> int:
    import contextlib as _c
    import io as _io

    from bench.study import propose_repairs

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    os.chdir(ROOT)

    art = boundary_artifact()
    # options are keyed by constraint id (see verify.projector._params)
    opts = {"dq.consistency.fd": {"fds": [(["ZipCode"], ["City"])]}}
    values = {"r2::City": "birmingham",       # preserves ZipCode -> City
              "r3::City": "montgomery",       # r3 has its own zip
              "r1::City": "huntsville"}       # breaks it: r1 and r2 share 35233
    refs = ["r2::City", "r1::City", "r3::City"]
    proposer = TwoValueProposer(values)

    print("constructed artifact: 3 rows, FD ZipCode -> City currently satisfied")
    print("candidates: r2->birmingham (preserves), r1->huntsville (breaks), "
          "r3->montgomery (independent)")

    from bench.study import _probe_verifier_constant
    from care.verify.projector import hard_violation_set
    from care.verify.verifier import Verifier

    # The sweep uses incremental verification for the floor and FDs and the constancy
    # probe otherwise; both are checked.

    # 1. floor only: the precondition holds
    reg_floor = build_registry(with_fd=False)
    buf = _io.StringIO()
    with _c.redirect_stdout(buf):
        held, _ = _probe_verifier_constant(
            art, refs, proposer, Verifier(), reg_floor, {},
            hard_violation_set(art, reg_floor, {}), n=len(refs))
    print(f"\n[1] completeness floor only -> probe precondition "
          f"{'HOLDS' if held else 'FAILED (unexpected)'}")

    # 2. with the FD: the probe must detect non-constancy
    reg_fd = build_registry(with_fd=True)
    buf = _io.StringIO()
    with _c.redirect_stdout(buf):
        const_fd, _ = _probe_verifier_constant(
            art, refs, proposer, Verifier(), reg_fd, opts,
            hard_violation_set(art, reg_fd, opts), n=len(refs))
    detected = not const_fd
    with _c.redirect_stdout(buf):
        fast = propose_repairs(art, refs, proposer, reg=reg_fd, options=opts,
                               fast_verify=True, max_workers=1)
    incremental = "incremental verification" in buf.getvalue()
    print(f"[2] + cross-row FD          -> probe detects non-constancy: {detected}; "
          f"sweep uses incremental verification: {incremental}")

    if not (held and detected and incremental):
        print("\nFAIL: the probe accepted a constraint set whose verification is not "
              "constant.\nThe fast path would have reused a wrong constant for every "
              "remaining cell.")
        return 1

    # 3. the fast path must equal the exact path
    buf = _io.StringIO()
    with _c.redirect_stdout(buf):
        exact = propose_repairs(art, refs, proposer, reg=reg_fd, options=opts,
                                fast_verify=False, max_workers=1)

    diffs = []
    for k in sorted(set(fast) | set(exact)):
        a, b = fast.get(k), exact.get(k)
        if (a is None) != (b is None):
            diffs.append((k, "present in one path only"))
            continue
        if a is None:
            continue
        for f in ("proposed_value", "feasible", "s_margin", "s_hat", "delta_objective"):
            x, y = getattr(a, f, None), getattr(b, f, None)
            if isinstance(x, float) and isinstance(y, float):
                if abs(x - y) > 1e-9:
                    diffs.append((k, f"{f}: {x} vs {y}"))
            elif x != y:
                diffs.append((k, f"{f}: {x!r} vs {y!r}"))

    print(f"[3] fast path vs exact      -> {len(fast)} and {len(exact)} repairs, "
          f"{len(diffs)} differing fields")
    for k, why in diffs[:5]:
        print(f"      {k}: {why}")

    ok = not diffs
    print("\n" + ("PASS: the probe detects its own precondition failing, and under the "
                  "FD the\nincremental path reproduces the exact path exactly."
                  if ok else "FAIL: the fast path differs from the exact path."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

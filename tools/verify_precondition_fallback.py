"""Demonstrate that the constant-time verification path detects its own precondition.

Section 4.1 claims the fast path is safe because it does not *assume* verification is
constant across candidates -- it probes k cells on the exact path, checks that
`feasible`, `s_margin` and `delta_objective` agree, and falls back to the exact path
when they do not. `verify_fastpath_equivalence.py` shows the two paths agree when the
precondition HOLDS. Nothing showed what happens when it fails, which leaves the
weakest engineering claim in the paper resting on a code comment.

This closes that. It constructs a constraint set the precondition genuinely fails
under, runs both paths, and checks three things:

  1. the probe DETECTS the non-constancy rather than proceeding,
  2. the run falls back to the exact path and says so,
  3. the output is identical to running the exact path directly.

Why the artifact is constructed rather than borrowed. The benchmark's default hard
constraint is the completeness floor, and no single-cell value repair can flip a floor
already violated by thousands of nulls -- which is exactly why the precondition holds
there and the fast path is sound. Enabling the cross-row FD predicate on hospital does
not help either: its `ZipCode -> City` dependency is already violated on 1,122 cells,
so one repair cannot move the aggregate verdict and verification stays constant for
the same reason.

Non-constancy needs a constraint set sitting at its boundary: an FD that currently
HOLDS, where one candidate value preserves it and another breaks it. That is a
property of the data, not of the dataset's size, so the honest demonstration builds
the smallest artifact with that property rather than hunting for a real table that
happens to have it. The mechanism under test -- probe, detect, fall back -- is the
same either way.

    python tools/verify_precondition_fallback.py

Exit status is 1 if the probe fails to detect non-constancy, or if the two paths
disagree. Either would be a real defect in the claim.
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
    """Smallest table where an FD holds and one repair can break it.

    Two rows share a ZipCode and agree on City, so `ZipCode -> City` is satisfied.
    Repairing r2's City to match r1 keeps it satisfied; repairing it to anything else
    violates it. Verification is therefore NOT constant across candidates, which is
    precisely the precondition the fast path must not assume.
    """
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
    """Offers an FD-preserving value for one ref and an FD-breaking value for another.

    Deliberately minimal: the point is that the two candidates verify differently, so
    a probe that samples only one of them would generalise a wrong constant.
    """

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
                    s_llm=0.9, s_agree=0.9, s_consistency=0.9))
        return out


def main() -> int:
    import contextlib as _c
    import io as _io

    from bench.study import propose_repairs

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    os.chdir(ROOT)

    art = boundary_artifact()
    # Options are keyed BY CONSTRAINT ID (see verify.projector._params); a flat
    # {"fds": ...} silently reaches no predicate and the FD is never evaluated.
    opts = {"dq.consistency.fd": {"fds": [(["ZipCode"], ["City"])]}}
    # r2 keeps the FD satisfied; r3 breaks it by disagreeing with nothing, so we make
    # r3 share r1's zip in the candidate space by repairing its City to a value that
    # conflicts with the r1/r2 group once its zip is corrected.
    values = {"r2::City": "birmingham",       # preserves ZipCode -> City
              "r3::City": "montgomery",       # r3 has its own zip: still fine
              "r1::City": "huntsville"}       # BREAKS it: r1 and r2 share 35233
    refs = ["r2::City", "r1::City", "r3::City"]
    proposer = TwoValueProposer(values)

    print("constructed artifact: 3 rows, FD ZipCode -> City currently SATISFIED")
    print("candidates: r2->birmingham (preserves), r1->huntsville (breaks), "
          "r3->montgomery (independent)")

    # ---- 1. floor only: the precondition holds, as in the paper's configuration --
    reg_floor = build_registry(with_fd=False)
    buf = _io.StringIO()
    with _c.redirect_stdout(buf):
        propose_repairs(art, refs, proposer, reg=reg_floor, options={},
                        fast_verify=True, max_workers=1)
    held = "CONSTANT for this constraint set" in buf.getvalue()
    print(f"\n[1] completeness floor only -> precondition "
          f"{'HOLDS, fast path used' if held else 'FAILED (unexpected)'}")

    # ---- 2. with the FD: the precondition fails and must be caught ---------------
    reg_fd = build_registry(with_fd=True)
    buf = _io.StringIO()
    with _c.redirect_stdout(buf):
        fast = propose_repairs(art, refs, proposer, reg=reg_fd, options=opts,
                               fast_verify=True, max_workers=1)
    out = buf.getvalue()
    detected = "NON-constant verification" in out
    fell_back = "falling back to" in out
    print(f"[2] + cross-row FD          -> non-constancy detected: {detected}, "
          f"fell back: {fell_back}")

    if not (detected and fell_back):
        print("\nFAIL: the probe accepted a constraint set whose verification is not "
              "constant.\nThe fast path would have reused a wrong constant for every "
              "remaining cell.")
        return 1

    # ---- 3. the fallback must equal the exact path -------------------------------
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

    print(f"[3] fallback vs exact       -> {len(fast)} and {len(exact)} repairs, "
          f"{len(diffs)} differing fields")
    for k, why in diffs[:5]:
        print(f"      {k}: {why}")

    ok = not diffs
    print("\n" + ("PASS: the precondition is verified at run time, its failure is "
                  "detected, and the\nfallback reproduces the exact path exactly."
                  if ok else "FAIL: fallback differs from the exact path."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

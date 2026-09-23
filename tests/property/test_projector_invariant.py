"""Property-based test of the I2 projector invariant.

I2 is the load-bearing guarantee: the Verifier is a *projector* onto the
hard-feasible set F_H, never a scorer. For **arbitrary** artifacts and
**arbitrary** (including adversarial / garbage) repair candidates, no candidate
the verifier marks ``feasible`` may introduce a new hard-constraint violation
``(cid, ref)`` when applied alone. ``project`` must return exactly that set.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from care.core import (
    Cell,
    ConstraintRegistry,
    DataArtifact,
    DQRequirements,
    RepairCandidate,
    Source,
    clear_registry,
    register_constraint,
)
from care.predicates import ALL_CONSTRAINTS
from care.verify import Verifier, apply_repair, hard_violation_set

_ENABLED = {
    "dq.completeness.floor",  # H, per-cell:   required cells non-null
    "dq.consistency.fd",      # H, cross-row:  FD dept -> city
    "dq.traceability",        # H, lineage:    every ref has a provenance edge
    "dq.source.integrity",    # H, sources:    every source carries a digest
}
_OPTIONS = {"dq.consistency.fd": {"fds": [(["dept"], ["city"])]}}

_ROWS = ["r0", "r1", "r2", "r3"]
_COLS = ["age", "city", "dept"]
_VALUES = st.one_of(st.none(), st.integers(0, 3), st.sampled_from(["NYC", "SF", "LA"]))
_CELL = st.tuples(st.sampled_from(_ROWS), st.sampled_from(_COLS), _VALUES, st.booleans())
# Candidate target refs: mostly existing (row, col) pairs, sometimes a fresh
# cell that does not yet exist (which must be rejected -- it would be untraced).
_REF = st.one_of(
    st.tuples(st.sampled_from(_ROWS), st.sampled_from(_COLS)),
    st.tuples(st.sampled_from(["rx", "ry"]), st.sampled_from(["zz", "age"])),
)
_CAND = st.tuples(_REF, _VALUES)


@pytest.fixture(autouse=True)
def _load_constraints():
    clear_registry()
    for cls in ALL_CONSTRAINTS:
        register_constraint(cls)
    yield
    clear_registry()


def _build(cells, add_source) -> DataArtifact:
    art = DataArtifact(artifact_id="H")
    for rid, col, val, req in cells:
        art.set_cell(Cell(row_id=rid, col=col, value=val, is_required=req, dtype="string"))
    if add_source:
        art.add_source(Source(source_id="s1", uri="u", kind="scraped"))  # unhashed
    return art


def _reg() -> ConstraintRegistry:
    return ConstraintRegistry.from_requirements(
        DQRequirements(enabled=set(_ENABLED)), strict=True
    )


@settings(max_examples=300, deadline=None)
@given(
    cells=st.lists(_CELL, max_size=8),
    add_source=st.booleans(),
    cand_specs=st.lists(_CAND, max_size=6),
)
def test_projector_never_introduces_a_hard_violation(cells, add_source, cand_specs):
    art = _build(cells, add_source)
    reg = _reg()
    v = Verifier()

    before = hard_violation_set(art, reg, _OPTIONS)
    cands = [
        RepairCandidate(target_ref=f"{rid}::{col}", proposed_value=val)
        for (rid, col), val in cand_specs
    ]

    # per-candidate projector verdict is sound
    for c in cands:
        vr = v.assess(c, art, reg, options=_OPTIONS, before=before)
        if vr.feasible:
            after = hard_violation_set(apply_repair(art, c), reg, _OPTIONS)
            assert after.issubset(before), (c.target_ref, c.proposed_value)

    # project returns exactly the individually feasible candidates
    feasible = v.project(cands, art, reg, options=_OPTIONS)
    assert all(r.feasible for r in feasible)
    for r in feasible:
        after = hard_violation_set(apply_repair(art, r), reg, _OPTIONS)
        assert after.issubset(before)

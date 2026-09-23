"""Unit tests for the Verifier: the per-candidate projector verdict and ``project``."""

from __future__ import annotations


import pytest

from care.core import (
    Cell,
    ConstraintRegistry,
    DataArtifact,
    DQRequirements,
    RepairCandidate,
    clear_registry,
    register_constraint,
)
from care.predicates import ALL_CONSTRAINTS
from care.verify import Verifier


@pytest.fixture(autouse=True)
def _load_constraints():
    clear_registry()
    for cls in ALL_CONSTRAINTS:
        register_constraint(cls)
    yield
    clear_registry()


def _reg(enabled, **kw):
    return ConstraintRegistry.from_requirements(
        DQRequirements(enabled=set(enabled), **kw), strict=True
    )


# --------------------------------------------------------------------------- #
# Verifier.assess -- the projector verdict                                      #
# --------------------------------------------------------------------------- #

def test_assess_accepts_repair_that_removes_violation():
    art = DataArtifact(artifact_id="A")
    art.set_cell(Cell(row_id="r1", col="age", value=None, is_required=True))
    reg = _reg({"dq.completeness.floor"})

    good = RepairCandidate(target_ref="r1::age", proposed_value=30)
    vr = Verifier().assess(good, art, reg)
    assert vr.feasible is True
    assert vr.s_margin == 1.0  # all hard constraints satisfied after the fix


def test_assess_rejects_repair_that_introduces_violation():
    art = DataArtifact(artifact_id="A")
    # r1 currently fine (non-null required); blanking it creates a new violation
    art.set_cell(Cell(row_id="r1", col="age", value=30, is_required=True))
    reg = _reg({"dq.completeness.floor"})

    bad = RepairCandidate(target_ref="r1::age", proposed_value=None)
    vr = Verifier().assess(bad, art, reg)
    assert vr.feasible is False


def test_project_drops_h_violating_candidates():
    art = DataArtifact(artifact_id="A")
    art.set_cell(Cell(row_id="r1", col="age", value=None, is_required=True))
    art.set_cell(Cell(row_id="r2", col="age", value=20, is_required=True))
    reg = _reg({"dq.completeness.floor"})

    cands = [
        RepairCandidate(target_ref="r1::age", proposed_value=25),   # fixes -> feasible
        RepairCandidate(target_ref="r2::age", proposed_value=None),  # breaks -> dropped
    ]
    feasible = Verifier().project(cands, art, reg)
    assert {r.target_ref for r in feasible} == {"r1::age"}

"""Unit tests for the data-quality predicates in care.predicates.dq."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from care.core import (
    Cell,
    DataArtifact,
    DQRequirements,
    ConstraintRegistry,
    ProvenanceEvent,
    Source,
    clear_registry,
    register_constraint,
)
from care.core.constraints import Params
from care.predicates import ALL_CONSTRAINTS, dq


@pytest.fixture(autouse=True)
def _registry():
    clear_registry()
    for cls in ALL_CONSTRAINTS:
        register_constraint(cls)
    yield
    clear_registry()


def _art(cells, *, sources=(), trace=True):
    art = DataArtifact(artifact_id="A")
    for rid, col, val, req in cells:
        art.set_cell(Cell(row_id=rid, col=col, value=val, is_required=req, dtype="string"))
    for s in sources:
        art.add_source(s)
    if trace:
        for i, c in enumerate(art.iter_cells()):
            art.add_event(ProvenanceEvent(
                event_id=f"e{i}", type="creation", target_ref=c.key.key,
                source_id="s1", actor="ingest", ts=datetime.now(timezone.utc),
            ))
    return art


def _p(**opts):
    return Params(options=opts)


# --------------------------------------------------------------------------- #

def test_completeness_measures_the_filled_ratio():
    art = _art([("r1", "a", "x", True), ("r2", "a", None, True)])
    r = dq.Completeness().evaluate(art, _p())
    assert r.score == 0.5
    assert r.violating_refs == ["r2::a"]


def test_completeness_is_vacuous_without_required_cells():
    art = _art([("r1", "a", None, False)])
    assert dq.Completeness().evaluate(art, _p()).score == 1.0


def test_completeness_floor_gates_on_any_missing_required_cell():
    ok = _art([("r1", "a", "x", True)])
    bad = _art([("r1", "a", None, True)])
    assert dq.CompletenessFloor().evaluate(ok, _p()).satisfied is True
    res = dq.CompletenessFloor().evaluate(bad, _p())
    assert res.satisfied is False and res.violating_refs == ["r1::a"]


def test_completeness_floor_cannot_be_newly_violated_by_filling_a_value():
    bad = _art([("r1", "a", None, True), ("r2", "a", "y", True)])
    before = set(dq.CompletenessFloor().evaluate(bad, _p()).violating_refs)
    bad.set_cell(Cell(row_id="r1", col="a", value="filled", is_required=True, dtype="string"))
    after = set(dq.CompletenessFloor().evaluate(bad, _p()).violating_refs)
    assert after <= before


def test_consistency_flags_rows_that_disagree_on_a_dependent_column():
    art = _art([
        ("r1", "dept", "eng", True), ("r1", "city", "berlin", True),
        ("r2", "dept", "eng", True), ("r2", "city", "munich", True),
    ])
    opts = _p(fds=[(["dept"], ["city"])])
    res = dq.ConsistencyFD().evaluate(art, opts)
    assert res.satisfied is False
    assert set(res.violating_refs) == {"r1::city", "r2::city"}


def test_consistency_is_satisfied_when_no_fds_are_declared():
    art = _art([("r1", "dept", "eng", True)])
    assert dq.ConsistencyFD().evaluate(art, _p()).satisfied is True


def test_traceability_flags_refs_with_no_provenance_edge():
    art = _art([("r1", "a", "x", True)], trace=False)
    res = dq.Traceability().evaluate(art, _p())
    assert res.satisfied is False and "r1::a" in res.violating_refs
    assert dq.Traceability().evaluate(_art([("r1", "a", "x", True)]), _p()).satisfied is True


def test_source_integrity_requires_a_digest_on_every_declared_source():
    unhashed = Source(source_id="s1", uri="u", kind="scraped")
    art = _art([("r1", "a", "x", True)], sources=[unhashed])
    res = dq.SourceIntegrity().evaluate(art, _p())
    assert res.satisfied is False and res.violating_refs == ["meta:source:s1"]


def test_source_integrity_is_vacuous_without_sources():
    assert dq.SourceIntegrity().evaluate(_art([("r1", "a", "x", True)]), _p()).satisfied is True


# --------------------------------------------------------------------------- #

def test_shipped_set_is_registrable_and_typed():
    cids = [c.cid for c in ALL_CONSTRAINTS]
    assert len(cids) == len(set(cids)), "duplicate cid"
    assert all(c.ctype in ("H", "W", "P") for c in ALL_CONSTRAINTS)
    assert all(c.source for c in ALL_CONSTRAINTS)
    assert "dq.completeness.floor" in cids, "the predicate every experiment enables"


def test_shipped_set_stays_heterogeneous():
    """The shipped predicates read different parts of the artifact."""
    reads = {frozenset(c.reads) for c in ALL_CONSTRAINTS}
    assert len(reads) >= 3, f"predicates read only {reads}"
    hard = [c for c in ALL_CONSTRAINTS if c.ctype == "H"]
    assert len(hard) >= 3, "need several hard predicates to project onto"


def test_only_the_completeness_floor_is_enabled_by_a_strict_requirement_set():
    clear_registry()
    for cls in ALL_CONSTRAINTS:
        register_constraint(cls)
    reg = ConstraintRegistry.from_requirements(
        DQRequirements(enabled={"dq.completeness.floor"}), strict=True
    )
    assert [c.cid for c in reg.all()] == ["dq.completeness.floor"]
    assert [c.cid for c in reg.hard()] == ["dq.completeness.floor"]

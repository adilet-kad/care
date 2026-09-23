"""Unit tests for the CARE core data model and constraint registry.

These assert the data-model contracts behave as specified and that the registry
enforces the H/W/P typing invariant. The heavy property-based tests (the I2
projector invariant, conformal coverage) live under tests/property/.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from care.core import (
    BaseConstraint,
    Cell,
    CellKey,
    ConstraintRegistry,
    ConstraintResult,
    DataArtifact,
    DQRequirements,
    Params,
    ProvenanceEvent,
    RepairCandidate,
    Source,
    VerifiedRepair,
    clear_registry,
    register_constraint,
    registered_constraints,
)
from care.core.provenance import uncovered_refs


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts and ends with an empty global registry."""
    clear_registry()
    yield
    clear_registry()


def _register_completeness() -> type[BaseConstraint]:
    """A minimal Completeness plugin used to exercise the registry machinery
    (the shipped plugins live in care.predicates)."""

    @register_constraint
    class Completeness(BaseConstraint):
        cid = "dq.completeness"
        source = "data-quality: completeness of required attributes"
        ctype = "W"
        reads = ["cells"]

        def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult:
            required = [c for c in art.iter_cells() if c.is_required]
            if not required:
                return ConstraintResult(score=1.0)
            missing = [c for c in required if c.value is None]
            return ConstraintResult(
                score=1.0 - len(missing) / len(required),
                violating_refs=[c.key.key for c in missing],
            )

    return Completeness


# --------------------------------------------------------------------------- #
# CellKey                                                                       #
# --------------------------------------------------------------------------- #

def test_cellkey_roundtrip():
    k = CellKey("r1", "age")
    assert str(k) == "r1::age"
    assert k.key == "r1::age"
    assert CellKey.parse("r1::age") == k


def test_cellkey_is_hashable_and_equal():
    assert CellKey("r1", "age") == CellKey("r1", "age")
    d = {CellKey("r1", "age"): 1}
    assert d[CellKey("r1", "age")] == 1


def test_cellkey_rejects_reserved_separator():
    with pytest.raises(ValueError):
        CellKey("r::1", "age")
    with pytest.raises(ValueError):
        CellKey.parse("no-separator-here")


# --------------------------------------------------------------------------- #
# Source validation                                                            #
# --------------------------------------------------------------------------- #

def test_source_trust_prior_bounds():
    Source(source_id="s1", uri="http://x", kind="api", trust_prior=0.3)
    with pytest.raises(Exception):
        Source(source_id="s1", uri="http://x", kind="api", trust_prior=1.5)


def test_source_sha256_validation_and_normalisation():
    digest = "A" * 64
    s = Source(source_id="s1", uri="u", kind="internal", sha256=digest)
    assert s.sha256 == "a" * 64  # normalised to lower-case
    with pytest.raises(Exception):
        Source(source_id="s2", uri="u", kind="internal", sha256="not-a-digest")


# --------------------------------------------------------------------------- #
# DataArtifact cell access + JSON round-trip                                    #
# --------------------------------------------------------------------------- #

def test_artifact_cell_access_and_json_roundtrip():
    art = DataArtifact(artifact_id="A")
    art.set_cell(Cell(row_id="r1", col="age", value=None, is_required=True))
    art.set_cell(Cell(row_id="r1", col="name", value="Ada"))

    assert art.cell("r1", "age").is_required is True
    assert art.get_cell(CellKey("r1", "name")).value == "Ada"
    assert art.get_cell("r1::name").value == "Ada"
    assert {k.col for k in art.cell_keys()} == {"age", "name"}

    blob = art.to_json()
    restored = DataArtifact.from_json(blob)
    assert restored.cell("r1", "name").value == "Ada"
    assert restored.cell("r1", "age").is_required is True


# --------------------------------------------------------------------------- #
# ConstraintResult bounds                                                      #
# --------------------------------------------------------------------------- #

def test_constraint_result_score_must_be_unit_interval():
    ConstraintResult(score=0.0)
    ConstraintResult(score=1.0)
    with pytest.raises(Exception):
        ConstraintResult(score=1.4)
    with pytest.raises(Exception):
        ConstraintResult(score=-0.1)


# --------------------------------------------------------------------------- #
# Registry                                                                     #
# --------------------------------------------------------------------------- #

def test_registry_assembles_from_statement_of_applicability():
    _register_completeness()
    cid = "dq.completeness"
    dqr = DQRequirements(enabled={cid}, targets={cid: 0.95}, weights={cid: 2.0})

    reg = ConstraintRegistry.from_requirements(dqr)
    assert len(reg) == 1
    assert len(reg.soft()) == 1
    assert reg.hard() == []
    assert reg.process() == []
    assert cid in reg

    c = reg.get(cid)
    art = DataArtifact(artifact_id="A")
    art.set_cell(Cell(row_id="r1", col="x", value=None, is_required=True))
    art.set_cell(Cell(row_id="r2", col="x", value=7, is_required=True))
    result = c.evaluate(art, Params.for_cid(cid, dqr))
    assert result.score == 0.5
    assert result.violating_refs == ["r1::x"]


def test_registry_respects_disabled_and_by_source():
    _register_completeness()

    # Nothing enabled -> empty assembled registry.
    empty = ConstraintRegistry.from_requirements(DQRequirements(enabled=set()))
    assert empty.all() == []

    # from_registered ignores the SoA; by_source does a substring match.
    everything = ConstraintRegistry.from_registered()
    assert len(everything.by_source("completeness")) == 1
    assert everything.by_source("provenance") == []


def test_registry_strict_mode_raises_on_unregistered():
    dqr = DQRequirements(enabled={"does.not.exist"})
    ConstraintRegistry.from_requirements(dqr)  # lenient: skips
    with pytest.raises(KeyError):
        ConstraintRegistry.from_requirements(dqr, strict=True)


def test_register_rejects_invalid_ctype():
    with pytest.raises(ValueError):

        @register_constraint
        class BadType(BaseConstraint):
            cid = "bad.type"
            source = "somewhere"
            ctype = "Z"  # not H/W/P

            def evaluate(self, art, params):
                return ConstraintResult()


def test_register_rejects_duplicate_cid():
    _register_completeness()
    with pytest.raises(ValueError):

        @register_constraint
        class Dup(BaseConstraint):
            cid = "dq.completeness"  # already taken
            source = "data-quality: completeness of required attributes"
            ctype = "W"

            def evaluate(self, art, params):
                return ConstraintResult()


def test_register_requires_cid_and_source():
    with pytest.raises(ValueError):

        @register_constraint
        class NoCid(BaseConstraint):
            cid = ""
            source = "x"
            ctype = "H"

            def evaluate(self, art, params):
                return ConstraintResult()


def test_registered_constraints_snapshot_is_a_copy():
    _register_completeness()
    snap = registered_constraints()
    snap.clear()
    assert len(registered_constraints()) == 1  # internal registry untouched


# --------------------------------------------------------------------------- #
# Repair lifecycle                                                             #
# --------------------------------------------------------------------------- #

def test_verified_repair_inherits_candidate_fields():
    vr = VerifiedRepair(
        target_ref="r1::x", proposed_value=42, feasible=True, delta_objective=0.7
    )
    assert isinstance(vr, RepairCandidate)
    assert vr.feasible is True
    assert vr.s_margin is None
    assert vr.proposed_value == 42


# --------------------------------------------------------------------------- #
# Provenance coverage                                                           #
# --------------------------------------------------------------------------- #

def test_uncovered_refs_and_has_provenance():
    art = DataArtifact(artifact_id="A")
    art.set_cell(Cell(row_id="r1", col="x", value=1))
    art.set_cell(Cell(row_id="r2", col="x", value=2))
    art.add_source(Source(source_id="s1", uri="u", kind="internal"))
    art.add_event(
        ProvenanceEvent(
            event_id="e1",
            type="creation",
            target_ref="r1::x",
            source_id="s1",
            actor="ingest",
            ts=datetime.now(timezone.utc),
        )
    )

    assert uncovered_refs(art) == ["r2::x"]
    assert art.has_provenance("r1::x") is True
    assert art.has_provenance("r2::x") is False

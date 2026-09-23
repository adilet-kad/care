"""Unit tests for the Auditor.

The Auditor is assessment-only: it runs the registry over an artifact and emits
(AuditReport, V). These tests lock down the soft-measure scores, the hard-violation
list, the work-queue construction (V = hard u soft-below-target), the
satisfied-fraction headline, and the per-cid options passthrough.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from care.audit import Auditor
from care.core import (
    Cell,
    ConstraintRegistry,
    DataArtifact,
    DQRequirements,
    MetadataTriple,
    ProvenanceEvent,
    Source,
    clear_registry,
    register_constraint,
)
from care.predicates import ALL_CONSTRAINTS


@pytest.fixture(autouse=True)
def _load_constraints():
    clear_registry()
    for cls in ALL_CONSTRAINTS:
        register_constraint(cls)
    yield
    clear_registry()


def _dirty_artifact() -> DataArtifact:
    """A toy artifact with a missing required cell, an unhashed source, a traced
    cell, an untraced cell, and a license triple (so r11 passes)."""
    art = DataArtifact(artifact_id="toy")
    art.set_cell(Cell(row_id="r1", col="age", value=None, is_required=True))
    art.set_cell(Cell(row_id="r2", col="age", value=42, is_required=True))
    art.add_source(Source(source_id="s1", uri="u", kind="scraped"))  # no sha256
    art.add_metadata(
        MetadataTriple(subject="toy", predicate="license", object="MIT", source_id="s1")
    )
    art.add_event(
        ProvenanceEvent(
            event_id="e1", type="creation", target_ref="r1::age",
            source_id="s1", actor="ingest", ts=datetime.now(timezone.utc),
        )
    )
    return art


def _reg(enabled: set[str]) -> ConstraintRegistry:
    return ConstraintRegistry.from_requirements(
        DQRequirements(enabled=enabled), strict=True
    )


def test_auditor_quality_vector_and_hard_violations():
    art = _dirty_artifact()
    reg = _reg({
        "dq.completeness",        # W -> 0.5
        "dq.completeness.floor",  # H -> fails (r1::age)
        "dq.source.integrity",                           # H -> fails (s1)
        "dq.metadata.usage_terms",                   # H -> passes
    })
    report, V = Auditor().run(art, reg)

    assert report.quality_vector["dq.completeness"] == 0.5
    failing = {v.cid for v in report.hard_violations}
    assert failing == {"dq.completeness.floor", "dq.source.integrity"}
    # license passed -> not in violations
    assert "dq.metadata.usage_terms" not in failing


def test_work_queue_is_hard_plus_soft_below_target():
    art = _dirty_artifact()
    # completeness W target 0.9, measured 0.5 -> soft violation; floor H -> hard.
    reg = ConstraintRegistry.from_requirements(
        DQRequirements(
            enabled={"dq.completeness", "dq.completeness.floor"},
            targets={"dq.completeness": 0.9},
        ),
        strict=True,
    )
    report, V = Auditor().run(art, reg)
    cids = {v.cid for v in V}
    assert cids == {"dq.completeness", "dq.completeness.floor"}
    # hard_violations in the report holds only the H failure
    assert {v.cid for v in report.hard_violations} == {"dq.completeness.floor"}


def test_soft_measure_meeting_target_is_not_queued():
    art = _dirty_artifact()
    reg = ConstraintRegistry.from_requirements(
        DQRequirements(
            enabled={"dq.completeness"},
            targets={"dq.completeness": 0.4},  # 0.5 >= 0.4 -> ok
        ),
        strict=True,
    )
    _report, V = Auditor().run(art, reg)
    assert V == []


def test_satisfied_fraction_matches_the_fraction_of_predicates_that_hold():
    art = _dirty_artifact()
    # 3 hard gates: floor (fails, r1::age null), integrity (fails, s1 unhashed),
    # usage terms (passes, the fixture declares a license triple).
    reg = _reg({
        "dq.completeness.floor",
        "dq.source.integrity",
        "dq.metadata.usage_terms",
    })
    report, _V = Auditor().run(art, reg)
    assert report.satisfied_fraction == pytest.approx(1 / 3)




def test_options_passthrough_for_consistency_fds():
    art = DataArtifact(artifact_id="A")
    art.set_cell(Cell(row_id="r1", col="dept", value="eng"))
    art.set_cell(Cell(row_id="r1", col="city", value="NYC"))
    art.set_cell(Cell(row_id="r2", col="dept", value="eng"))
    art.set_cell(Cell(row_id="r2", col="city", value="SF"))  # FD dept->city broken
    reg = _reg({"dq.consistency.fd"})

    # Without FDs: vacuously satisfied.
    report0, V0 = Auditor().run(art, reg)
    assert V0 == []

    # With FDs supplied per cid: hard violation surfaces.
    options = {"dq.consistency.fd": {"fds": [(["dept"], ["city"])]}}
    report1, V1 = Auditor().run(art, reg, options=options)
    assert {v.cid for v in V1} == {"dq.consistency.fd"}
    refs = report1.hard_violations[0].refs
    assert refs == ["r1::city", "r2::city"]


def test_clean_artifact_scores_perfectly():
    art = DataArtifact(artifact_id="clean")
    art.set_cell(Cell(row_id="r1", col="age", value=30, is_required=True))
    art.add_source(Source(source_id="s1", uri="u", kind="internal", sha256="a" * 64))
    art.add_metadata(
        MetadataTriple(subject="clean", predicate="license", object="MIT", source_id="s1")
    )
    art.add_event(
        ProvenanceEvent(
            event_id="e1", type="creation", target_ref="r1::age",
            source_id="s1", actor="ingest", ts=datetime.now(timezone.utc),
        )
    )
    art.add_event(
        ProvenanceEvent(
            event_id="e2", type="creation", target_ref="clean|license|MIT",
            source_id="s1", actor="ingest", ts=datetime.now(timezone.utc),
        )
    )
    reg = _reg({
        "dq.completeness.floor",
        "dq.traceability",
        "dq.source.integrity",
        "dq.metadata.usage_terms",
    })
    report, V = Auditor().run(art, reg)
    assert V == []
    assert report.satisfied_fraction == 1.0
    assert report.hard_violations == []

"""care.predicates.dq -- the shipped data-quality predicates.

Six predicates, chosen to be structurally different rather than numerous. Each
reads a different part of the artifact, so the projector invariant of Section 4.1
is tested against genuinely distinct evaluation shapes:

    dq.completeness           W   per-cell     ratio of required cells that are filled
    dq.completeness.floor     H   per-cell     no required cell is null
    dq.consistency.fd         H   cross-row    every declared FD holds
    dq.traceability           H   lineage      every ref carries a provenance edge
    dq.source.integrity       H   sources      every declared source carries a digest
    dq.metadata.usage_terms   H   metadata     the artifact declares its usage terms

Only ``dq.completeness.floor`` is enabled in the paper's runs; see the package
docstring for why the rest are here.
"""

from __future__ import annotations

from care.core.artifact import CellKey, DataArtifact
from care.core.constraints import BaseConstraint, ConstraintResult, Params
from care.core.provenance import uncovered_refs
from care.core.registry import register_constraint


def _source_ref(source_id: str) -> str:
    """Reference form for a source, so violations point at something addressable."""
    return f"meta:source:{source_id}"


# --------------------------------------------------------------------------- #
# Completeness: a graded measure and the hard floor beneath it                  #
# --------------------------------------------------------------------------- #

@register_constraint
class Completeness(BaseConstraint):
    """W: (#non-null required cells) / (#required cells)."""

    cid = "dq.completeness"
    source = "data-quality: completeness of required attributes"
    ctype = "W"
    reads = ["cells"]

    def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult:
        required = [c for c in art.iter_cells() if c.is_required]
        if not required:
            return ConstraintResult(score=1.0, detail={"required": 0})
        missing = [c for c in required if c.value is None]
        return ConstraintResult(
            score=1.0 - len(missing) / len(required),
            violating_refs=[c.key.key for c in missing],
            detail={"required": len(required), "missing": len(missing)},
        )


@register_constraint
class CompletenessFloor(BaseConstraint):
    """H: no null in any ``is_required`` cell.

    This is the predicate every experiment in the paper runs with, and the reason
    is that it is monotone in the right direction: writing a value into a missing
    cell can never *newly* violate it. The verifier therefore cannot reject a
    proposer's correction for a reason unrelated to the correction's quality, so
    the certified automation measures the conformal gate rather than the filter.
    """

    cid = "dq.completeness.floor"
    source = "data-quality: completeness (hard floor)"
    ctype = "H"
    reads = ["cells"]

    def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult:
        missing = [c for c in art.iter_cells() if c.is_required and c.value is None]
        return ConstraintResult(
            satisfied=len(missing) == 0,
            violating_refs=[c.key.key for c in missing],
        )


# --------------------------------------------------------------------------- #
# Consistency: functional dependencies (the cross-row predicate)                #
# --------------------------------------------------------------------------- #

@register_constraint
class ConsistencyFD(BaseConstraint):
    """H: every declared functional dependency holds.

    FDs arrive via ``params.options["fds"]`` as ``(determinant, dependent)``
    column-list pairs. For each X -> Y, rows agreeing on X must agree on every
    column of Y; rows that disagree are flagged. This is the predicate whose
    evaluation is genuinely cross-row, which is what makes it useful for testing
    that hoisting the violation set is sound: a single-cell repair can in
    principle change another row's verdict.
    """

    cid = "dq.consistency.fd"
    source = "data-quality: functional-dependency consistency"
    ctype = "H"
    reads = ["cells"]

    def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult:
        fds = params.options.get("fds", [])
        if not fds:
            return ConstraintResult(satisfied=True, detail={"fds": 0})

        rows: dict[str, dict[str, object]] = {}
        for c in art.iter_cells():
            rows.setdefault(c.row_id, {})[c.col] = c.value

        violating: set[str] = set()
        for det_cols, dep_cols in fds:
            det_cols = list(det_cols)
            dep_cols = list(dep_cols)
            groups: dict[tuple, list[tuple[str, dict]]] = {}
            for rid, cols in rows.items():
                if any(cols.get(d) is None for d in det_cols):
                    continue
                key = tuple(cols.get(d) for d in det_cols)
                groups.setdefault(key, []).append((rid, cols))
            for members in groups.values():
                for dep in dep_cols:
                    distinct = {m[1].get(dep) for m in members if m[1].get(dep) is not None}
                    if len(distinct) > 1:
                        for rid, _cols in members:
                            violating.add(CellKey(rid, dep).key)

        return ConstraintResult(
            satisfied=len(violating) == 0,
            violating_refs=sorted(violating),
            detail={"fds": len(fds)},
        )


# --------------------------------------------------------------------------- #
# Provenance: the lineage-graph and source-integrity predicates                 #
# --------------------------------------------------------------------------- #

@register_constraint
class Traceability(BaseConstraint):
    """H: every cell carries at least one provenance edge (invariant I6).

    Reads the lineage graph rather than the cells, which is what makes it a
    different evaluation shape from the value predicates above. Empty artifacts
    are vacuously traceable.
    """

    cid = "dq.traceability"
    source = "data-quality: provenance coverage"
    ctype = "H"
    reads = ["cells", "metadata", "provenance"]

    def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult:
        missing = uncovered_refs(art)
        return ConstraintResult(satisfied=len(missing) == 0, violating_refs=missing)


@register_constraint
class SourceIntegrity(BaseConstraint):
    """H: every declared source carries a recorded content digest.

    Reads the source table, so its violating refs are sources rather than cells
    -- the one predicate here whose violations are not addressable as ``row::col``.
    An artifact with no sources is vacuously satisfied. A source failing this is
    quarantined at ingestion rather than repaired, so no proposer can fix it,
    which is precisely why it is useful in the projector test: it is a hard
    predicate that repairs cannot influence in either direction.
    """

    cid = "dq.source.integrity"
    source = "data-quality: per-source content digest"
    ctype = "H"
    reads = ["provenance"]

    def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult:
        sources = art.provenance.sources
        if not sources:
            return ConstraintResult(satisfied=True, detail={"sources": 0})
        missing = [sid for sid, s in sources.items() if not s.sha256]
        return ConstraintResult(
            satisfied=len(missing) == 0,
            violating_refs=[_source_ref(sid) for sid in missing],
            detail={"sources": len(sources), "missing": len(missing)},
        )


@register_constraint
class UsageTermsDeclared(BaseConstraint):
    """H: the artifact declares its usage terms as a metadata triple.

    Reads the metadata triples rather than the cells, so its violating ref is a
    ``meta:<artifact>:<predicate>`` address and a repair for it is a *triple*
    rather than a value. That is the point of keeping it: it is the one shipped
    predicate whose repairs are not cell writes, which keeps the proposer's
    metadata path and the verifier's non-cell projection covered. The required
    predicate name is configurable via ``params.options["predicate"]``.
    """

    cid = "dq.metadata.usage_terms"
    source = "data-quality: declared usage terms"
    ctype = "H"
    reads = ["metadata"]

    def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult:
        want = params.options.get("predicate", "license")
        present = any(t.predicate == want and t.object for t in art.metadata)
        return ConstraintResult(
            satisfied=present,
            violating_refs=[] if present else [f"meta:{art.artifact_id}:{want}"],
            detail={"predicate": want},
        )


PLUGINS = [
    Completeness,
    CompletenessFloor,
    ConsistencyFD,
    Traceability,
    SourceIntegrity,
    UsageTermsDeclared,
]

__all__ = [c.__name__ for c in PLUGINS] + ["PLUGINS"]

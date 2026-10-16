"""Data-quality predicates.

    dq.completeness           W   ratio of required cells that are filled
    dq.completeness.floor     H   no required cell is null
    dq.consistency.fd         H   every declared FD holds
    dq.consistency.cfd        H   every declared conditional FD holds
    dq.traceability           H   every ref carries a provenance edge
    dq.source.integrity       H   every declared source carries a digest
    dq.metadata.usage_terms   H   the artifact declares its usage terms
"""

from __future__ import annotations

from care.core.artifact import CellKey, DataArtifact
from care.core.constraints import BaseConstraint, ConstraintResult, Params
from care.core.provenance import uncovered_refs
from care.core.registry import register_constraint


def _source_ref(source_id: str) -> str:
    """Ref string for a source."""
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
    """H: no null in any ``is_required`` cell. Filling a missing cell never violates it."""

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
    column of Y; rows that disagree are flagged.
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


@register_constraint
class ConsistencyCFD(BaseConstraint):
    """H: every declared conditional functional dependency holds.

    A CFD ``(X -> Y, t_p)`` applies only to rows whose X values match the constants in
    its pattern ``t_p`` (Bohannon et al., ICDE 2007). CFDs arrive via
    ``params.options["cfds"]`` as ``(det_cols, dep_cols, lhs_pattern, rhs_pattern)``,
    the two patterns being ``{column: constant}`` dicts (empty = wildcard). Among the
    rows matching ``lhs_pattern`` (with no null in X):

      * a dependent column with a constant in ``rhs_pattern`` must equal it, a
        single-row check: cell (r, y) violates iff its value is non-null and differs;
      * a dependent column without one must be functionally determined by X, exactly
        as in ``dq.consistency.fd`` but over the matching rows only.

    With both patterns empty a CFD is an FD.
    """

    cid = "dq.consistency.cfd"
    source = "data-quality: conditional functional-dependency consistency"
    ctype = "H"
    reads = ["cells"]

    def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult:
        cfds = params.options.get("cfds", [])
        if not cfds:
            return ConstraintResult(satisfied=True, detail={"cfds": 0})

        rows: dict[str, dict[str, object]] = {}
        for c in art.iter_cells():
            rows.setdefault(c.row_id, {})[c.col] = c.value

        violating: set[str] = set()
        for det_cols, dep_cols, lhs, rhs in cfds:
            det_cols, dep_cols = list(det_cols), list(dep_cols)
            groups: dict[tuple, list[tuple[str, dict]]] = {}
            for rid, cols in rows.items():
                if any(cols.get(d) is None for d in det_cols):
                    continue
                if any(cols.get(col) != const for col, const in lhs.items()):
                    continue
                groups.setdefault(tuple(cols.get(d) for d in det_cols), []).append(
                    (rid, cols))
            for members in groups.values():
                for dep in dep_cols:
                    if dep in rhs:
                        for rid, cols in members:
                            v = cols.get(dep)
                            if v is not None and v != rhs[dep]:
                                violating.add(CellKey(rid, dep).key)
                        continue
                    distinct = {m[1].get(dep) for m in members if m[1].get(dep) is not None}
                    if len(distinct) > 1:
                        for rid, _cols in members:
                            violating.add(CellKey(rid, dep).key)

        return ConstraintResult(
            satisfied=len(violating) == 0,
            violating_refs=sorted(violating),
            detail={"cfds": len(cfds)},
        )


# --------------------------------------------------------------------------- #
# Provenance: the lineage-graph and source-integrity predicates                 #
# --------------------------------------------------------------------------- #

@register_constraint
class Traceability(BaseConstraint):
    """H: every cell and metadata triple carries at least one provenance edge."""

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

    Violating refs are sources (``meta:source:<id>``), which repairs cannot change.
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

    The violating ref is ``meta:<artifact>:<predicate>`` and its repair is a triple.
    The predicate name is set via ``params.options["predicate"]``.
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
    ConsistencyCFD,
    Traceability,
    SourceIntegrity,
    UsageTermsDeclared,
]

__all__ = [c.__name__ for c in PLUGINS] + ["PLUGINS"]

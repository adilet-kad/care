"""Projection primitives onto the hard-feasible set F_H.

A candidate is feasible iff applying it introduces no new hard violation
``(cid, ref)`` relative to the current artifact; it may remove violations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from care.core.artifact import (
    Cell,
    CellKey,
    DataArtifact,
    MetadataTriple,
    ProvenanceEvent,
)
from care.core.constraints import Params
from care.core.registry import ConstraintRegistry
from care.core.repair import RepairCandidate

Options = Mapping[str, Mapping[str, Any]]


def _params(reg: ConstraintRegistry, cid: str, options: Options) -> Params:
    return Params.for_cid(cid, reg.requirements, **dict(options.get(cid, {})))


def apply_repair(art: DataArtifact, repair: RepairCandidate) -> DataArtifact:
    """Return a deep copy of ``art`` with ``repair`` applied.

    A ``MetadataTriple`` is appended to M; otherwise the value is written to the
    cell ``target_ref`` (created if missing). Other refs are a no-op.
    """
    new = art.model_copy(deep=True)
    value = repair.proposed_value

    if isinstance(value, MetadataTriple):
        new.add_metadata(value)
        # record provenance so the new triple is itself traceable
        if value.source_id:
            new.add_event(
                ProvenanceEvent(
                    event_id=f"repair:{value.triple_id}",
                    type="creation",
                    target_ref=value.triple_id,
                    source_id=value.source_id,
                    actor="care.repair",
                    ts=datetime.now(timezone.utc),
                )
            )
        return new

    try:
        key = CellKey.parse(repair.target_ref)
    except ValueError:
        return new  # non-cell ref with a non-triple value -> nothing to do

    existing = new.get_cell(key)
    if existing is None:
        new.set_cell(Cell(row_id=key.row_id, col=key.col, value=value))
    else:
        new.set_cell(existing.model_copy(update={"value": value}))
    return new


def hard_violation_set(
    art: DataArtifact, reg: ConstraintRegistry, options: Options | None = None
) -> set[tuple[str, str]]:
    """All ``(cid, ref)`` pairs currently violating a hard constraint."""
    options = options or {}
    out: set[tuple[str, str]] = set()
    for c in reg.hard():
        res = c.evaluate(art, _params(reg, c.cid, options))
        if not res.satisfied:
            for ref in res.violating_refs:
                out.add((c.cid, ref))
    return out


def _score(res) -> float:
    return res.score if res.score is not None else 1.0


def soft_objective_gain(
    art: DataArtifact,
    trial: DataArtifact,
    reg: ConstraintRegistry,
    options: Options | None = None,
) -> float:
    """Sum_i w_i * (q_i(trial) - q_i(art)) over the soft measures."""
    options = options or {}
    gain = 0.0
    for c in reg.soft():
        p = _params(reg, c.cid, options)
        q0 = _score(c.evaluate(art, p))
        q1 = _score(c.evaluate(trial, p))
        gain += reg.requirements.weight(c.cid) * (q1 - q0)
    return gain


def edit_cost(
    repair: RepairCandidate, art: DataArtifact, default_trust: float = 1.0
) -> float:
    """Trust-weighted edit cost ``1 + (1 - tau)``, tau = mean evidence trust.

    Without known evidence, tau is ``default_trust``.
    """
    if repair.evidence:
        trusts = [
            art.provenance.sources[s].trust_prior
            for s in repair.evidence
            if s in art.provenance.sources
        ]
        trust = sum(trusts) / len(trusts) if trusts else default_trust
    else:
        trust = default_trust
    return 1.0 + (1.0 - trust)


def hard_satisfied_fraction(
    art: DataArtifact, reg: ConstraintRegistry, options: Options | None = None
) -> float:
    """Fraction of hard constraints satisfied (used as ``s_margin``)."""
    options = options or {}
    hards = reg.hard()
    if not hards:
        return 1.0
    ok = sum(1 for c in hards if c.evaluate(art, _params(reg, c.cid, options)).satisfied)
    return ok / len(hards)


__all__ = [
    "apply_repair",
    "hard_violation_set",
    "soft_objective_gain",
    "edit_cost",
    "hard_satisfied_fraction",
]

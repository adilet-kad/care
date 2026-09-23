"""care.core.constraints -- the constraint contract C = (H, W, P).

A constraint is a machine-decidable predicate over the artifact, typed by the
kind of guarantee it carries -- the "separated guarantee types":

    H  hard     {0,1} gate; the verifier projects onto the feasible set F_H
    W  soft     q_i in [0,1] graded measure, reported by the auditor
    P  process  clause about procedure; routed to humans, never faked as data

Concrete predicates are *not* implemented here -- they live as plugins under
``care/predicates/``. This module defines only the contract they fill.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from care.core.artifact import DataArtifact, DQRequirements

ConstraintType = Literal["H", "W", "P"]


class ConstraintResult(BaseModel):
    """The outcome of evaluating one constraint against an artifact.

    For H / P constraints, ``satisfied`` is set and ``score`` is None. For W
    constraints, ``score`` (q_i) is set in [0, 1] and ``satisfied`` is None.
    ``violating_refs`` lists the cells/triples that need repair -- this is what
    the Auditor turns into the Proposer's work queue.
    """

    model_config = ConfigDict(extra="forbid")

    satisfied: bool | None = None       # H / P
    score: float | None = None          # q_i in [0, 1] for W
    violating_refs: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def _bounded(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("score (q_i) must lie in [0, 1]")
        return v


class Params(BaseModel):
    """Per-constraint configuration bound at evaluation time.

    ``theta`` (the target) and ``weight`` (w_i) come from the DQRequirements;
    ``options`` carries constraint-specific knobs (e.g. the FD set for
    ``dq.consistency.fd``, the predicate name for ``dq.metadata.usage_terms``).
    """

    model_config = ConfigDict(extra="forbid")

    theta: float | None = None
    weight: float = 1.0
    options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def for_cid(cls, cid: str, dqr: DQRequirements, **options: Any) -> "Params":
        return cls(theta=dqr.theta(cid), weight=dqr.weight(cid), options=options)


class BaseConstraint(ABC):
    """Base class for constraint plugins under ``care/predicates/``.

    ``cid`` is the stable identifier (e.g. ``"dq.completeness"``), ``source``
    a human-readable description, ``ctype`` the H/W/P type, and ``reads``
    declares which artifact slices the predicate touches (``"cells"`` /
    ``"metadata"`` / ``"provenance"``). Authoring a new constraint is: subclass,
    declare ``cid / source / ctype / reads``, implement ``evaluate``, and
    decorate with ``@register_constraint``.
    """

    cid: ClassVar[str]
    source: ClassVar[str]
    ctype: ClassVar[ConstraintType]
    reads: ClassVar[list[str]] = []

    @abstractmethod
    def evaluate(self, art: DataArtifact, params: Params) -> ConstraintResult: ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} cid={getattr(self, 'cid', '?')!r} ctype={getattr(self, 'ctype', '?')!r}>"


__all__ = [
    "ConstraintType",
    "ConstraintResult",
    "Params",
    "BaseConstraint",
]

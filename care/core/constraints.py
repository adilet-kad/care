"""Constraint contract C = (H, W, P).

    H  hard     {0,1} gate; the verifier projects onto the feasible set F_H
    W  soft     graded measure q_i in [0,1], reported by the auditor
    P  process  procedural clause, routed to humans

Concrete predicates live in ``care.predicates``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from care.core.artifact import DataArtifact, DQRequirements

ConstraintType = Literal["H", "W", "P"]


class ConstraintResult(BaseModel):
    """Outcome of one constraint: ``satisfied`` for H/P, ``score`` in [0, 1] for W."""

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
    """Per-constraint target, weight and options (e.g. the FD set)."""

    model_config = ConfigDict(extra="forbid")

    theta: float | None = None
    weight: float = 1.0
    options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def for_cid(cls, cid: str, dqr: DQRequirements, **options: Any) -> "Params":
        return cls(theta=dqr.theta(cid), weight=dqr.weight(cid), options=options)


class BaseConstraint(ABC):
    """Base class for constraint plugins.

    Subclasses declare ``cid``, ``source`` (description), ``ctype`` (H/W/P) and
    ``reads`` (artifact parts touched), implement ``evaluate`` and are decorated
    with ``@register_constraint``.
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

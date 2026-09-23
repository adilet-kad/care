"""care.core.repair -- the repair lifecycle objects.

A repair flows through the pipeline as:

    Violation            (Auditor: a constraint failed, on these refs)
      -> RepairCandidate (Proposer: a proposed fix + evidence + s_llm/s_agree)
      -> VerifiedRepair  (Verifier: feasible against H? + delta_objective)
      -> Decision        (Conformal controller: auto_apply vs escalate)
      -> CalibrationRecord (human label, feeds recalibration)

``AuditReport`` is the Auditor's summary output (the soft-measure scores, the
failed hard predicates and the satisfied fraction). ``ThresholdTable`` is the
conformal controller's per-stratum output.

Crucially, ``RepairCandidate`` is what the LLM produces -- it never writes to D
(I1). Only after the Verifier projects it to a ``VerifiedRepair`` with
``feasible=True`` (I2) and the controller emits an ``auto_apply`` Decision can
it be applied.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ActionType = Literal["auto_apply", "escalate"]


class Violation(BaseModel):
    """A failed constraint and the refs responsible. The Auditor's work queue V
    is a collection of these (hard failures union soft measures below theta).
    """

    model_config = ConfigDict(extra="forbid")

    cid: str
    refs: list[str] = Field(default_factory=list)
    severity: float = 1.0


class RepairCandidate(BaseModel):
    """A proposed repair for a single ref. Produced by the Proposer; the LLM
    fills value/rationale/evidence/s_llm/s_agree. ``s_margin`` (verifier) and
    ``s_hat`` (controller) are filled downstream.

    ``proposed_value`` is a cell value for X, or a ``MetadataTriple`` for M.
    ``evidence`` lists the retrieved source_ids backing the edit -- required so
    that score gains are source-justified (I4); evidence-free metadata repairs
    are rejected by the H-floor.
    """

    model_config = ConfigDict(extra="forbid")

    target_ref: str
    proposed_value: Any = None          # cell value, or a MetadataTriple for M
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)   # retrieved source_ids (I4)
    s_llm: float | None = Field(None, ge=0.0, le=1.0)   # LLM likelihood; None when no logprob
    s_agree: float = Field(0.0, ge=0.0, le=1.0)         # self-consistency / ensemble
    s_consistency: float | None = None  # per-cell confidence signal from the proposer's log
    s_margin: float | None = None       # verifier/constraint margin (filled by the verifier)
    s_hat: float | None = None          # combined confidence (filled by the controller)


class VerifiedRepair(RepairCandidate):
    """A candidate that has passed through the Verifier's projector.

    ``feasible`` MUST be True for the repair to be applicable -- it is the
    engineering form of "projected onto F_H" (I2), asserted by the property-
    based projector test. ``delta_objective`` is the soft-objective gain
    ``sum_i w_i * dq_i - lambda * Delta`` of applying the repair (I3: a
    heuristic diagnostic; correctness is not claimed from it).
    """

    feasible: bool = False
    delta_objective: float = 0.0


class CalibrationRecord(BaseModel):
    """One labelled outcome used to calibrate the conformal threshold. ``correct``
    is the human/ground-truth verdict on a repair with confidence ``s_hat`` in a
    given ``stratum`` (for group-conditional / Mondrian control).
    """

    model_config = ConfigDict(extra="forbid")

    target_ref: str
    s_hat: float
    correct: bool
    stratum: str = "default"


class Decision(BaseModel):
    """The controller's auto-apply / escalate decision for one repair, recording
    the confidence and the stratum threshold that produced it.
    """

    model_config = ConfigDict(extra="forbid")

    target_ref: str
    action: ActionType
    s_hat: float
    lambda_hat: float
    stratum: str = "default"


class ThresholdTable(BaseModel):
    """Per-stratum auto/escalate thresholds lambda_hat from the conformal
    controller, guaranteeing R(lambda_hat) <= alpha w.p. 1 - delta. Auto-apply
    iff ``s_hat >= threshold_for(stratum)``.
    """

    model_config = ConfigDict(extra="forbid")

    alpha: float
    delta: float
    thresholds: dict[str, float] = Field(default_factory=dict)  # stratum -> lambda_hat
    default: float = 1.0

    def threshold_for(self, stratum: str) -> float:
        return self.thresholds.get(stratum, self.default)


class AuditReport(BaseModel):
    """The Auditor's assessment-only summary of an artifact.

    ``quality_vector`` maps each soft predicate's cid to its graded score in
    [0, 1]; ``hard_violations`` lists the boolean predicates that failed; and
    ``satisfied_fraction`` is the headline — the fraction of enabled predicates
    the artifact satisfies, counting a soft one as satisfied when it meets its
    declared target. The work queue is returned separately by the Auditor, so
    this report describes the artifact rather than prescribing repairs.
    """

    model_config = ConfigDict(extra="forbid")

    quality_vector: dict[str, float] = Field(default_factory=dict)  # cid -> q_i
    hard_violations: list[Violation] = Field(default_factory=list)
    satisfied_fraction: float = 0.0


__all__ = [
    "ActionType",
    "Violation",
    "RepairCandidate",
    "VerifiedRepair",
    "CalibrationRecord",
    "Decision",
    "ThresholdTable",
    "AuditReport",
]

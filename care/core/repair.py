"""Repair lifecycle objects.

    Violation          -> RepairCandidate   (cleaner's proposed fix)
                       -> VerifiedRepair    (verifier: feasible against H)
                       -> Decision          (controller: auto_apply / escalate)
    CalibrationRecord  labelled outcome used to calibrate the threshold
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ActionType = Literal["auto_apply", "escalate"]


class Violation(BaseModel):
    """A failed constraint and the refs responsible."""

    model_config = ConfigDict(extra="forbid")

    cid: str
    refs: list[str] = Field(default_factory=list)
    severity: float = 1.0


class RepairCandidate(BaseModel):
    """A cleaner's proposed repair for one ref.

    ``proposed_value`` is a cell value, or a ``MetadataTriple`` for M.
    ``s_margin`` and ``s_hat`` are filled by the verifier and controller.
    """

    model_config = ConfigDict(extra="forbid")

    target_ref: str
    proposed_value: Any = None          # cell value, or a MetadataTriple for M
    rationale: str = ""
    evidence: list[str] = Field(default_factory=list)   # source_ids backing the edit
    s_conf: float | None = Field(None, ge=0.0, le=1.0)   # optional extra score; unset by the log reader
    s_agree: float = Field(0.0, ge=0.0, le=1.0)         # cleaner's confidence (a constant if the log has none)
    s_consistency: float | None = None  # cleaner's confidence, None if the log has none
    s_margin: float | None = None       # hard-constraint margin (verifier)
    s_hat: float | None = None          # combined confidence (controller)


class VerifiedRepair(RepairCandidate):
    """A candidate assessed by the verifier.

    Only ``feasible`` repairs may be applied. ``delta_objective`` is the
    diagnostic gain ``sum_i w_i * dq_i - lambda * Delta``.
    """

    feasible: bool = False
    delta_objective: float = 0.0


class CalibrationRecord(BaseModel):
    """A labelled repair (confidence, correctness, stratum) used for calibration."""

    model_config = ConfigDict(extra="forbid")

    target_ref: str
    s_hat: float
    correct: bool
    stratum: str = "default"


class Decision(BaseModel):
    """Auto-apply / escalate decision for one repair, with its score and threshold."""

    model_config = ConfigDict(extra="forbid")

    target_ref: str
    action: ActionType
    s_hat: float
    lambda_hat: float
    stratum: str = "default"


class ThresholdTable(BaseModel):
    """Per-stratum thresholds lambda_hat; auto-apply iff ``s_hat >= threshold_for(stratum)``."""

    model_config = ConfigDict(extra="forbid")

    alpha: float
    delta: float
    thresholds: dict[str, float] = Field(default_factory=dict)  # stratum -> lambda_hat
    default: float = 1.0

    def threshold_for(self, stratum: str) -> float:
        return self.thresholds.get(stratum, self.default)


class AuditReport(BaseModel):
    """Auditor summary: soft scores, failed hard predicates and the satisfied fraction.

    ``satisfied_fraction`` counts a soft predicate as satisfied when it meets its target.
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

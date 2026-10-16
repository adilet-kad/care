"""Verifier: keeps only candidates that introduce no new hard violation."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from care.core.artifact import DataArtifact
from care.core.registry import ConstraintRegistry
from care.core.repair import RepairCandidate, VerifiedRepair
from care.verify.projector import (
    apply_repair,
    edit_cost,
    hard_satisfied_fraction,
    hard_violation_set,
    soft_objective_gain,
)

Options = Mapping[str, Mapping[str, Any]]


class Verifier:
    """Projects repair candidates onto F_H."""

    def __init__(self, lam: float = 1.0, default_trust: float = 1.0):
        self.lam = lam
        self.default_trust = default_trust

    def assess(
        self,
        cand: RepairCandidate,
        art: DataArtifact,
        reg: ConstraintRegistry,
        *,
        options: Options | None = None,
        before: set[tuple[str, str]] | None = None,
    ) -> VerifiedRepair:
        """Assess one candidate: feasibility, delta_objective and margin."""
        options = options or {}
        if before is None:
            before = hard_violation_set(art, reg, options)
        trial = apply_repair(art, cand)
        after = hard_violation_set(trial, reg, options)
        feasible = after.issubset(before)
        delta = soft_objective_gain(art, trial, reg, options) - self.lam * edit_cost(
            cand, art, self.default_trust
        )
        margin = hard_satisfied_fraction(trial, reg, options)
        return VerifiedRepair(
            target_ref=cand.target_ref,
            proposed_value=cand.proposed_value,
            rationale=cand.rationale,
            evidence=list(cand.evidence),
            s_conf=cand.s_conf,
            s_agree=cand.s_agree,
            s_consistency=cand.s_consistency,
            s_margin=margin,
            s_hat=cand.s_hat,
            feasible=feasible,
            delta_objective=delta,
        )

    def project(
        self,
        cands: Iterable[RepairCandidate],
        art: DataArtifact,
        reg: ConstraintRegistry,
        *,
        options: Options | None = None,
        before: set[tuple[str, str]] | None = None,
    ) -> list[VerifiedRepair]:
        """Return the feasible candidates.

        ``before`` is the artifact's current hard-violation set. It does not change
        while candidates are assessed (I1), so pass it in to avoid a full scan per
        candidate.
        """
        options = options or {}
        if before is None:
            before = hard_violation_set(art, reg, options)
        assessed = (self.assess(c, art, reg, options=options, before=before) for c in cands)
        return [vr for vr in assessed if vr.feasible]


__all__ = ["Verifier"]

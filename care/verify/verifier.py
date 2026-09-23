"""care.verify.verifier -- the Verifier facade.

``project`` is the hard guarantee (I2): each candidate is assessed and only
those that keep the artifact in F_H (introduce no new hard violation) are
returned; H-violating proposals are dropped. The per-candidate verdict is
exposed via ``assess`` so the property-based I2 test can inspect both accepted
and rejected proposals.

``s_margin`` (boundary distance) is filled here; ``s_hat`` stays ``None`` until
the conformal controller combines the confidence components.
"""

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
        """Assess one candidate: feasibility (I2), delta_objective (I3), margin."""
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
            s_llm=cand.s_llm,
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
        """Assess all candidates and drop the H-violating ones (return feasible).

        ``before`` is the artifact's current hard-violation set. Computing it scans
        EVERY cell, and it is INVARIANT while proposing (invariant I1: the proposer
        never writes to the artifact), so callers that assess many candidates against
        the same artifact should compute it once and pass it in. Leaving it to be
        recomputed per call is O(cells) per candidate -- on tax (121K candidate cells
        x 3M cells) that measured ~25 s/cell, i.e. months. See bench.study.propose_repairs.
        """
        options = options or {}
        if before is None:
            before = hard_violation_set(art, reg, options)
        assessed = (self.assess(c, art, reg, options=options, before=before) for c in cands)
        return [vr for vr in assessed if vr.feasible]


__all__ = ["Verifier"]

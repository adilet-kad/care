"""Auditor: assessment without repair.

Produces the soft-measure scores, the failed hard predicates, the work queue
V (hard failures plus soft measures below target) and the satisfied fraction.
"""

from __future__ import annotations

from typing import Any, Mapping

from care.core.artifact import DataArtifact
from care.core.constraints import Params
from care.core.registry import ConstraintRegistry
from care.core.repair import AuditReport, Violation


class Auditor:
    """Runs ``C = (H, W, P)`` over D to produce ``(AuditReport, V)``."""

    def run(
        self,
        art: DataArtifact,
        reg: ConstraintRegistry,
        *,
        options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[AuditReport, list[Violation]]:
        options = options or {}
        dqr = reg.requirements

        quality_vector: dict[str, float] = {}
        hard_violations: list[Violation] = []
        soft_violations: list[Violation] = []
        # cid -> passed (H) / met its target (W)
        satisfied: dict[str, bool] = {}

        primary = list(reg.all())

        for c in primary:
            # P clauses are routed to humans, not evaluated
            if c.ctype == "P":
                continue

            params = Params.for_cid(c.cid, dqr, **dict(options.get(c.cid, {})))
            res = c.evaluate(art, params)

            if c.ctype == "H":
                ok = bool(res.satisfied)
                satisfied[c.cid] = ok
                if not ok:
                    hard_violations.append(
                        Violation(cid=c.cid, refs=res.violating_refs, severity=1.0)
                    )
            else:  # W
                q = res.score if res.score is not None else 1.0
                quality_vector[c.cid] = q
                theta = dqr.theta(c.cid)
                meets = theta is None or q >= theta
                satisfied[c.cid] = meets
                if not meets:
                    soft_violations.append(
                        Violation(
                            cid=c.cid,
                            refs=res.violating_refs,
                            severity=max(0.0, theta - q),
                        )
                    )

        # vacuously 1.0 when no predicate is enabled
        satisfied_fraction = (
            sum(1 for ok in satisfied.values() if ok) / len(satisfied)
            if satisfied
            else 1.0
        )

        report = AuditReport(
            quality_vector=quality_vector,
            hard_violations=hard_violations,
            satisfied_fraction=satisfied_fraction,
        )
        work_queue = hard_violations + soft_violations
        return report, work_queue


__all__ = ["Auditor"]

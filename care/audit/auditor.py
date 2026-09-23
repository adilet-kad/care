"""care.audit.auditor -- the Auditor: assessment without repair.

The Auditor runs the assembled ``ConstraintRegistry`` over a ``DataArtifact``
and produces:

  * the soft-measure scores (cid -> q_i for the W predicates),
  * the hard-violation list (H predicates that fail),
  * the work queue ``V = {hard h_k : satisfied=False} u {soft w_i : q_i < theta_i}``
    -- the refs a proposer is asked about, and
  * the headline ``satisfied_fraction``: the share of enabled predicates the
    artifact satisfies.

It does not write to D and it never calls a proposer. In the benchmark it is the
``detection="constraints"`` path (``bench.study.detect``).

Constraint-specific knobs (e.g. the FD set for ``dq.consistency.fd``) are
supplied per cid through the ``options`` mapping.
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
        # cid -> did it pass (H) / meet its target (W). Feeds satisfied_fraction.
        satisfied: dict[str, bool] = {}

        primary = list(reg.all())

        for c in primary:
            # P (process) clauses route to humans; they are never evaluated as
            # data predicates here. None of the shipped predicates is typed P.
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

        # Headline: the fraction of enabled predicates the artifact satisfies.
        # Hard predicates count as satisfied when they hold; soft ones when they
        # meet their declared target. Vacuously 1.0 on an empty requirement set,
        # matching the predicates' own empty-input convention.
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
        # V = hard failures u soft-below-target (the Proposer's work queue).
        work_queue = hard_violations + soft_violations
        return report, work_queue


__all__ = ["Auditor"]

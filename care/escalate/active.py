"""Active escalator: applies the trust override and builds the review queue.

A repair whose best evidence trust is below tau_min is escalated as
``suspected_poison`` even if the conformal threshold would auto-apply it (I4).
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from care.core.artifact import Source
from care.core.repair import Decision, RepairCandidate
from care.conformal.strata import StrataFn, marginal_strata
from care.escalate.queue import (
    EscalationItem,
    EscalationQueue,
    EscalationReason,
)


def _trust(sources: Mapping[str, object], sid: str) -> float | None:
    """Trust of a source id, or None if unknown (unknown is not treated as untrusted)."""
    s = sources.get(sid)
    if s is None:
        return None
    return s.trust_prior if isinstance(s, Source) else float(s)


def _evidence_trust(
    repair: RepairCandidate | None, sources: Mapping[str, object]
) -> float | None:
    """Max trust among a repair's known evidence sources, or None."""
    if repair is None or not repair.evidence:
        return None
    known = [t for t in (_trust(sources, sid) for sid in repair.evidence) if t is not None]
    return max(known) if known else None


class ActiveEscalator:
    def __init__(
        self,
        *,
        tau_min: float = 0.5,
        strata_fn: StrataFn = marginal_strata,
    ):
        self.tau_min = tau_min
        self.strata_fn = strata_fn

    def route(
        self,
        decisions: Sequence[Decision],
        repairs: Iterable[RepairCandidate],
        *,
        sources: Mapping[str, object] | None = None,
    ) -> tuple[list[Decision], EscalationQueue]:
        """Return (post-override decisions, review queue)."""
        sources = dict(sources or {})
        by_ref = {r.target_ref: r for r in repairs}

        final: list[Decision] = []
        items: list[EscalationItem] = []
        for d in decisions:
            repair = by_ref.get(d.target_ref)
            min_trust = _evidence_trust(repair, sources)
            evidence = list(repair.evidence) if repair is not None else []

            poison = min_trust is not None and min_trust < self.tau_min
            action = "escalate" if (poison or d.action == "escalate") else "auto_apply"
            final.append(d.model_copy(update={"action": action}))

            if action == "auto_apply":
                continue
            if poison:
                reason = EscalationReason.SUSPECTED_POISON
            elif math.isinf(d.lambda_hat):
                reason = EscalationReason.UNCONTROLLABLE
            else:
                reason = EscalationReason.BELOW_THRESHOLD
            items.append(
                EscalationItem(
                    target_ref=d.target_ref,
                    s_hat=d.s_hat,
                    stratum=d.stratum,
                    lambda_hat=d.lambda_hat,
                    reason=reason,
                    evidence=evidence,
                    min_trust=min_trust,
                )
            )

        return final, EscalationQueue(items)


__all__ = ["ActiveEscalator"]

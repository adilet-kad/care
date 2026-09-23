"""care.escalate.active -- the active escalator (tau-aware poison routing).

Wraps the controller's decisions with a safety override and a review queue:

  1. **tau-aware poison routing**. A repair justified only by a low-trust source
     (max evidence trust < tau_min) is escalated to human review as
     ``suspected_poison`` even if the conformal layer would auto-apply it. This
     enforces "source-justified gains" at the decision boundary (I4): a trust
     floor on the *evidence* behind an otherwise-confident repair.
  2. **routing** -- every escalated repair becomes an ``EscalationItem`` tagged
     with its reason.

The escalator never writes to the data artifact and never relaxes a guarantee --
it only decides which repairs a human sees.
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
    """Trust of a source id, or None if the id is unknown to the sources map.

    A source absent from the map is *unknown*, not *untrusted*: returning 0.0
    here would let an empty/partial sources map flag every evidence-backed
    repair as poison. Unknown -> None, and the poison gate ignores None."""
    s = sources.get(sid)
    if s is None:
        return None
    return s.trust_prior if isinstance(s, Source) else float(s)


def _evidence_trust(
    repair: RepairCandidate | None, sources: Mapping[str, object]
) -> float | None:
    """Best (max) trust among a repair's evidence sources, or None if no evidence."""
    if repair is None or not repair.evidence:
        return None
    known = [t for t in (_trust(sources, sid) for sid in repair.evidence) if t is not None]
    return max(known) if known else None  # all-unknown evidence -> no trust signal


class ActiveEscalator:
    def __init__(
        self,
        *,
        tau_min: float = 0.5,
        strata_fn: StrataFn = marginal_strata,
    ):
        self.tau_min = tau_min
        self.strata_fn = strata_fn

    # --- poison routing and queue construction ----------------------------- #

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

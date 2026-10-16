"""Unit tests for the escalator: tau-aware poison routing and escalation reasons."""

from __future__ import annotations

import math

from care.core import (
    Decision,
    RepairCandidate,
    Source,
)
from care.escalate import ActiveEscalator, EscalationReason


def _sources(*specs):
    return {sid: Source(source_id=sid, uri="u", kind="api", trust_prior=t) for sid, t in specs}


# --------------------------------------------------------------------------- #
# tau-aware poison routing (safety override)                                    #
# --------------------------------------------------------------------------- #

def test_poison_override_escalates_low_trust_auto_apply():
    esc = ActiveEscalator(tau_min=0.5)
    # s_hat is above threshold, but the only evidence is a low-trust source -> escalate
    decisions = [Decision(target_ref="r1::x", action="auto_apply", s_hat=0.95, lambda_hat=0.6)]
    repairs = [RepairCandidate(target_ref="r1::x", s_hat=0.95, evidence=["leak"])]
    final, queue = esc.route(decisions, repairs, sources=_sources(("leak", 0.1)))
    assert final[0].action == "escalate"
    assert [it for it in queue if it.reason == EscalationReason.SUSPECTED_POISON]
    assert queue.items[0].min_trust == 0.1


def test_high_trust_evidence_is_not_poison_routed():
    esc = ActiveEscalator(tau_min=0.5)
    decisions = [Decision(target_ref="r1::x", action="auto_apply", s_hat=0.95, lambda_hat=0.6)]
    repairs = [RepairCandidate(target_ref="r1::x", s_hat=0.95, evidence=["trusted"])]
    final, queue = esc.route(decisions, repairs, sources=_sources(("trusted", 0.9)))
    assert final[0].action == "auto_apply"
    assert len(queue) == 0


def test_no_evidence_repair_is_governed_only_by_threshold():
    # a cell repair with no external evidence is not poison-flagged
    esc = ActiveEscalator(tau_min=0.5)
    decisions = [Decision(target_ref="r1::x", action="auto_apply", s_hat=0.95, lambda_hat=0.6)]
    repairs = [RepairCandidate(target_ref="r1::x", s_hat=0.95, evidence=[])]
    final, queue = esc.route(decisions, repairs, sources={})
    assert final[0].action == "auto_apply"
    assert len(queue) == 0


def test_absent_source_is_not_treated_as_poison_regression():
    """A source id absent from the sources map means unknown trust, not zero trust.

    evaluate_split passes sources={} when poison_frac=0; the poison gate must then
    leave auto_apply intact.
    """
    esc = ActiveEscalator(tau_min=0.5)
    decisions = [Decision(target_ref="r1::x", action="auto_apply", s_hat=0.95, lambda_hat=0.6)]
    repairs = [RepairCandidate(target_ref="r1::x", s_hat=0.95, evidence=["reference"])]
    final, queue = esc.route(decisions, repairs, sources={})
    assert final[0].action == "auto_apply"      # not overridden to escalate
    assert len(queue) == 0
    assert not [it for it in queue if it.reason == EscalationReason.SUSPECTED_POISON]


def test_uncontrollable_stratum_reason():
    esc = ActiveEscalator(tau_min=0.5)
    decisions = [Decision(target_ref="r1::x", action="escalate", s_hat=0.99, lambda_hat=math.inf)]
    repairs = [RepairCandidate(target_ref="r1::x", s_hat=0.99)]
    _final, queue = esc.route(decisions, repairs, sources={})
    assert queue.items[0].reason == EscalationReason.UNCONTROLLABLE

"""Unit tests for the conformal controller: bounds, the RCPS threshold, the combiner, decisions."""

from __future__ import annotations

import math

import pytest

from care.conformal import (
    ConformalController,
    MeanCombiner,
    clopper_pearson_upper,
    hoeffding_upper,
    rcps_threshold,
)
from care.core import CalibrationRecord, RepairCandidate, ThresholdTable


# --------------------------------------------------------------------------- #
# Bounds                                                                        #
# --------------------------------------------------------------------------- #

def test_hoeffding_upper_basic():
    assert hoeffding_upper(0, 100, 0.1) == pytest.approx(math.sqrt(math.log(10) / 200))
    # more data -> tighter bound
    assert hoeffding_upper(5, 1000, 0.1) < hoeffding_upper(5, 100, 0.1)
    assert hoeffding_upper(50, 100, 0.1) > 0.5  # rhat=0.5 plus slack


def test_clopper_pearson_upper_bounds_and_monotonicity():
    assert clopper_pearson_upper(5, 5, 0.1) == 1.0          # all wrong -> 1.0
    cp = clopper_pearson_upper(0, 100, 0.05)
    assert 0.0 < cp < 0.05                                   # 0 errors, tight upper
    # tighter than Hoeffding for the same data
    assert clopper_pearson_upper(2, 100, 0.1) < hoeffding_upper(2, 100, 0.1)


# --------------------------------------------------------------------------- #
# RCPS threshold                                                                #
# --------------------------------------------------------------------------- #

def _separable():
    # high-confidence repairs are correct; low-confidence ones are wrong.
    # Sized so the (conservative) bound can certify the good tiers at alpha=0.1.
    s = [0.95] * 400 + [0.9] * 400 + [0.3] * 200
    c = [True] * 800 + [False] * 200
    return s, c


def test_rcps_finds_controlling_threshold():
    s, c = _separable()
    lam, trace = rcps_threshold(s, c, alpha=0.1, delta=0.1, bound="hoeffding")
    # the 0.3 (all-wrong) tier must be excluded; threshold lands in the good tiers
    assert 0.3 < lam <= 0.95
    assert trace  # diagnostic trace recorded


def test_rcps_escalates_all_when_uncontrollable():
    # everything wrong -> no threshold can control -> +inf
    lam, _ = rcps_threshold([0.9, 0.8, 0.7], [False, False, False], alpha=0.1, delta=0.1)
    assert lam == math.inf


# --------------------------------------------------------------------------- #
# Combiners                                                                     #
# --------------------------------------------------------------------------- #

def test_mean_combiner_skips_none():
    assert MeanCombiner().combine(0.8, None, 0.4) == pytest.approx(0.6)
    assert MeanCombiner().combine(None, None, None) == 0.0


# --------------------------------------------------------------------------- #
# Controller: combine / calibrate / decide                                      #
# --------------------------------------------------------------------------- #

def _records(s_vals, c_vals, stratum="default"):
    return [
        CalibrationRecord(target_ref=f"r{i}", s_hat=s, correct=c, stratum=stratum)
        for i, (s, c) in enumerate(zip(s_vals, c_vals))
    ]


def test_controller_combine_uses_components():
    ctrl = ConformalController()
    r = RepairCandidate(target_ref="x", s_llm=0.8, s_agree=0.4, s_margin=None)
    assert ctrl.combine(r) == pytest.approx(0.6)


def test_controller_calibrate_and_decide():
    s, c = _separable()
    ctrl = ConformalController(method="rcps", bound="hoeffding")
    thr = ctrl.calibrate(_records(s, c), alpha=0.1, delta=0.1)
    assert isinstance(thr, ThresholdTable)
    assert thr.threshold_for("default") is not None

    high = RepairCandidate(target_ref="hi", s_hat=0.99)
    low = RepairCandidate(target_ref="lo", s_hat=0.2)
    decisions = ctrl.decide([high, low], thr)
    actions = {d.target_ref: d.action for d in decisions}
    assert actions["hi"] == "auto_apply"
    assert actions["lo"] == "escalate"


def test_controller_uncontrollable_escalates_all():
    ctrl = ConformalController()
    thr = ctrl.calibrate(_records([0.9, 0.8], [False, False]), alpha=0.1, delta=0.1)
    assert thr.threshold_for("default") == math.inf
    d = ctrl.decide([RepairCandidate(target_ref="x", s_hat=1.0)], thr)
    assert d[0].action == "escalate"


def test_assign_s_hat_fills_from_combiner():
    ctrl = ConformalController()
    [r] = ctrl.assign_s_hat([RepairCandidate(target_ref="x", s_llm=0.6, s_agree=0.6, s_margin=0.6)])
    assert r.s_hat == pytest.approx(0.6)

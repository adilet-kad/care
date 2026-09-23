"""Conformal-coverage guarantee -- the second load-bearing test.

The headline claim is distribution-free error control: with the threshold
calibrated at level (alpha, delta), the realized error among auto-applied
repairs is <= alpha with probability >= 1 - delta. We validate it the standard
way -- Monte-Carlo over fresh calibration/test splits, checking the fraction of
trials with test error <= alpha is at least 1 - delta.

The data process is a mixture: a reliable high-confidence mass and an unreliable
low-confidence mass. This both (a) makes the test meaningful -- the threshold
genuinely auto-applies a non-trivial set rather than escalating everything -- and
(b) keeps the guarantee honest, since the auto-applied set still carries real
error that must stay under alpha. Seeds are fixed for reproducibility.
"""

from __future__ import annotations

import random

import pytest

from care.conformal import ConformalController
from care.core import CalibrationRecord


def _mixture(rng: random.Random, n: int) -> list[tuple[float, bool]]:
    out = []
    for _ in range(n):
        if rng.random() < 0.6:  # reliable, high-confidence repairs
            s, p = rng.uniform(0.82, 1.0), 0.95
        else:  # unreliable, low-confidence repairs
            s, p = rng.uniform(0.0, 0.6), 0.55
        out.append((s, rng.random() < p))
    return out


def _realized_error(test, lam):
    selected = [c for s, c in test if s >= lam]
    if not selected:
        return 0.0, 0.0  # nothing auto-applied -> no automated error
    err = sum(1 for c in selected if not c) / len(selected)
    return err, len(selected) / len(test)


def _run(ctrl, alpha, delta, *, trials, n_cal, n_test, seed0):
    covered = 0
    auto_fracs = []
    for t in range(trials):
        rng = random.Random(seed0 + t)
        cal, test = _mixture(rng, n_cal), _mixture(rng, n_test)
        recs = [CalibrationRecord(target_ref=f"c{i}", s_hat=s, correct=c)
                for i, (s, c) in enumerate(cal)]
        thr = ctrl.calibrate(recs, alpha=alpha, delta=delta)
        err, frac = _realized_error(test, thr.threshold_for("default"))
        auto_fracs.append(frac)
        if err <= alpha:
            covered += 1
    return covered / trials, sum(auto_fracs) / len(auto_fracs)


@pytest.mark.parametrize("alpha,delta", [(0.1, 0.1), (0.2, 0.1)])
def test_marginal_coverage_holds(alpha, delta):
    ctrl = ConformalController(method="rcps", bound="hoeffding")
    coverage, _frac = _run(
        ctrl, alpha, delta, trials=60, n_cal=900, n_test=400, seed0=10_000
    )
    assert coverage >= (1.0 - delta) - 0.02, (alpha, delta, coverage)


@pytest.mark.parametrize("alpha,delta", [(0.1, 0.1), (0.15, 0.1)])
def test_exact_bound_covers_and_actually_automates(alpha, delta):
    """The tight Clopper-Pearson bound auto-applies a real fraction while keeping
    coverage -- the 'minimal human intervention' regime, not escalate-all."""
    ctrl = ConformalController(method="rcps", bound="exact")
    coverage, auto_frac = _run(
        ctrl, alpha, delta, trials=60, n_cal=500, n_test=400, seed0=50_000
    )
    assert coverage >= (1.0 - delta) - 0.03, (alpha, delta, coverage)
    assert auto_frac > 0.2, f"threshold escalated almost everything ({auto_frac:.2f})"

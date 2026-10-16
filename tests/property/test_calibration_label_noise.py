"""Coverage under symmetric and one-sided calibration label noise, and the alpha - eta fix."""

from __future__ import annotations

import random

from care.conformal.rcps import rcps_threshold

_ALPHA, _ETA, _DELTA = 0.20, 0.20, 0.1
_NCAL, _NTEST, _TRIALS = 1500, 800, 120


def _draw(rng, n):
    s = [rng.random() for _ in range(n)]
    return s, [rng.random() < (0.30 + 0.68 * si) for si in s]   # P(correct) rises with s


def _sym_flip(rng, c, eta):
    return [(not x) if rng.random() < eta else x for x in c]


def _adv_flip(rng, c, eta):                                     # only wrong -> "correct"
    return [True if (not x and rng.random() < eta) else x for x in c]


def _coverage(flip, *, calibrate_at, eta, seed0):
    covered = 0
    for t in range(_TRIALS):
        rng = random.Random(seed0 + t)
        s_cal, true_cal = _draw(rng, _NCAL)
        lam, _ = rcps_threshold(s_cal, flip(rng, true_cal, eta),
                                alpha=calibrate_at, delta=_DELTA, bound="exact")
        s_test, true_test = _draw(rng, _NTEST)
        applied = [tc for si, tc in zip(s_test, true_test) if si >= lam]
        err = 0.0 if not applied else sum(1 for tc in applied if not tc) / len(applied)
        if err <= _ALPHA + 1e-9:
            covered += 1
    return covered / _TRIALS


def test_symmetric_label_noise_retains_validity():
    cov = _coverage(_sym_flip, calibrate_at=_ALPHA, eta=_ETA, seed0=10)
    assert cov >= (1.0 - _DELTA) - 0.03, f"symmetric-noise coverage {cov} dropped below target"


def test_adversarial_calibration_poisoning_breaks_validity():
    cov = _coverage(_adv_flip, calibrate_at=_ALPHA, eta=_ETA, seed0=10)
    assert cov < (1.0 - _DELTA) - 0.05, (
        f"expected adversarial poisoning to break coverage (got {cov}); if this "
        f"now passes, the threat model or bound changed -- revisit the analysis")


def test_deflation_restores_validity_under_adversarial_poisoning():
    cov = _coverage(_adv_flip, calibrate_at=_ALPHA - _ETA, eta=_ETA, seed0=10)
    assert cov >= (1.0 - _DELTA) - 0.03, f"deflated coverage {cov} did not restore validity"

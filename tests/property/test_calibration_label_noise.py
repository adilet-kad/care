"""Monte-Carlo characterization of calibration label corruption.

Assumption A1 (clean calibration) is load-bearing. This test empirically maps what
actually happens when it is violated -- and the result is more nuanced (and more
favorable) than "any label noise breaks the guarantee." Two regimes, verified here:

  (1) RANDOM / symmetric labeler error (each label flipped w.p. eta, both
      directions). Does NOT break validity: the exact (Clopper-Pearson) bound
      absorbs it, coverage stays >= 1 - delta. The cost is THROUGHPUT -- noisy
      calibration widens the bound, so CARE certifies a higher threshold and
      escalates more. So CARE is *robust* to honest labeler mistakes; they cost
      automation, not safety. (Deflating the budget would only reduce automation
      further -- it is NOT the right response to symmetric noise.)

  (2) ADVERSARIAL / biased calibration poisoning (an attacker flips only
      wrong -> "correct", hiding errors from the calibrator). This DOES break
      validity: the controller under-estimates the true error, certifies too
      permissive a threshold, and coverage collapses (measured ~0.60 at eta=0.20,
      target 0.90). This is the genuine threat and a real limitation of A1.

  (3) MITIGATION for the adversarial case: calibrating at the deflated budget
      alpha - eta restores coverage >= 1 - delta. It is a *blunt* fallback -- it
      buys safety by shrinking the certifiable budget (often to escalate-all), so
      it trades automation for a validity floor when calibration integrity cannot
      be trusted. The graceful fix (multi-labeler agreement to shrink effective
      eta) is a design option discussed in the paper, not coded here.

Numbers verified in-session; small sizes keep the test fast + deterministic.
"""

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
    """Honest labeler error costs automation, not safety: coverage stays >= 1-delta."""
    cov = _coverage(_sym_flip, calibrate_at=_ALPHA, eta=_ETA, seed0=10)
    assert cov >= (1.0 - _DELTA) - 0.03, f"symmetric-noise coverage {cov} dropped below target"


def test_adversarial_calibration_poisoning_breaks_validity():
    """The real threat: one-sided (error-hiding) poisoning UNDER-covers.
    Documents the limitation so it can't silently regress into a false 'we're fine'."""
    cov = _coverage(_adv_flip, calibrate_at=_ALPHA, eta=_ETA, seed0=10)
    assert cov < (1.0 - _DELTA) - 0.05, (
        f"expected adversarial poisoning to break coverage (got {cov}); if this "
        f"now passes, the threat model or bound changed -- revisit the analysis")


def test_deflation_restores_validity_under_adversarial_poisoning():
    """Mitigation: calibrating at alpha - eta restores the validity floor."""
    cov = _coverage(_adv_flip, calibrate_at=_ALPHA - _ETA, eta=_ETA, seed0=10)
    assert cov >= (1.0 - _DELTA) - 0.03, f"deflated coverage {cov} did not restore validity"

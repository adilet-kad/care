"""Monte-Carlo validation of the two robustness theorems, exercised through the
REAL pipeline (bench.study.evaluate_split: controller calibrate/decide + escalator
route + scoring), not a re-implementation.

Theorem 1 (detectable corruption -> retained validity). Poison attributed to a
LOW-trust source is removed by the escalation gate, so the auto-applied set is
uncontaminated (eps_S = 0) and realized error <= alpha w.p. >= 1 - delta.

Theorem 2 (undetectable corruption -> graceful degradation). Poison attributed to
a HIGH-trust source passes the gate and can be auto-applied; realized error is then
bounded by  alpha + eps_S * (1 - alpha)  w.p. >= 1 - delta, where eps_S is the
corrupted fraction of the auto-applied set (Scores.applied_contamination).

We check the guarantee the honest way: the FRACTION of independent trials whose
per-trial bound holds must be >= 1 - delta (with a small MC slack).
"""

from __future__ import annotations

import random

from bench.study import evaluate_split
from care.core.repair import RepairCandidate


def _synthetic_repairs(rng, n):
    """A mixture of reliable (high-agreement, usually correct) and unreliable
    (low-agreement, often wrong) cells, as RepairCandidates + a gold map."""
    repairs, gold = {}, {}
    for i in range(n):
        ref = f"{i}::col"
        if rng.random() < 0.6:                       # reliable
            s = rng.uniform(0.82, 1.0)
            correct = rng.random() < 0.95
        else:                                        # unreliable
            s = rng.uniform(0.0, 0.6)
            correct = rng.random() < 0.55
        gold[ref] = "RIGHT"
        proposed = "RIGHT" if correct else "WRONG"
        repairs[ref] = RepairCandidate(target_ref=ref, proposed_value=proposed,
                                       evidence=["reference"], s_agree=s, s_hat=s)
    return repairs, gold


def _coverage_and_bound(*, trusted, alpha, delta, trials, n, seed0):
    ok_thm = 0            # per-trial theorem bound holds
    covered_plain = 0     # realized error <= alpha (Thm 1 form)
    for t in range(trials):
        rng = random.Random(seed0 + t)
        repairs, gold = _synthetic_repairs(rng, n)
        sc = evaluate_split(repairs, gold, alpha=alpha, delta=delta, seed=seed0 + t,
                            poison_frac=0.2, trusted_poison=trusted)["CARE"]
        eps_s = sc.applied_contamination
        bound = alpha + eps_s * (1.0 - alpha)
        if sc.realized_error <= bound + 1e-9:
            ok_thm += 1
        if sc.realized_error <= alpha + 1e-9:
            covered_plain += 1
    return ok_thm / trials, covered_plain / trials


def test_theorem1_low_trust_poison_retains_validity():
    # detectable poison: eps_S ~ 0, realized error <= alpha w.p. >= 1 - delta
    frac_covered, plain = _coverage_and_bound(
        trusted=False, alpha=0.1, delta=0.1, trials=200, n=500, seed0=300_000)
    assert plain >= (1.0 - 0.1) - 0.03, plain          # plain alpha-coverage holds
    assert frac_covered >= (1.0 - 0.1) - 0.03, frac_covered


def test_theorem2_trusted_poison_within_degradation_bound():
    # undetectable poison: realized error <= alpha + eps_S(1-alpha) w.p. >= 1 - delta
    frac_within, _plain = _coverage_and_bound(
        trusted=True, alpha=0.1, delta=0.1, trials=200, n=500, seed0=400_000)
    assert frac_within >= (1.0 - 0.1) - 0.03, frac_within

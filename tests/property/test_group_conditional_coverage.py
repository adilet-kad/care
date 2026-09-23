"""Group-conditional (Mondrian) coverage.

Marginal control only bounds the *pooled* error: under heterogeneity a single
threshold can satisfy the overall budget while quietly violating a hard subgroup
(the easy majority's low error masks it). Group-conditional (Mondrian) control
calibrates one threshold per stratum, giving a per-stratum guarantee.

We validate both halves of that claim by Monte-Carlo over a heterogeneous data
process (a reliable "easy" stratum and a less-reliable "hard" stratum sharing the
same s_hat range):

  * group-conditional -> EACH stratum's realized error <= alpha at rate >= 1-delta;
  * marginal -> the hard stratum is under-covered (the motivation for Mondrian control).

Seeds are fixed for reproducibility.
"""

from __future__ import annotations

import random


from care.conformal import ConformalController
from care.core import CalibrationRecord

STRATA = ("easy", "hard")


def _hetero(rng: random.Random, n: int):
    out = []
    for _ in range(n):
        s = rng.uniform(0.6, 1.0)
        if rng.random() < 0.7:  # easy: reliable regardless of s
            g, p = "easy", 0.98
        else:  # hard: only high-confidence repairs are reliable
            g, p = "hard", 0.5 + 0.45 * s
        out.append((s, rng.random() < p, g))
    return out


def _per_stratum_coverage(ctrl, alpha, delta, *, grouped, trials, n_cal, n_test, seed0):
    covered = {g: 0 for g in STRATA}
    seen = {g: 0 for g in STRATA}
    for t in range(trials):
        rng = random.Random(seed0 + t)
        cal, test = _hetero(rng, n_cal), _hetero(rng, n_test)
        recs = [
            CalibrationRecord(
                target_ref=f"c{i}", s_hat=s, correct=c,
                stratum=(g if grouped else "default"),
            )
            for i, (s, c, g) in enumerate(cal)
        ]
        thr = ctrl.calibrate(recs, alpha=alpha, delta=delta)
        for g in STRATA:
            lam = thr.threshold_for(g if grouped else "default")
            sel = [c for s, c, gg in test if gg == g and s >= lam]
            seen[g] += 1
            err = (sum(1 for c in sel if not c) / len(sel)) if sel else 0.0
            if err <= alpha:
                covered[g] += 1
    return {g: covered[g] / seen[g] for g in STRATA}


def test_group_conditional_covers_every_stratum():
    ctrl = ConformalController(method="rcps", bound="exact")
    cov = _per_stratum_coverage(
        ctrl, 0.1, 0.1, grouped=True, trials=50, n_cal=1500, n_test=800, seed0=20_000
    )
    for g in STRATA:
        assert cov[g] >= (1.0 - 0.1) - 0.03, (g, cov[g])


def test_marginal_undercovers_the_hard_stratum():
    ctrl = ConformalController(method="rcps", bound="exact")
    cov = _per_stratum_coverage(
        ctrl, 0.1, 0.1, grouped=False, trials=50, n_cal=1500, n_test=800, seed0=30_000
    )
    # the pooled budget is met, but the hard subgroup is clearly under-covered
    assert cov["hard"] < 0.8, cov["hard"]

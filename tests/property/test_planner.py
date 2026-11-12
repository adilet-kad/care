"""The label planner's closed forms must agree with the procedure they model."""
from __future__ import annotations

import random

import pytest

from care.conformal.planner import (best_labels, certify_kmax, expected_saving,
                                    min_labels, p_certify, p_certify_beta,
                                    p_certify_finite, plan)
from care.conformal.rcps import rcps_threshold

DELTA = 0.1


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_kmax_is_the_certifier_boundary(alpha):
    """For a constant score, rcps admits iff #wrong <= k_max(n)."""
    for n in [10, 24, 25, 51, 52, 80, 105, 106, 150, 300]:
        kmax = certify_kmax(n, alpha, DELTA)
        for k in range(0, min(n, int(alpha * n) + 3) + 1):
            lam, _ = rcps_threshold([1.0] * n, [i >= k for i in range(n)],
                                    alpha=alpha, delta=DELTA, bound="exact")
            assert (lam != float("inf")) == (k <= kmax), (n, k, kmax)


def test_min_labels():
    assert [min_labels(a, DELTA) for a in (0.05, 0.1, 0.2)] == [105, 51, 24]


@pytest.mark.parametrize("seed", range(5))
def test_p_certify_matches_simulation(seed):
    rng = random.Random(seed)
    alpha = rng.choice([0.05, 0.1, 0.2])
    n = rng.randint(min_labels(alpha, DELTA), 400)
    p = rng.uniform(0, alpha)
    kmax = certify_kmax(n, alpha, DELTA)
    trials = 4000
    hits = sum(sum(rng.random() < p for _ in range(n)) <= kmax for _ in range(trials))
    assert abs(hits / trials - p_certify(n, p, alpha, DELTA)) < 0.03


@pytest.mark.parametrize("seed", range(5))
def test_finite_matches_split_simulation(seed):
    """Hypergeometric form == drawing n of m without replacement (evaluate_split)."""
    rng = random.Random(100 + seed)
    alpha = 0.1
    m = rng.randint(80, 600)
    wrong = rng.randint(0, m // 10)
    n = max(min_labels(alpha, DELTA), int(0.4 * m))
    if n > m:
        pytest.skip("stratum below the minimum")
    pool = [True] * wrong + [False] * (m - wrong)
    kmax = certify_kmax(n, alpha, DELTA)
    trials = 4000
    hits = sum(sum(rng.sample(pool, n)) <= kmax for _ in range(trials))
    assert abs(hits / trials - p_certify_finite(n, m, wrong, alpha, DELTA)) < 0.03


def test_beta_concentrates_on_binomial():
    n, p = 200, 0.03
    exact = p_certify(n, p, 0.1, DELTA)
    tight = p_certify_beta(n, 1 + p * 1e5, 1 + (1 - p) * 1e5, 0.1, DELTA)
    assert abs(exact - tight) < 0.01


def test_saving_identity_and_kappa_one_never_loses():
    for m, p in [(60, 0.0), (200, 0.05), (1000, 0.08), (5000, 0.02)]:
        n, s = best_labels(m, p, alpha=0.1, delta=DELTA, kappa=1.0)
        assert s >= 0
        if n:
            pc = p_certify(n, p, 0.1, DELTA)
            assert abs(s - expected_saving(m, n, p, alpha=0.1, delta=DELTA, kappa=1.0,
                                           pcert=pc)) < 1e-9
            assert abs(s - (m - n) * pc) < 1e-9


def test_pooling_rescues_thin_strata():
    """Ten thin, clean strata each lose at kappa=2; pooled they pay."""
    strata = {f"c{i}": (60, 0.01) for i in range(10)}
    no_pool = plan(strata, alpha=0.1, delta=DELTA, kappa=2.0, pool=False)
    pooled = plan(strata, alpha=0.1, delta=DELTA, kappa=2.0, pool=True)
    assert no_pool.saving == 0 and all(u.action == "escalate" for u in no_pool.units)
    assert pooled.saving > 0
    assert sum(u.m for u in pooled.units) == 600


def test_plan_escalates_what_cannot_pay():
    p = plan({"bad": (100_000, 0.5)}, alpha=0.1, delta=DELTA, kappa=2.0)
    assert p.units[0].action == "escalate" and p.saving == 0


def test_kmax_matches_clopper_pearson_everywhere():
    from care.conformal.bounds import clopper_pearson_upper
    for alpha in (0.05, 0.1, 0.2):
        dc = DELTA / 21
        for n in range(1, 1200, 7):
            kmax = certify_kmax(n, alpha, DELTA)
            if kmax >= 0:
                assert clopper_pearson_upper(kmax, n, dc) <= alpha
            if kmax + 1 < n:
                assert clopper_pearson_upper(kmax + 1, n, dc) > alpha

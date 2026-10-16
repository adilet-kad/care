"""Risk control of the threshold procedures in care.conformal.rcps.

A repair with score s ~ U[0, 1] is wrong w.p. (1 - s) / 2, so the error above lambda is (1 - lambda) / 4.
"""
import math
import random

import pytest

from care.conformal.rcps import rcps_threshold


def _risk(lam):
    return 0.0 if math.isinf(lam) else (1.0 - lam) / 4.0


@pytest.mark.parametrize("procedure", ["bonferroni", "fixed_sequence", "sgr"])
def test_procedure_controls_risk(procedure):
    alpha, delta, trials, n = 0.1, 0.1, 300, 150
    rng = random.Random(7)
    bad = 0
    for _ in range(trials):
        s = [rng.random() for _ in range(n)]
        ok = [rng.random() >= (1 - x) / 2 for x in s]
        lam, _ = rcps_threshold(s, ok, alpha=alpha, delta=delta, bound="exact",
                                procedure=procedure)
        bad += _risk(lam) > alpha + 1e-12
    assert bad / trials <= delta + 0.05


def test_constant_score_single_test():
    # 60 error-free labels certify alpha = 0.05 with one test at delta = 0.1
    # (needs 45), but not under the 21-way Bonferroni correction (needs 105).
    s, ok = [0.25] * 60, [True] * 60
    one, _ = rcps_threshold(s, ok, alpha=0.05, delta=0.1, bound="exact", grid=[0.25])
    many, _ = rcps_threshold(s, ok, alpha=0.05, delta=0.1, bound="exact")
    assert one == 0.25 and math.isinf(many)


def test_unknown_procedure_rejected():
    with pytest.raises(ValueError):
        rcps_threshold([0.5], [True], alpha=0.1, delta=0.1, procedure="holm")


def test_wrong_labels_only_raise_the_bound():
    # adding wrong labels never lowers the Clopper-Pearson upper bound
    from care.conformal.bounds import clopper_pearson_upper as U
    for n in (5, 20, 80, 300):
        for k in range(0, n, max(1, n // 7)):
            for m in (1, 3, 10, 50):
                assert U(k + m, n + m, 0.01) >= U(k, n, 0.01) - 1e-9

"""care.conformal.bounds -- upper confidence bounds on the selective risk.

The selective risk at a threshold is a [0,1]-bounded mean (here the error rate
``k/n`` among the ``n`` auto-applied calibration repairs, ``k`` of them wrong).
A valid ``(1 - delta)`` upper confidence bound ``R^+`` on this mean is the engine
of the RCPS / Learn-then-Test threshold search:

  * ``hoeffding_upper`` -- closed-form, always valid, mildly conservative.
  * ``clopper_pearson_upper`` -- exact binomial (Beta-quantile) upper bound,
    tighter than Hoeffding/Bentkus, computed without SciPy by bisecting the
    binomial CDF in log-space.
"""

from __future__ import annotations

import math
from functools import lru_cache


def hoeffding_upper(k: int, n: int, delta: float) -> float:
    """Hoeffding ``(1 - delta)`` UCB on a [0,1] mean from ``k`` "successes" in ``n``."""
    if n <= 0:
        return 1.0
    rhat = k / n
    return min(1.0, rhat + math.sqrt(math.log(1.0 / delta) / (2.0 * n)))


def _binom_cdf_le(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed stably in log-space."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    logp, logq = math.log(p), math.log1p(-p)
    total = 0.0
    for j in range(0, k + 1):
        logc = math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        total += math.exp(logc + j * logp + (n - j) * logq)
    return min(1.0, total)


@lru_cache(maxsize=100_000)
def clopper_pearson_upper(k: int, n: int, delta: float, *, iters: int = 60) -> float:
    """Exact ``(1 - delta)`` Clopper-Pearson upper bound on a binomial proportion.

    ``p_U`` solves ``P(X <= k; n, p_U) = delta`` (the CDF is decreasing in ``p``),
    found by bisection. ``k = n`` (all wrong) gives 1.0.
    """
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _binom_cdf_le(k, n, mid) > delta:
            lo = mid  # CDF too high -> p_U is larger
        else:
            hi = mid
    return min(1.0, hi)


__all__ = ["hoeffding_upper", "clopper_pearson_upper"]

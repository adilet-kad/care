"""Threshold search over a fixed grid of candidate thresholds (Learn-then-Test).

Finds the smallest lambda_hat such that auto-applying iff ``s_hat >= lambda_hat``
keeps the selective error ``R(lambda) = P(wrong | s_hat >= lambda)`` at most alpha
with probability ``>= 1 - delta``. Each grid point is tested with a
``(1 - delta/m)`` upper confidence bound (Bonferroni over the ``m`` grid points;
Angelopoulos et al. 2021). If none admits, ``lambda_hat = +inf``.
"""

from __future__ import annotations

import bisect
import math
from typing import Callable, Sequence

from care.conformal.bounds import clopper_pearson_upper, hoeffding_upper

_BOUNDS: dict[str, Callable[[int, int, float], float]] = {
    "hoeffding": hoeffding_upper,
    "exact": clopper_pearson_upper,
    "clopper_pearson": clopper_pearson_upper,
}

# default grid: 21 thresholds 0, 0.05, ..., 1.0
_DEFAULT_GRID = 20


def _candidate_grid(grid: int | Sequence[float]) -> list[float]:
    """Fixed candidate thresholds in ascending order."""
    if isinstance(grid, int):
        if grid < 1:
            raise ValueError("grid must be a positive int or a sequence of floats")
        return [i / grid for i in range(grid + 1)]
    return sorted(set(float(x) for x in grid))


def rcps_threshold(
    s_hat: Sequence[float],
    correct: Sequence[bool],
    *,
    alpha: float,
    delta: float,
    bound: str = "hoeffding",
    grid: int | Sequence[float] = _DEFAULT_GRID,
    procedure: str = "bonferroni",
) -> tuple[float, list[dict]]:
    """Return ``(lambda_hat, trace)``; ``trace`` records n, k and the bound per threshold.

    ``grid``: an int G gives ``0, 1/G, ..., 1``; a sequence is used as given.
    It must not depend on the calibration data.
    """
    if bound not in _BOUNDS:
        raise ValueError(f"unknown bound {bound!r}; choose from {sorted(_BOUNDS)}")
    if procedure not in _PROCEDURES:
        raise ValueError(f"unknown procedure {procedure!r}; choose from {_PROCEDURES}")
    ucb = _BOUNDS[bound]

    pairs = sorted(zip(s_hat, correct), key=lambda t: t[0])  # ascending s_hat
    s_sorted = [s for s, _ in pairs]
    total = len(pairs)
    suffix_wrong = [0] * (total + 1)  # suffix_wrong[i] = #wrong in pairs[i:]
    for i in range(total - 1, -1, -1):
        suffix_wrong[i] = suffix_wrong[i + 1] + (0 if pairs[i][1] else 1)

    if procedure == "sgr":
        return _sgr(pairs, suffix_wrong, alpha, delta, ucb)
    candidate_lambdas = _candidate_grid(grid)
    if procedure == "fixed_sequence":
        return _fixed_sequence(s_sorted, suffix_wrong, candidate_lambdas, alpha, delta, ucb)
    m = max(1, len(candidate_lambdas))
    # Bonferroni over the m grid points
    delta_c = delta / m

    lambda_hat = math.inf
    trace: list[dict] = []
    for lam in candidate_lambdas:  # ascending -> most permissive first
        i = bisect.bisect_left(s_sorted, lam)
        n = total - i
        k = suffix_wrong[i]
        if n == 0 or k / n > alpha:
            continue  # empirical risk > alpha, cannot admit
        r_ucb = ucb(k, n, delta_c)
        trace.append({"lambda": lam, "n": n, "k": k, "rhat": k / n, "ucb": r_ucb})
        if r_ucb <= alpha:
            lambda_hat = lam  # smallest admitted lambda -> maximal automation
            break
    return lambda_hat, trace


_PROCEDURES = ("bonferroni", "fixed_sequence", "sgr")


def _fixed_sequence(s_sorted, suffix_wrong, lambdas, alpha, delta, ucb):
    """Fixed-sequence testing: from the highest threshold down, each at level delta,
    stopping at the first that fails. Returns the last admitted threshold or +inf."""
    total = len(s_sorted)
    lambda_hat, trace = math.inf, []
    for lam in sorted(lambdas, reverse=True):
        i = bisect.bisect_left(s_sorted, lam)
        n, k = total - i, suffix_wrong[i]
        r_ucb = ucb(k, n, delta) if n > 0 else 1.0
        trace.append({"lambda": lam, "n": n, "k": k, "ucb": r_ucb})
        if r_ucb > alpha:
            break
        lambda_hat = lam
    return lambda_hat, trace


def _sgr(pairs, suffix_wrong, alpha, delta, ucb):
    """Selection with Guaranteed Risk (Geifman and El-Yaniv, 2017): binary search
    over the calibration scores, each test at ``delta / ceil(log2 n)``."""
    n_all = len(pairs)
    if n_all == 0:
        return math.inf, []
    steps = max(1, math.ceil(math.log2(n_all)))
    dc = delta / steps
    lo, hi = 0, n_all - 1          # indices into ascending scores
    best, trace = math.inf, []
    for _ in range(steps):
        mid = (lo + hi) // 2
        lam = pairs[mid][0]
        i = bisect.bisect_left([p[0] for p in pairs], lam)
        n, k = n_all - i, suffix_wrong[i]
        r_ucb = ucb(k, n, dc)
        trace.append({"lambda": lam, "n": n, "k": k, "ucb": r_ucb})
        if r_ucb <= alpha:
            best = min(best, lam)
            hi = mid - 1
        else:
            lo = mid + 1
        if lo > hi:
            break
    return best, trace


__all__ = ["rcps_threshold"]

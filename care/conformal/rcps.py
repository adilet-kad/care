"""care.conformal.rcps -- Risk-Controlling Prediction Sets threshold search.

Given calibration repairs with confidence ``s_hat`` and a binary ``correct``
label, find the auto-apply threshold lambda_hat so that auto-applying iff
``s_hat >= lambda_hat`` controls the selective error
``R(lambda) = E[wrong | s_hat >= lambda]`` at ``R(lambda_hat) <= alpha`` with
probability ``>= 1 - delta``.

Construction (multiplicity-honest). We search a **fixed, data-independent grid**
of candidate thresholds ``Lambda`` (default: ``0, 1/G, ..., 1``). Each candidate
``lambda`` is a hypothesis ``H_lambda: R(lambda) > alpha``, certified when its
``(1 - delta')`` upper confidence bound ``R^+`` (``bounds.py``) on the selected
set's error is ``<= alpha``. Because the threshold is *selected from the data*
among ``m = |Lambda|`` candidates, we control the family-wise error with a
Bonferroni level ``delta' = delta / m`` (Learn-then-Test, Angelopoulos et al.
2021). Among the certified thresholds we return the smallest -- the largest
auto-applied set, i.e. maximal automation. If none certifies, ``lambda_hat =
+inf`` and everything escalates.

Why the fixed grid + correction (and not the naive "smallest data value whose
pointwise UCB <= alpha"). Selecting a threshold from *all distinct calibration
s_hat values* and testing each with a pointwise UCB is a multiple-comparisons
procedure with no correction: in simulation it UNDER-COVERS at the nominal
1-delta where the fixed grid does not (``tests/property`` asserts this). A fixed
coarse grid keeps ``m`` small so the Bonferroni penalty is mild, restores valid
coverage, and still auto-applies a real fraction. The genuine RCPS
tail-monotonicity argument (no union bound) requires certifying the *entire* tail
``lambda' >= lambda``; on a data-dependent grid the extreme-high thresholds have
tiny support and unbounded CIs, which collapses the tail rule to escalate-all.
The fixed-grid LTT view is the honest, robust choice.

Validity rests on the pointwise-valid UCB at each grid point, the Bonferroni
union bound over the fixed grid, and exchangeability of calibration and test
repairs (heterogeneity is handled per stratum by the controller's strata_fn).
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

# Default fixed grid granularity: 21 thresholds at 0, 0.05, ..., 1.0. Fine enough
# to sit a threshold near any reliable/unreliable confidence boundary, coarse
# enough that the Bonferroni penalty delta/21 barely dents the bound.
_DEFAULT_GRID = 20


def _candidate_grid(grid: int | Sequence[float]) -> list[float]:
    """A fixed, data-independent grid of candidate thresholds in ascending order."""
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
) -> tuple[float, list[dict]]:
    """Return ``(lambda_hat, trace)``; ``trace`` records each evaluated n/k/UCB.

    ``grid`` -- fixed candidate thresholds. An ``int`` G yields ``0, 1/G, ..., 1``;
    a sequence is used verbatim. Kept data-independent so the Bonferroni family
    size ``m`` is fixed and the coverage guarantee is honest.
    """
    if bound not in _BOUNDS:
        raise ValueError(f"unknown bound {bound!r}; choose from {sorted(_BOUNDS)}")
    ucb = _BOUNDS[bound]

    pairs = sorted(zip(s_hat, correct), key=lambda t: t[0])  # ascending s_hat
    s_sorted = [s for s, _ in pairs]
    total = len(pairs)
    suffix_wrong = [0] * (total + 1)  # suffix_wrong[i] = #wrong in pairs[i:]
    for i in range(total - 1, -1, -1):
        suffix_wrong[i] = suffix_wrong[i + 1] + (0 if pairs[i][1] else 1)

    candidate_lambdas = _candidate_grid(grid)
    m = max(1, len(candidate_lambdas))
    # Fixed-grid Learn-then-Test: the threshold is chosen from m data-independent
    # candidates, so a delta/m union bound controls the family-wise error. This is
    # exactly the correction that makes the selected-threshold guarantee valid; it
    # is *not* optional under the tight bound (see module docstring).
    delta_c = delta / m

    lambda_hat = math.inf
    trace: list[dict] = []
    for lam in candidate_lambdas:  # ascending -> most permissive first
        i = bisect.bisect_left(s_sorted, lam)
        n = total - i
        k = suffix_wrong[i]
        if n == 0 or k / n > alpha:
            continue  # empirical risk already > alpha -> UCB > alpha, cannot certify
        r_ucb = ucb(k, n, delta_c)
        trace.append({"lambda": lam, "n": n, "k": k, "rhat": k / n, "ucb": r_ucb})
        if r_ucb <= alpha:
            lambda_hat = lam  # smallest certified lambda -> maximal automation
            break
    return lambda_hat, trace


__all__ = ["rcps_threshold"]

"""bench.stats -- confidence intervals and seed aggregation for the paper tables."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Agg:
    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def bootstrap_ci(values: list[float], *, iters: int = 2000, seed: int = 0, alpha: float = 0.05) -> Agg:
    """Percentile bootstrap CI over per-seed values."""
    if not values:
        return Agg(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return Agg(mean=sum(values) / n, lo=lo, hi=hi, n=n)


__all__ = ["bootstrap_ci", "Agg"]

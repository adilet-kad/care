"""Label-budget planner: how many calibration labels to buy per stratum.

For a stratum with a single confidence value (an unranked cleaner) the
certification probability has a closed form:

    k_max(n)      = largest k with CP_upper(k, n; delta/|Lambda|) <= alpha
    P_cert(n, p)  = P(Binomial(n, p) <= k_max(n))

With m repairs in the stratum and kappa reviews per label, the expected saving
in reviews is

    S(n) = (m - n) * P_cert(n, p) - (kappa - 1) * n.

For ranked cleaners the plan is conservative. The error-rate belief p must come
from a prior or a pilot sample disjoint from the calibration labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache


GRID_SIZE = 21          # |Lambda| of the default fixed grid 0, 1/20, ..., 1


# --------------------------------------------------------------------------- #
# Certification probability                                                    #
# --------------------------------------------------------------------------- #

def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


@lru_cache(maxsize=200_000)
def certify_kmax(n: int, alpha: float, delta: float, grid_size: int = GRID_SIZE) -> int:
    """Largest number of wrong calibration repairs out of n that still certifies.

    Uses CP_upper(k, n; d) <= alpha  iff  P(Binomial(n, alpha) <= k) <= d, so one
    cumulative sum suffices. Returns -1 when even k = 0 fails.
    """
    if n <= 0:
        return -1
    dc = delta / grid_size
    la, lq = math.log(alpha), math.log1p(-alpha)
    cdf, k = 0.0, -1
    while k + 1 < n:
        j = k + 1
        cdf += math.exp(_log_binom(n, j) + j * la + (n - j) * lq)
        if cdf > dc:
            break
        k = j
    return k


def min_labels(alpha: float, delta: float, grid_size: int = GRID_SIZE) -> int:
    """Smallest n at which a stratum with zero observed errors certifies."""
    n = 1
    while certify_kmax(n, alpha, delta, grid_size) < 0:
        n += 1
    return n


def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    log_t = n * lq                                     # log P(X = 0)
    r = lp - lq
    total = math.exp(log_t)
    for j in range(1, k + 1):
        log_t += math.log((n - j + 1) / j) + r
        total += math.exp(log_t)
    return min(1.0, total)


def p_certify(n: int, p: float, alpha: float, delta: float,
              grid_size: int = GRID_SIZE) -> float:
    """P(an unranked stratum with error rate p certifies from n labels)."""
    return _binom_cdf(certify_kmax(n, alpha, delta, grid_size), n, p)


def p_certify_finite(n: int, m: int, wrong: int, alpha: float, delta: float,
                     grid_size: int = GRID_SIZE) -> float:
    """As ``p_certify``, with n labels drawn without replacement from m repairs of
    which ``wrong`` are wrong (hypergeometric)."""
    kmax = certify_kmax(n, alpha, delta, grid_size)
    if kmax < 0 or n > m:
        return 0.0
    tot = _log_binom(m, n)
    s = 0.0
    for k in range(max(0, n - (m - wrong)), min(kmax, wrong, n) + 1):
        s += math.exp(_log_binom(wrong, k) + _log_binom(m - wrong, n - k) - tot)
    return min(1.0, s)


def p_certify_beta(n: int, a: float, b: float, alpha: float, delta: float,
                   grid_size: int = GRID_SIZE) -> float:
    """P_cert averaged over a Beta(a, b) belief about p (beta-binomial).

    Use a = 1 + wrong, b = 1 + right from a pilot sample disjoint from calibration.
    """
    kmax = certify_kmax(n, alpha, delta, grid_size)
    if kmax < 0:
        return 0.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    s = 0.0
    for k in range(kmax + 1):
        s += math.exp(_log_binom(n, k) + math.lgamma(k + a) + math.lgamma(n - k + b)
                      - math.lgamma(n + a + b) - lbeta)
    return min(1.0, s)


# --------------------------------------------------------------------------- #
# Economics of one stratum                                                     #
# --------------------------------------------------------------------------- #

def expected_saving(m: int, n: int, p: float, *, alpha: float, delta: float,
                    kappa: float = 1.0, grid_size: int = GRID_SIZE,
                    pcert: float | None = None) -> float:
    """Expected reviews saved versus reviewing all m cells: (m-n)P - (kappa-1)n."""
    if n <= 0 or n >= m:
        return 0.0 if n <= 0 else -(kappa - 1.0) * m
    pc = p_certify(n, p, alpha, delta, grid_size) if pcert is None else pcert
    return (m - n) * pc - (kappa - 1.0) * n


@lru_cache(maxsize=None)
def _n_grid(n0: int, m: int) -> tuple:
    """Candidate label counts: every n up to 300, then 2% steps."""
    out, n = [], n0
    while n < m:
        out.append(n)
        n = n + 1 if n < 300 else int(n * 1.02) + 1
    return tuple(out)


def best_labels(m: int, p, *, alpha: float, delta: float, kappa: float = 1.0,
                grid_size: int = GRID_SIZE) -> tuple[int, float]:
    """(n*, S(n*)) maximising the expected saving; (0, 0.0) means escalate.

    ``p`` is a planning error rate or a Beta belief ``(a, b)`` (e.g. from a pilot
    sample: a = 1 + wrong, b = 1 + right). Exhaustive up to 300 labels; beyond, a
    2% grid followed by an exhaustive search between the best point's neighbours.
    """
    if m <= 1:
        return 0, 0.0
    n0 = min_labels(alpha, delta, grid_size)

    def S(n):
        return expected_saving(m, n, p, alpha=alpha, delta=delta, kappa=kappa,
                               grid_size=grid_size,
                               pcert=_pcert_cached(n, p, alpha, delta, grid_size))
    grid = _n_grid(n0, m)
    best_n, best_s, best_i = 0, 0.0, -1
    for i, n in enumerate(grid):
        s = S(n)
        if s > best_s:
            best_n, best_s, best_i = n, s, i
    if best_n > 300:
        lo = grid[best_i - 1] + 1
        hi = grid[best_i + 1] if best_i + 1 < len(grid) else m
        for n in range(lo, hi):
            s = S(n)
            if s > best_s:
                best_n, best_s = n, s
    return best_n, best_s


@lru_cache(maxsize=500_000)
def _pcert_cached(n, p, alpha, delta, grid_size):
    """``p`` is a planning error rate (float) or a Beta belief ``(a, b)``."""
    if isinstance(p, tuple):
        return p_certify_beta(n, p[0], p[1], alpha, delta, grid_size)
    return p_certify(n, p, alpha, delta, grid_size)


def _mean(p) -> float:
    return p[0] / (p[0] + p[1]) if isinstance(p, tuple) else float(p)


def _pool(items):
    """Pool (m, belief) pairs: weighted mean for rates, summed pseudo-counts for Betas."""
    kinds = {isinstance(p, tuple) for _, p in items}
    if len(kinds) > 1:
        raise ValueError("pool either rates or Beta beliefs, not a mix")
    if kinds == {True}:
        a = 1 + sum(p[0] - 1 for _, p in items)
        b = 1 + sum(p[1] - 1 for _, p in items)
        return (a, b)
    m = sum(m for m, _ in items)
    return sum(m * p for m, p in items) / m


def break_even_size(p: float, *, alpha: float, delta: float, kappa: float,
                    grid_size: int = GRID_SIZE, m_max: int = 1_000_000) -> int | None:
    """Smallest stratum size m with positive expected saving (meaningful for kappa > 1).

    Returns None if no m up to ``m_max`` breaks even.
    """
    def pays(m):
        return best_labels(m, p, alpha=alpha, delta=delta, kappa=kappa,
                           grid_size=grid_size)[1] > 0
    lo = min_labels(alpha, delta, grid_size) + 1
    hi = lo
    while not pays(hi):
        hi *= 2
        if hi > m_max:
            return None
    while lo < hi:                                     # saving is ~monotone in m
        mid = (lo + hi) // 2
        if pays(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


# --------------------------------------------------------------------------- #
# The plan                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class StratumPlan:
    strata: tuple            # the original strata this unit covers (one, or a pool)
    m: int                   # repairs in the unit
    p: float                 # planning error rate (mean of the belief)
    action: str              # "calibrate" | "escalate"
    labels: int              # labels to buy (0 if escalated)
    p_certify: float
    saving: float            # expected reviews saved


@dataclass
class Plan:
    units: list[StratumPlan] = field(default_factory=list)

    @property
    def m(self) -> int:
        return sum(u.m for u in self.units)

    @property
    def labels(self) -> int:
        return sum(u.labels for u in self.units)

    @property
    def saving(self) -> float:
        return sum(u.saving for u in self.units)

    @property
    def expected_auto(self) -> float:
        return sum((u.m - u.labels) * u.p_certify for u in self.units)

    def strata_fn(self, base_fn):
        """Wrap a stratifier so pooled strata map to one pool label."""
        alias = {}
        for i, u in enumerate(self.units):
            if len(u.strata) > 1:
                for s in u.strata:
                    alias[s] = f"pool:{i}"
        return lambda repair: alias.get(base_fn(repair), base_fn(repair))


def _unit(strata, m, p, alpha, delta, kappa, grid_size):
    n, sv = best_labels(m, p, alpha=alpha, delta=delta, kappa=kappa, grid_size=grid_size)
    if n == 0:
        return StratumPlan(tuple(strata), m, _mean(p), "escalate", 0, 0.0, 0.0)
    return StratumPlan(tuple(strata), m, _mean(p), "calibrate", n,
                       _pcert_cached(n, p, alpha, delta, grid_size), sv)


def plan(strata: dict, *, alpha: float, delta: float, kappa: float = 1.0,
         grid_size: int = GRID_SIZE, pool: bool = True) -> Plan:
    """Decide, per stratum, calibrate(n*) / pool / escalate.

    ``strata`` maps a stratum name to ``(m, p)``: its number of repairs and an
    error-rate belief (a rate, or a Beta ``(a, b)`` from a pilot sample not reused
    for calibration).

    Two partitions are costed and the better returned: keep every stratum that pays
    for itself and pool the rest (best prefix by error rate), or pool every eligible
    stratum. Strata believed to be at or above alpha are never pooled. Anything not
    calibrated is escalated. A pool's guarantee holds over the pool jointly.
    """
    beliefs = {nm: (int(m), tuple(p) if isinstance(p, (tuple, list)) else float(p))
               for nm, (m, p) in strata.items()}
    single = {nm: _unit([nm], m, p, alpha, delta, kappa, grid_size)
              for nm, (m, p) in beliefs.items()}
    winners = {nm for nm, u in single.items() if u.action == "calibrate" and u.saving > 0}
    eligible = {nm for nm, (_, p) in beliefs.items() if _mean(p) < alpha}

    def pooled(names):
        names = sorted(names, key=lambda nm: _mean(beliefs[nm][1]))
        items = [beliefs[nm] for nm in names]
        return _unit(names, sum(m for m, _ in items), _pool(items), alpha, delta, kappa,
                     grid_size)

    def finish(units):
        used = {s for u in units for s in u.strata}
        out = Plan(list(units))
        out.units += [StratumPlan((nm,), m, _mean(p), "escalate", 0, 0.0, 0.0)
                      for nm, (m, p) in beliefs.items() if nm not in used]
        return out

    # candidate 1: winners alone, best prefix-pool of eligible losers
    units = [single[nm] for nm in winners]
    losers = sorted((nm for nm in eligible if nm not in winners),
                    key=lambda nm: _mean(beliefs[nm][1]))
    if pool and losers:
        best = None
        for i in range(1, len(losers) + 1):
            u = pooled(losers[:i])
            if best is None or u.saving > best.saving:
                best = u
        if best.saving > 0:
            units.append(best)
    cand1 = finish(units)
    if not pool or len(eligible) < 2:
        return cand1
    # candidate 2: one pool of every eligible stratum
    u = pooled(eligible)
    units = [single[nm] for nm in winners if nm not in eligible]
    if u.saving > 0:
        units.append(u)
    cand2 = finish(units)
    return cand2 if cand2.saving > cand1.saving else cand1


__all__ = ["GRID_SIZE", "certify_kmax", "min_labels", "p_certify", "p_certify_finite",
           "p_certify_beta", "expected_saving", "best_labels", "break_even_size",
           "StratumPlan", "Plan", "plan"]

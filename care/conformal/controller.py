"""Conformal controller: calibrates per-stratum thresholds and makes decisions.

A repair is auto-applied iff ``s_hat >= lambda_hat[stratum]``; calibration
bounds ``R(lambda_hat) <= alpha`` with probability ``>= 1 - delta``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable, Sequence

from care.core.repair import (
    CalibrationRecord,
    Decision,
    RepairCandidate,
    ThresholdTable,
)
from care.conformal.combine import Combiner, MeanCombiner
from care.conformal.rcps import rcps_threshold
from care.conformal.strata import DEFAULT_STRATUM, marginal_strata


class ConformalController:
    """Calibrates and applies the conformal auto/escalate threshold."""

    def __init__(
        self,
        combiner: Combiner | None = None,
        *,
        method: str = "rcps",
        bound: str = "hoeffding",
        strata_fn: Callable = marginal_strata,
        grid=None,
        procedure: str = "bonferroni",
    ):
        if method != "rcps":
            raise ValueError("method must be 'rcps'")
        self.combiner = combiner or MeanCombiner()
        self.method = method
        self.bound = bound
        self.strata_fn = strata_fn
        # fixed candidate grid; its size sets the Bonferroni penalty delta/|Lambda|
        self.grid = grid
        # "bonferroni" (default), "fixed_sequence" or "sgr"
        self.procedure = procedure

    # --- s_hat ------------------------------------------------------------- #

    def combine(self, repair: RepairCandidate) -> float:
        return self.combiner.combine(repair.s_conf, repair.s_margin, repair.s_agree, repair.s_consistency)

    def assign_s_hat(self, repairs: Iterable[RepairCandidate]) -> list[RepairCandidate]:
        """Return copies with ``s_hat`` filled from the combiner."""
        return [r.model_copy(update={"s_hat": self.combine(r)}) for r in repairs]

    # --- calibration ------------------------------------------------------- #

    def _threshold(self, s_hat: Sequence[float], correct: Sequence[bool], alpha, delta) -> float:
        kw = {} if self.grid is None else {"grid": self.grid}
        lam, _trace = rcps_threshold(
            s_hat, correct, alpha=alpha, delta=delta, bound=self.bound,
            procedure=self.procedure, **kw
        )
        return lam

    def calibrate(
        self,
        cal: Sequence[CalibrationRecord],
        *,
        alpha: float,
        delta: float,
    ) -> ThresholdTable:
        groups: dict[str, list[CalibrationRecord]] = defaultdict(list)
        for rec in cal:
            groups[rec.stratum].append(rec)

        thresholds: dict[str, float] = {}
        for stratum, recs in groups.items():
            thresholds[stratum] = self._threshold(
                [r.s_hat for r in recs], [r.correct for r in recs], alpha, delta
            )
        # unseen strata use the "default" threshold if calibrated, else +inf
        default = thresholds.get(DEFAULT_STRATUM, math.inf)
        return ThresholdTable(
            alpha=alpha, delta=delta, thresholds=thresholds, default=default
        )

    # --- decision ---------------------------------------------------------- #

    def decide(
        self,
        repairs: Iterable[RepairCandidate],
        thr: ThresholdTable,
        *,
        strata_fn: Callable | None = None,
    ) -> list[Decision]:
        strata_fn = strata_fn or self.strata_fn
        out: list[Decision] = []
        for r in repairs:
            s_hat = r.s_hat if r.s_hat is not None else self.combine(r)
            stratum = strata_fn(r)
            lam = thr.threshold_for(stratum)
            out.append(
                Decision(
                    target_ref=r.target_ref,
                    action="auto_apply" if s_hat >= lam else "escalate",
                    s_hat=s_hat,
                    lambda_hat=lam,
                    stratum=stratum,
                )
            )
        return out


__all__ = ["ConformalController"]

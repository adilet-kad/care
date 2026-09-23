"""care.conformal.combine -- the confidence combiner s_hat = f(s_llm, s_margin, s_agree, s_consistency).

The conformal layer only needs a *usable ranking* plus a calibration set, so the
combiner is not load-bearing for the guarantee: the threshold calibrates whatever
score it is handed. ``MeanCombiner`` (the default) takes the arithmetic mean of
the components that are present (``None`` skipped).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Combiner(Protocol):
    def combine(
        self,
        s_llm: float | None = None,
        s_margin: float | None = None,
        s_agree: float | None = None,
        s_consistency: float | None = None,
    ) -> float: ...


def _present(*values: float | None) -> list[float]:
    return [max(0.0, min(1.0, v)) for v in values if v is not None]


class MeanCombiner:
    """Arithmetic mean of the present (non-None) components."""

    def combine(self, s_llm=None, s_margin=None, s_agree=None, s_consistency=None) -> float:
        vals = _present(s_llm, s_margin, s_agree, s_consistency)
        return sum(vals) / len(vals) if vals else 0.0


__all__ = ["Combiner", "MeanCombiner"]

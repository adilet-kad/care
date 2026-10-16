"""Confidence combiner: s_hat = f(s_conf, s_margin, s_agree, s_consistency).

The threshold calibrates whatever score it receives, so the combiner affects
only the ranking, not the guarantee.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Combiner(Protocol):
    def combine(
        self,
        s_conf: float | None = None,
        s_margin: float | None = None,
        s_agree: float | None = None,
        s_consistency: float | None = None,
    ) -> float: ...


def _present(*values: float | None) -> list[float]:
    return [max(0.0, min(1.0, v)) for v in values if v is not None]


class MeanCombiner:
    """Arithmetic mean of the present (non-None) components."""

    def combine(self, s_conf=None, s_margin=None, s_agree=None, s_consistency=None) -> float:
        vals = _present(s_conf, s_margin, s_agree, s_consistency)
        return sum(vals) / len(vals) if vals else 0.0


__all__ = ["Combiner", "MeanCombiner"]

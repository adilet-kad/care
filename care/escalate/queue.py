"""care.escalate.queue -- the human-review queue.

Repairs the controller does not auto-apply are escalated to humans as
``EscalationItem``s, each tagged with the reason it was escalated: below the
stratum's threshold, an uncontrollable (+inf) stratum, or suspected poison (a
low-trust source backed the repair -- the tau override in ``active.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class EscalationReason(str, Enum):
    BELOW_THRESHOLD = "below_threshold"        # s_hat < lambda_hat for the stratum
    UNCONTROLLABLE = "uncontrollable_stratum"  # stratum threshold is +inf
    SUSPECTED_POISON = "suspected_poison"      # low-trust evidence (tau override)


@dataclass
class EscalationItem:
    target_ref: str
    s_hat: float
    stratum: str
    lambda_hat: float
    reason: EscalationReason
    evidence: list[str] = field(default_factory=list)
    min_trust: float | None = None


class EscalationQueue:
    """The set of human-review items produced by one routing pass."""

    def __init__(self, items: Sequence[EscalationItem]):
        self.items: list[EscalationItem] = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


__all__ = [
    "EscalationReason",
    "EscalationItem",
    "EscalationQueue",
]

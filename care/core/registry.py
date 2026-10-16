"""Constraint plugin registration and the per-run ConstraintRegistry."""

from __future__ import annotations

from typing import Iterable

from care.core.artifact import DQRequirements
from care.core.constraints import BaseConstraint, ConstraintType

# cid -> constraint class, populated at import time by @register_constraint
_REGISTRY: dict[str, type[BaseConstraint]] = {}

_VALID_TYPES: tuple[ConstraintType, ...] = ("H", "W", "P")


def register_constraint(cls: type[BaseConstraint]) -> type[BaseConstraint]:
    """Class decorator that validates and registers a constraint plugin."""
    cid = getattr(cls, "cid", None)
    ctype = getattr(cls, "ctype", None)
    if not cid:
        raise ValueError(f"{cls.__name__} must declare a non-empty 'cid'")
    if not getattr(cls, "source", None):
        raise ValueError(f"{cls.__name__} (cid={cid!r}) must declare a 'source'")
    if ctype not in _VALID_TYPES:
        raise ValueError(
            f"{cls.__name__}.ctype must be one of {_VALID_TYPES}, got {ctype!r}"
        )
    existing = _REGISTRY.get(cid)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Duplicate constraint cid {cid!r}: already registered to {existing.__name__}"
        )
    _REGISTRY[cid] = cls
    return cls


def registered_constraints() -> dict[str, type[BaseConstraint]]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Empty the global registry (for tests)."""
    _REGISTRY.clear()


class ConstraintRegistry:
    """The active constraint instances for one run."""

    def __init__(
        self,
        constraints: Iterable[BaseConstraint],
        requirements: DQRequirements | None = None,
    ):
        self._by_cid: dict[str, BaseConstraint] = {c.cid: c for c in constraints}
        # source of theta_i / w_i for the auditor and verifier
        self.requirements: DQRequirements = (
            requirements if requirements is not None else DQRequirements()
        )

    # --- construction ------------------------------------------------------ #
    @classmethod
    def from_requirements(
        cls, dqr: DQRequirements, *, strict: bool = False
    ) -> "ConstraintRegistry":
        """Instantiate registered plugins enabled in ``dqr``; unknown cids raise if ``strict``."""
        active: list[BaseConstraint] = []
        for cid in dqr.enabled:
            plugin = _REGISTRY.get(cid)
            if plugin is None:
                if strict:
                    raise KeyError(f"Enabled constraint not registered: {cid!r}")
                continue
            active.append(plugin())
        return cls(active, requirements=dqr)

    @classmethod
    def from_registered(cls) -> "ConstraintRegistry":
        """Instantiate every registered plugin, ignoring ``enabled``."""
        return cls(plugin() for plugin in _REGISTRY.values())

    # --- access ------------------------------------------------------------ #
    def all(self) -> list[BaseConstraint]:
        return list(self._by_cid.values())

    def get(self, cid: str) -> BaseConstraint | None:
        return self._by_cid.get(cid)

    def __len__(self) -> int:
        return len(self._by_cid)

    def __contains__(self, cid: object) -> bool:
        return cid in self._by_cid

    # --- typed slices ------------------------------------------------------ #
    def hard(self) -> list[BaseConstraint]:
        return [c for c in self._by_cid.values() if c.ctype == "H"]

    def soft(self) -> list[BaseConstraint]:
        return [c for c in self._by_cid.values() if c.ctype == "W"]

    def process(self) -> list[BaseConstraint]:
        return [c for c in self._by_cid.values() if c.ctype == "P"]

    def by_source(self, src: str) -> list[BaseConstraint]:
        """Constraints whose ``source`` contains ``src`` (case-insensitive)."""
        needle = src.lower()
        return [c for c in self._by_cid.values() if needle in c.source.lower()]


__all__ = [
    "register_constraint",
    "registered_constraints",
    "clear_registry",
    "ConstraintRegistry",
]

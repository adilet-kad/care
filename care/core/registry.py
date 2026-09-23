"""care.core.registry -- the ConstraintRegistry and plugin registration.

Constraint plugins live under ``care.predicates`` and self-register with the
``@register_constraint`` decorator. A ``ConstraintRegistry`` is the *assembled,
active* set for a particular run: it is built from a DQRequirements (only the
``enabled`` cids) and exposes the constraints sliced by type (hard / soft /
process) and by source. Adding a predicate is a matter of dropping a class into
``care/predicates/`` -- the engine is untouched.
"""

from __future__ import annotations

from typing import Iterable

from care.core.artifact import DQRequirements
from care.core.constraints import BaseConstraint, ConstraintType

# Global registry of constraint *classes*, keyed by cid. Populated at import
# time by the @register_constraint decorator on plugins under care/predicates/.
_REGISTRY: dict[str, type[BaseConstraint]] = {}

_VALID_TYPES: tuple[ConstraintType, ...] = ("H", "W", "P")


def register_constraint(cls: type[BaseConstraint]) -> type[BaseConstraint]:
    """Class decorator that registers a constraint plugin.

    Validates the H/W/P typing invariant (I5) and rejects duplicate cids at
    import time, turning a whole class of configuration errors into immediate,
    loud failures.
    """
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
    """A copy of the global cid -> plugin-class map."""
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Empty the global registry (primarily for tests)."""
    _REGISTRY.clear()


class ConstraintRegistry:
    """An assembled, active set of constraint instances for one run."""

    def __init__(
        self,
        constraints: Iterable[BaseConstraint],
        requirements: DQRequirements | None = None,
    ):
        self._by_cid: dict[str, BaseConstraint] = {c.cid: c for c in constraints}
        # The statement of applicability this set was assembled from. Downstream
        # modules (Auditor, Verifier) read theta_i / w_i from here, so the
        # documented run(art, reg) signature needs no extra dqr argument.
        self.requirements: DQRequirements = (
            requirements if requirements is not None else DQRequirements()
        )

    # --- construction ------------------------------------------------------ #
    @classmethod
    def from_requirements(
        cls, dqr: DQRequirements, *, strict: bool = False
    ) -> "ConstraintRegistry":
        """Instantiate the registered plugins whose cid is in ``dqr.enabled``.

        With ``strict=True`` an enabled-but-unregistered cid raises; otherwise it
        is skipped.
        """
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
        """Instantiate every registered plugin (ignores ``enabled`` -- mostly for
        introspection / tests).
        """
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

    # --- typed slices (the H / W / Pi separation) -------------------------- #
    def hard(self) -> list[BaseConstraint]:
        return [c for c in self._by_cid.values() if c.ctype == "H"]

    def soft(self) -> list[BaseConstraint]:
        return [c for c in self._by_cid.values() if c.ctype == "W"]

    def process(self) -> list[BaseConstraint]:
        return [c for c in self._by_cid.values() if c.ctype == "P"]

    def by_source(self, src: str) -> list[BaseConstraint]:
        """Constraints whose ``source`` citation contains ``src`` (case-insensitive)."""
        needle = src.lower()
        return [c for c in self._by_cid.values() if needle in c.source.lower()]


__all__ = [
    "register_constraint",
    "registered_constraints",
    "clear_registry",
    "ConstraintRegistry",
]

"""Shipped data-quality predicates; importing the package registers them.

Only ``dq.completeness.floor`` is enabled in the benchmark. The other predicates
cover different evaluation shapes (cross-row, lineage, sources, metadata) and are
exercised by the projector-invariant property test.
"""

from __future__ import annotations

from care.predicates import dq  # noqa: F401

ALL_CONSTRAINTS = list(dq.PLUGINS)

__all__ = ["ALL_CONSTRAINTS", "dq"]

"""care.predicates -- the data-quality predicates the verifier projects onto.

CARE's verifier (Section 4) is a *projector*, not a scorer: a candidate repair
violating a declared hard predicate is removed deterministically, with no
probabilistic content. This package holds those predicates. Each self-registers
at import time via ``@register_constraint``, so importing the package populates
the global registry as a side effect; ``ALL_CONSTRAINTS`` lists the classes
explicitly, which is what lets a test re-register a known set after
``clear_registry``.

Predicates are typed. An **H** predicate is a boolean gate the verifier can
project onto; a **W** predicate is a graded measure in [0, 1] that the auditor
reports but the verifier never gates on. Completeness ships as both, because the
verifier needs a genuine boolean, not a thresholded score.

WHAT THE BENCHMARK ENABLES, AND WHY THE REST EXIST. Every experiment in the
paper runs with a single predicate active -- ``dq.completeness.floor``, selected
in ``bench.study._registry``. That is deliberate and conservative: filling a
missing value can never *newly* violate a completeness floor, so the verifier
cannot spuriously reject a proposer's correction, and the reported automation is
attributable to the conformal gate rather than to constraint filtering.

The others are not decoration and are not a second research agenda. Section 4.1
claims the hoisted verification path is correct for ANY predicate set, not just
the one the benchmark enables, and that claim is discharged by
``tests/property/test_projector_invariant.py``, which registers all of them and
asserts over generated artifacts that projection never introduces a hard
violation (invariant I2). They are kept deliberately heterogeneous in *how* they
evaluate -- a per-cell value check, a cross-row functional dependency, a
provenance-coverage check over the lineage graph, and a source-integrity check
over declared sources -- because a projector invariant tested against four
structurally identical predicates would be evidence of very little.

So: auditing a reported *number* needs only the completeness floor. Auditing the
*verifier invariant* needs the set.
"""

from __future__ import annotations

from care.predicates import dq  # noqa: F401

ALL_CONSTRAINTS = list(dq.PLUGINS)

__all__ = ["ALL_CONSTRAINTS", "dq"]

"""care.audit -- the Auditor.

Runs the assembled constraint registry over a data artifact to produce the
soft-measure scores, the failed hard predicates and the work queue V, with no
repair. In the benchmark this is the ``detection="constraints"`` path.
"""

from care.audit.auditor import Auditor

__all__ = ["Auditor"]

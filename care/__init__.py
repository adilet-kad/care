"""CARE: error-bounded automation for data repair.

Decides, per cell, whether a cleaner's repair is auto-applied or escalated to a
human, with the error rate of the auto-applied set bounded by alpha at
confidence 1 - delta.

The cleaner only proposes and never writes. Each repair passes three steps
(Section 4 and Algorithm 1 of the paper):
    Step 1, Verify      Any applied repair satisfies every hard (H) predicate.
    Step 2, Govern      The error bound comes only from the conformal threshold.
    Step 3, Trust gate  A repair whose evidence trust is below tau_min is escalated.

Subpackages: ``core`` (data model, constraints, registry), ``audit``,
``predicates``, ``verify``, ``conformal`` and ``escalate``.
"""

__version__ = "1.0.0"

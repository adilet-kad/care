"""CARE: error-bounded automation for data repair.

Decides, per cell, whether a cleaner's repair is auto-applied or escalated to a
human, with the error rate of the auto-applied set bounded by alpha at
confidence 1 - delta.

Invariants:
    I1  Cleaners only propose; every write passes the verifier.
    I2  Any applied repair satisfies every hard (H) predicate.
    I3  The error bound comes only from the conformal threshold.
    I4  A repair whose evidence trust is below tau_min is escalated.

Subpackages: ``core`` (data model, constraints, registry), ``audit``,
``predicates``, ``verify``, ``conformal`` and ``escalate``.
"""

__version__ = "1.0.0"

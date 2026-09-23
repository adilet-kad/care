"""CARE -- Certifiable Automation for data REpair.

A governance layer, not a cleaner. CARE sits between an arbitrary repair proposer
and the database and decides, per cell, whether a repair may be auto-applied or
must be escalated to a human, under a distribution-free bound on the error rate of
the auto-applied set at a user-chosen budget alpha with confidence 1-delta.

The package is organised around four invariants that the paper's proofs rely on,
so they are structural here rather than conventions:

    I1  The proposer only proposes; every write passes the verifier. Because the
        proposer never mutates the artifact, the hard-violation scan can be
        hoisted out of the per-candidate loop (Section 4.1).
    I2  Any applied repair satisfies every hard (H) predicate. The verifier is a
        projector onto the hard-feasible set, not a scorer, so it contributes no
        probabilistic content.
    I3  Every probabilistic claim is confined to the conformal threshold. One
        mechanism carries the guarantee, which is what makes the proofs tractable.
    I4  Score gains must be source-justified: a repair whose evidence trust falls
        below tau_min is escalated whatever the model's confidence.

Two further conventions shape the data model: constraints are typed H (hard
gate) / W (graded measure) / P (process, routed to humans and never faked as a
data predicate), and every cell and metadata triple can carry a provenance edge
that the traceability predicate checks.

Subpackages: ``core`` (artifact, constraints, registry, provenance), ``audit``
(violation detection), ``predicates`` (the shipped constraint plugins),
``verify`` (the projector), ``conformal`` (risk control) and ``escalate``
(tau-aware routing). Proposers live outside the package: the benchmark replays
an external cleaner's offline repair log (``bench.baran_proposer``).
"""

__version__ = "1.0.0"

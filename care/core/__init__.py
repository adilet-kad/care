"""care.core -- the data-model contracts every other module plugs into.

Re-exports the data model (artifact), the constraint contract (constraints),
the repair lifecycle (repair), and the plugin registry (registry). Provenance
analysis helpers are available from ``care.core.provenance``.
"""

from care.core.artifact import (
    Cell,
    CellKey,
    DataArtifact,
    DQRequirements,
    MetadataTriple,
    ProvenanceEvent,
    ProvenanceEventType,
    ProvenanceGraph,
    Source,
    SourceKind,
)
from care.core.constraints import (
    BaseConstraint,
    ConstraintResult,
    ConstraintType,
    Params,
)
from care.core.registry import (
    ConstraintRegistry,
    clear_registry,
    register_constraint,
    registered_constraints,
)
from care.core.repair import (
    ActionType,
    AuditReport,
    CalibrationRecord,
    Decision,
    RepairCandidate,
    ThresholdTable,
    VerifiedRepair,
    Violation,
)

__all__ = [
    # artifact
    "SourceKind",
    "ProvenanceEventType",
    "CellKey",
    "Source",
    "ProvenanceEvent",
    "ProvenanceGraph",
    "MetadataTriple",
    "Cell",
    "DQRequirements",
    "DataArtifact",
    # constraints
    "ConstraintType",
    "ConstraintResult",
    "Params",
    "BaseConstraint",
    # repair
    "ActionType",
    "Violation",
    "RepairCandidate",
    "VerifiedRepair",
    "CalibrationRecord",
    "Decision",
    "ThresholdTable",
    "AuditReport",
    # registry
    "register_constraint",
    "registered_constraints",
    "clear_registry",
    "ConstraintRegistry",
]

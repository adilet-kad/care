"""care.core.provenance -- provenance analysis over G.

Pure helpers behind the traceability predicate (``dq.traceability``). The data
contracts (Source, ProvenanceEvent, ProvenanceGraph) live in ``artifact.py``;
this module adds the coverage computation layered on top, keeping "the data"
and "the analysis" separate.
"""

from __future__ import annotations

from care.core.artifact import DataArtifact, ProvenanceGraph


def all_refs(art: DataArtifact) -> list[str]:
    """Every traceable ref in the artifact: cell keys plus metadata triple ids."""
    return list(art.cells.keys()) + [t.triple_id for t in art.metadata]


def has_provenance_edge(graph: ProvenanceGraph, ref: str) -> bool:
    """Whether ``ref`` has at least one provenance event."""
    return len(graph.edges_for(ref)) > 0


def uncovered_refs(art: DataArtifact) -> list[str]:
    """Refs with no provenance edge -- the traceability violation set."""
    return [r for r in all_refs(art) if not has_provenance_edge(art.provenance, r)]


__all__ = [
    "all_refs",
    "has_provenance_edge",
    "uncovered_refs",
]

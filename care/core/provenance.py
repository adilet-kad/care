"""Provenance coverage helpers used by the traceability predicate."""

from __future__ import annotations

from care.core.artifact import DataArtifact, ProvenanceGraph


def all_refs(art: DataArtifact) -> list[str]:
    """Cell keys plus metadata triple ids."""
    return list(art.cells.keys()) + [t.triple_id for t in art.metadata]


def has_provenance_edge(graph: ProvenanceGraph, ref: str) -> bool:
    return len(graph.edges_for(ref)) > 0


def uncovered_refs(art: DataArtifact) -> list[str]:
    """Refs with no provenance edge."""
    return [r for r in all_refs(art) if not has_provenance_edge(art.provenance, r)]


__all__ = [
    "all_refs",
    "has_provenance_edge",
    "uncovered_refs",
]

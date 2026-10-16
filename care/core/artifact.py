"""Data artifact D = (X, M, G).

    X  cells        relational values keyed by "{row_id}::{col}"
    M  metadata     provenance-linked (subject, predicate, object) triples
    G  provenance   sources with trust priors and edit-lineage events
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --------------------------------------------------------------------------- #
# Vocabularies                                                                 #
# --------------------------------------------------------------------------- #

SourceKind = Literal["internal", "scraped", "api", "synthetic", "lake", "human"]

# Provenance-event vocabulary.
ProvenanceEventType = Literal[
    "creation",
    "update",
    "transcription",
    "abstraction",
    "validation",
    "transfer",
]


# --------------------------------------------------------------------------- #
# CellKey -- the identity of a cell in X                                       #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class CellKey:
    """Identity of a cell in X; canonical string form ``"{row_id}::{col}"``."""

    row_id: str
    col: str

    _SEP: ClassVar[str] = "::"

    def __post_init__(self) -> None:
        if self._SEP in self.row_id or self._SEP in self.col:
            raise ValueError(
                f"row_id/col must not contain the reserved separator {self._SEP!r}"
            )

    def __str__(self) -> str:
        return f"{self.row_id}{self._SEP}{self.col}"

    @property
    def key(self) -> str:
        """Canonical string form, used as the key in ``DataArtifact.cells``."""
        return str(self)

    @classmethod
    def parse(cls, key: str) -> "CellKey":
        row_id, sep, col = key.partition(cls._SEP)
        if not sep:
            raise ValueError(f"Malformed CellKey string: {key!r}")
        return cls(row_id=row_id, col=col)


# --------------------------------------------------------------------------- #
# G -- sources, provenance events, provenance graph                            #
# --------------------------------------------------------------------------- #

class Source(BaseModel):
    """A provenance source with an optional content digest and a trust prior tau(s)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    uri: str
    kind: SourceKind
    sha256: str | None = None              # per-source content digest
    trust_prior: float = Field(0.5, ge=0.0, le=1.0)  # tau(s) in [0, 1]

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if len(v) != 64 or any(c not in "0123456789abcdefABCDEF" for c in v):
            raise ValueError("sha256 must be a 64-character hex digest")
        return v.lower()


class ProvenanceEvent(BaseModel):
    """A provenance event over a cell or metadata triple."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    type: ProvenanceEventType
    target_ref: str            # CellKey string or MetadataTriple.triple_id
    source_id: str
    actor: str
    ts: datetime


class ProvenanceGraph(BaseModel):
    """Provenance graph G: sources keyed by id plus an event list."""

    model_config = ConfigDict(extra="forbid")

    sources: dict[str, Source] = Field(default_factory=dict)
    events: list[ProvenanceEvent] = Field(default_factory=list)

    def add_source(self, source: Source) -> None:
        self.sources[source.source_id] = source

    def add_event(self, event: ProvenanceEvent) -> None:
        self.events.append(event)

    def edges_for(self, ref: str) -> list[ProvenanceEvent]:
        """All provenance events touching ``ref``."""
        return [e for e in self.events if e.target_ref == ref]


# --------------------------------------------------------------------------- #
# M -- metadata triples                                                        #
# --------------------------------------------------------------------------- #

class MetadataTriple(BaseModel):
    """A metadata triple in M, linked to its source via ``source_id``."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str
    source_id: str

    @property
    def triple_id(self) -> str:
        """Reference used as a provenance ``target_ref``."""
        return f"{self.subject}|{self.predicate}|{self.object}"


# --------------------------------------------------------------------------- #
# X -- cells                                                                   #
# --------------------------------------------------------------------------- #

class Cell(BaseModel):
    """A single (row, col) value in X with dtype, requiredness and provenance refs."""

    model_config = ConfigDict(extra="forbid")

    row_id: str
    col: str
    value: Any = None
    dtype: str = "string"
    is_required: bool = False
    provenance: list[str] = Field(default_factory=list)  # event_ids / source_ids

    @property
    def key(self) -> CellKey:
        return CellKey(row_id=self.row_id, col=self.col)


# --------------------------------------------------------------------------- #
# DQ requirements -- which predicates are in scope, with weights and targets    #
# --------------------------------------------------------------------------- #

class DQRequirements(BaseModel):
    """Enabled constraint ids with their weights ``w_i`` and targets ``theta_i``."""

    model_config = ConfigDict(extra="forbid")

    targets: dict[str, float] = Field(default_factory=dict)  # cid -> theta_i
    weights: dict[str, float] = Field(default_factory=dict)  # cid -> w_i
    enabled: set[str] = Field(default_factory=set)           # in-scope cids

    def theta(self, cid: str) -> float | None:
        return self.targets.get(cid)

    def weight(self, cid: str) -> float:
        return self.weights.get(cid, 1.0)


# --------------------------------------------------------------------------- #
# DataArtifact -- D = (X, M, G) plus requirements                               #
# --------------------------------------------------------------------------- #

class DataArtifact(BaseModel):
    """Data artifact D = (X, M, G) with its DQ requirements; cells keyed by ``CellKey.key``."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    cells: dict[str, Cell] = Field(default_factory=dict)   # keyed by CellKey.key
    metadata: list[MetadataTriple] = Field(default_factory=list)
    provenance: ProvenanceGraph = Field(default_factory=ProvenanceGraph)
    dq_requirements: DQRequirements = Field(default_factory=DQRequirements)

    # --- cell access keyed by typed CellKey -------------------------------- #
    def set_cell(self, cell: Cell) -> None:
        self.cells[cell.key.key] = cell

    def get_cell(self, key: CellKey | str) -> Cell | None:
        return self.cells.get(key if isinstance(key, str) else key.key)

    def cell(self, row_id: str, col: str) -> Cell | None:
        return self.get_cell(CellKey(row_id=row_id, col=col))

    def iter_cells(self) -> Iterator[Cell]:
        return iter(self.cells.values())

    def cell_keys(self) -> list[CellKey]:
        return [CellKey.parse(k) for k in self.cells]

    # --- metadata + provenance --------------------------------------------- #
    def add_metadata(self, triple: MetadataTriple) -> None:
        self.metadata.append(triple)

    def add_source(self, source: Source) -> None:
        self.provenance.add_source(source)

    def add_event(self, event: ProvenanceEvent) -> None:
        self.provenance.add_event(event)

    def has_provenance(self, ref: str) -> bool:
        """Whether ``ref`` (cell key or triple id) has any provenance edge."""
        return len(self.provenance.edges_for(ref)) > 0

    # --- portability ------------------------------------------------------- #
    def to_json(self, **kwargs: Any) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, data: str) -> "DataArtifact":
        return cls.model_validate_json(data)


__all__ = [
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
]

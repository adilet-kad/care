"""LogProposer: replay a cleaner's offline log as CARE repair candidates.

Log format (checked by prepare_log.py): ``row_id, column, value, [confidence], [source]``.
An empty value or an absent cell means no proposal, so CARE escalates the cell.
``confidence`` is the cleaner's reported confidence in [0, 1]; ``source`` is an
optional per-cell evidence id read by the trust gate.
"""

from __future__ import annotations

import csv
import os

from care.core.repair import RepairCandidate, Violation


# s_agree when the log has no confidence column; each stratum is then admitted all-or-nothing.
_CONST_CONFIDENCE = 0.5


class LogProposer:
    """Propose the value a cleaner logged for each cell, or nothing if it made no proposal."""

    def __init__(
        self,
        log_path: str,
        *,
        source_id: str = "baran",
        constant_confidence: float = _CONST_CONFIDENCE,
    ):
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"cleaner log not found: {log_path}")
        self.source_id = source_id
        self.constant_confidence = constant_confidence
        # ref -> (value, confidence|None, source_id|None)
        self._by_ref: dict[str, tuple[str, float | None, str | None]] = {}
        self._has_conf = False
        self._has_source = False
        self._load(log_path)

    def _load(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            required = {"row_id", "column", "value"}
            missing = required - cols
            if missing:
                raise ValueError(
                    f"cleaner log {path} missing required columns {missing}; has {cols}. "
                    f"Run prepare_log.py first."
                )
            self._has_conf = "confidence" in cols
            self._has_source = "source" in cols
            for row in reader:
                value = (row.get("value") or "").strip()
                if not value:
                    continue  # empty value: no proposal
                ref = f"{row['row_id']}::{row['column']}"
                conf = None
                if self._has_conf:
                    try:
                        c = float(row["confidence"])
                        conf = max(0.0, min(1.0, c))
                    except (ValueError, TypeError):
                        conf = None
                src = (row.get("source") or "").strip() if self._has_source else ""
                self._by_ref[ref] = (value, conf, src or None)

    def propose(self, v: Violation, art=None, *, options=None) -> list[RepairCandidate]:
        out: list[RepairCandidate] = []
        for ref in v.refs:
            entry = self._by_ref.get(ref)
            if entry is None:
                continue  # no proposal: CARE escalates
            value, conf, src = entry
            s_agree = conf if conf is not None else self.constant_confidence
            out.append(
                RepairCandidate(
                    target_ref=ref,
                    proposed_value=value,
                    rationale=f"external repair ({src or self.source_id})",
                    # per-cell source if the log has one, else the default
                    evidence=[src or self.source_id],
                    s_conf=None,
                    s_agree=s_agree,
                    s_consistency=conf,          # None if the log has no confidence
                )
            )
        return out



__all__ = ["LogProposer"]

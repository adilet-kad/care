"""bench.baran_proposer -- offline-log proposer wrapping an external repair system.

The "universal safety layer" experiment: CARE governs ANY proposer, not just the LLM.
Rather than integrate Baran's live ML pipeline (dependency hell, coupled reproducibility),
we run Baran natively once, dump its corrections to a CSV log, and read them here. The
same adapter works for any external system (HoloClean, commercial tools) -- emit the log
format, plug it in.

Log contract (validated by prepare_log.py):
    row_id, column, value, [confidence], [source]
    - ref = "{row_id}::{column}" must match CARE's CellKey format
    - empty value  -> the system abstained on that cell
    - confidence   -> OPTIONAL float in [0,1]; absent -> constant fallback
    - source       -> OPTIONAL per-cell evidence source_id (e.g. RetClean's
      "retclean_lake" when a repair is grounded in a retrieved data-lake tuple vs
      "retclean_model" when the model guessed with no citation). Absent -> the
      proposer's default source_id. This is what lets an external system's own
      provenance flow into CARE's trust gate (a lake-grounded repair is evidence-
      backed; an ungrounded guess is evidence-thin, and a compromised lake source
      can be marked low-trust so the gate escalates repairs that cite it -- the
      retrieval-poisoning defence).
    - cells absent from the log -> the system made no proposal -> CARE escalates
      (identical semantics to an LLM abstention)

This proposer is proposer-agnostic by design: it emits RepairCandidates with a
confidence channel, and the conformal controller bounds the error of whatever it
proposes -- the whole point of the experiment. The same class wraps Baran,
RetClean, HoloClean, or any external system that can emit this log.
"""

from __future__ import annotations

import csv
import os

from care.core.repair import RepairCandidate, Violation


# A constant s_agree used when the external log carries no per-cell confidence.
# Value is arbitrary-but-fixed: with constant confidence, within a stratum every
# Baran repair has the same s_hat, so the conformal controller can only certify the
# whole stratum (all-or-nothing per column) -- it certifies iff the stratum's error
# <= alpha. That is the honest v1 result; a real per-cell confidence (if the log
# provides one) gives graded within-stratum automation instead.
_CONST_CONFIDENCE = 0.5


class BaranProposer:
    """Proposes the corrected value an external system logged for each cell.

    ``propose(violation, art, *, options=None)`` returns ``list[RepairCandidate]``
    (one element, or empty when the system abstained / never proposed on that
    ref).
    """

    def __init__(
        self,
        log_path: str,
        *,
        source_id: str = "baran",
        constant_confidence: float = _CONST_CONFIDENCE,
    ):
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"Baran log not found: {log_path}")
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
                    f"Baran log {path} missing required columns {missing}; has {cols}. "
                    f"Run prepare_log.py first."
                )
            self._has_conf = "confidence" in cols
            self._has_source = "source" in cols
            for row in reader:
                value = (row.get("value") or "").strip()
                if not value:
                    continue  # logged abstention -> no candidate (CARE escalates)
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
                continue  # system made no proposal on this cell -> escalate
            value, conf, src = entry
            s_agree = conf if conf is not None else self.constant_confidence
            out.append(
                RepairCandidate(
                    target_ref=ref,
                    proposed_value=value,
                    rationale=f"external repair ({src or self.source_id})",
                    # per-cell evidence source if the log provided one (RetClean's
                    # lake vs model provenance), else the proposer's default source.
                    evidence=[src or self.source_id],
                    s_llm=None,                  # no logprob from an external system
                    s_agree=s_agree,
                    s_consistency=conf,          # per-cell signal if the log had one, else None
                )
            )
        return out


# The offline-log proposer is proposer-agnostic; the Baran name is historical.
OfflineLogProposer = BaranProposer


__all__ = ["BaranProposer"]

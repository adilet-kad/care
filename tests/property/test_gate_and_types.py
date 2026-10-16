"""Gate-first calibration and the type stratifier: small, fast checks."""
from __future__ import annotations

from care.conformal import type_domains
from care.core.artifact import Cell, DataArtifact, Source
from care.core.repair import RepairCandidate


def test_type_domains():
    art = DataArtifact(artifact_id="t")
    for i in range(100):
        art.set_cell(Cell(row_id=str(i), col="num", value=str(i * 1.5)))
        art.set_cell(Cell(row_id=str(i), col="day", value=f"2020-01-{1 + i % 28:02d}"))
        art.set_cell(Cell(row_id=str(i), col="cat", value="ab"[i % 2]))
        art.set_cell(Cell(row_id=str(i), col="txt", value=f"name {i}"))
    assert type_domains(art) == {"num": "type:numeric", "day": "type:date",
                                 "cat": "type:categorical", "txt": "type:text"}


def test_gate_rule_matches_escalator():
    from bench.study import _gated
    src = {"bad": Source(source_id="bad", uri="u", kind="scraped", trust_prior=0.1),
           "ok": Source(source_id="ok", uri="u", kind="internal", trust_prior=0.9)}
    mk = lambda ev: RepairCandidate(target_ref="0::a", proposed_value="x", evidence=ev)
    assert _gated(mk(["bad"]), src, 0.5)
    assert not _gated(mk(["bad", "ok"]), src, 0.5)      # best source decides
    assert not _gated(mk(["unknown"]), src, 0.5)        # unknown is not untrusted
    assert not _gated(mk([]), src, 0.5)                 # no evidence is not gated

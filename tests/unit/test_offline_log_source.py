"""LogProposer's optional per-cell `source` column, and the default when it is absent."""

from __future__ import annotations

import csv

from bench.log_proposer import LogProposer
from care.core.repair import Violation


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_per_cell_source_becomes_evidence(tmp_path):
    p = tmp_path / "log.csv"
    _write(p, ["row_id", "column", "value", "source"],
           [["0", "city", "boston", "lake"],
            ["1", "city", "denver", "model"]])
    prop = LogProposer(str(p), source_id="cleaner")
    cands = {c.target_ref: c for c in prop.propose(Violation(cid="t", refs=["0::city", "1::city"]))}
    assert cands["0::city"].evidence == ["lake"]
    assert cands["1::city"].evidence == ["model"]
    assert cands["0::city"].proposed_value == "boston"


def test_absent_source_column_falls_back_to_default(tmp_path):
    # no `source` column -> evidence is the default source_id
    p = tmp_path / "baran.csv"
    _write(p, ["row_id", "column", "value"], [["0", "city", "boston"]])
    prop = LogProposer(str(p), source_id="baran")
    (cand,) = prop.propose(Violation(cid="t", refs=["0::city"]))
    assert cand.evidence == ["baran"]


def test_blank_source_falls_back_to_default(tmp_path):
    p = tmp_path / "mixed.csv"
    _write(p, ["row_id", "column", "value", "source"],
           [["0", "city", "boston", ""], ["1", "city", "denver", "lake"]])
    prop = LogProposer(str(p), source_id="cleaner")
    cands = {c.target_ref: c for c in prop.propose(Violation(cid="t", refs=["0::city", "1::city"]))}
    assert cands["0::city"].evidence == ["cleaner"]         # blank -> default
    assert cands["1::city"].evidence == ["lake"]

"""The offline-log proposer's optional per-cell `source` column (RetClean lake vs
model provenance), and back-compat when the column is absent (Baran)."""

from __future__ import annotations

import csv

from bench.baran_proposer import BaranProposer
from care.core.repair import Violation


def _write(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_per_cell_source_becomes_evidence(tmp_path):
    p = tmp_path / "retclean.csv"
    _write(p, ["row_id", "column", "value", "source"],
           [["0", "city", "boston", "retclean_lake"],
            ["1", "city", "denver", "retclean_model"]])
    prop = BaranProposer(str(p), source_id="retclean")
    cands = {c.target_ref: c for c in prop.propose(Violation(cid="t", refs=["0::city", "1::city"]))}
    assert cands["0::city"].evidence == ["retclean_lake"]    # lake-grounded
    assert cands["1::city"].evidence == ["retclean_model"]   # model-guessed
    assert cands["0::city"].proposed_value == "boston"


def test_absent_source_column_falls_back_to_default(tmp_path):
    # Baran-style log with no `source` column -> evidence is the default source_id
    p = tmp_path / "baran.csv"
    _write(p, ["row_id", "column", "value"], [["0", "city", "boston"]])
    prop = BaranProposer(str(p), source_id="baran")
    (cand,) = prop.propose(Violation(cid="t", refs=["0::city"]))
    assert cand.evidence == ["baran"]


def test_blank_source_falls_back_to_default(tmp_path):
    p = tmp_path / "mixed.csv"
    _write(p, ["row_id", "column", "value", "source"],
           [["0", "city", "boston", ""], ["1", "city", "denver", "retclean_lake"]])
    prop = BaranProposer(str(p), source_id="retclean")
    cands = {c.target_ref: c for c in prop.propose(Violation(cid="t", refs=["0::city", "1::city"]))}
    assert cands["0::city"].evidence == ["retclean"]         # blank -> default
    assert cands["1::city"].evidence == ["retclean_lake"]

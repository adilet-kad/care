"""Tests for the non-oracle detection-log path (bench.study.detect, mode='log:').

This is the counterpart to the Baran offline-log adapter, but for DETECTION: a
real detector (e.g. Raha) runs natively in its own repo, exports the cells it
flagged as a plain ``row_id,column`` CSV, and this reads that log as the work
queue -- so CARE never imports the detector's code, mirroring BaranProposer.
"""

from __future__ import annotations

import csv

import pytest

from care.core import Cell, DataArtifact
from bench.datasets import Defects
from bench.study import _load_detection_log, detect


def _write_log(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "column"])
        w.writerows(rows)


def _artifact():
    art = DataArtifact(artifact_id="A")
    art.set_cell(Cell(row_id="0", col="age", value=None, is_required=True))
    art.set_cell(Cell(row_id="1", col="age", value=40, is_required=True))
    art.set_cell(Cell(row_id="0", col="city", value="nyc", is_required=True))
    return art


def test_load_detection_log_parses_refs(tmp_path):
    p = tmp_path / "detected.csv"
    _write_log(p, [("0", "age"), ("1", "city")])
    refs = _load_detection_log(str(p))
    assert refs == {"0::age", "1::city"}


def test_load_detection_log_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "col"])  # wrong header names
        w.writerow(["0", "age"])
    with pytest.raises(ValueError, match="missing columns"):
        _load_detection_log(str(p))


def test_detect_log_mode_filters_to_known_cells(tmp_path):
    art = _artifact()
    p = tmp_path / "detected.csv"
    # includes one real cell (0::age), one real cell not flagged (1::age is
    # absent -> a detector false negative, fine), and one bogus ref that is not
    # a cell in this artifact at all (a stray/miskeyed detector output).
    _write_log(p, [("0", "age"), ("0", "city"), ("99", "nonexistent_col")])
    defects = Defects(gold={"0::age": "30"})
    refs = detect(art, reg=None, defects=defects, mode=f"log:{p}")
    assert set(refs) == {"0::age", "0::city"}   # bogus ref silently dropped


def test_detect_oracle_mode_unaffected():
    defects = Defects(gold={"0::age": "30", "1::city": "sf"})
    refs = detect(art=None, reg=None, defects=defects, mode="oracle")
    assert refs == list(defects.gold)

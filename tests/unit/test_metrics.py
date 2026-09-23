"""Unit tests for bench.metrics (scoring and the Pareto front)."""

from __future__ import annotations

from bench.metrics import RepairResult, pareto_front, score


def test_score_computes_precision_recall_error():
    gold = {"a": "x", "b": "y", "c": "z"}
    # auto-applied a (right), b (wrong); c escalated
    result = RepairResult(name="t", applied={"a": "x", "b": "WRONG"}, escalated={"c"})
    s = score(result, gold)
    assert s.repair_precision == 0.5          # 1 of 2 applied correct
    assert s.realized_error == 0.5
    assert s.human_cost == 1 / 3


def test_pareto_front_keeps_lower_left():
    pts = [(0.0, 0.4), (0.25, 0.02), (0.5, 0.05), (1.0, 0.0)]
    front = pareto_front(pts)
    assert (0.5, 0.05) not in front           # dominated by (0.25, 0.02)
    assert (0.25, 0.02) in front and (1.0, 0.0) in front

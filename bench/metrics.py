"""Scores for one governed-repair run against a truth map."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RepairResult:
    name: str
    applied: dict[str, object]      # ref -> value the system wrote (auto-applied only)
    escalated: set[str]             # refs routed to a human


@dataclass
class Scores:
    repair_precision: float
    repair_recall: float
    repair_f1: float
    realized_error: float           # error among auto-applied repairs (vs alpha)
    human_cost: float               # escalated fraction of defects
    n_auto: int
    n_escalated: int
    applied_contamination: float = 0.0  # eps_S: corrupted fraction of the auto-applied set


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def score(
    result: RepairResult,
    gold: dict[str, object],
    *,
    contaminated: set[str] = frozenset(),
) -> Scores:
    """Score ``result`` against ``gold``.

    ``contaminated`` (e.g. injected poison) only sets ``applied_contamination``.
    """
    refs = set(gold)
    applied = {k: v for k, v in result.applied.items() if k in refs}
    correct_applied = sum(1 for k, v in applied.items() if v == gold[k])

    precision = correct_applied / len(applied) if applied else 0.0
    recall = correct_applied / len(refs) if refs else 0.0
    realized_error = 1.0 - precision if applied else 0.0
    human_cost = len(result.escalated & refs) / len(refs) if refs else 0.0
    n_contam = sum(1 for k in applied if k in contaminated)
    applied_contamination = n_contam / len(applied) if applied else 0.0

    return Scores(
        repair_precision=precision,
        repair_recall=recall,
        repair_f1=_f1(precision, recall),
        realized_error=realized_error,
        human_cost=human_cost,
        n_auto=len(applied),
        n_escalated=len(result.escalated & refs),
        applied_contamination=applied_contamination,
    )


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left Pareto front of (human_cost, realized_error) points."""
    pts = sorted(set(points))
    front: list[tuple[float, float]] = []
    best_err = float("inf")
    for hc, err in pts:
        if err < best_err:
            front.append((hc, err))
            best_err = err
    return front


__all__ = ["RepairResult", "Scores", "score", "pareto_front"]

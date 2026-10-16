"""Multi-seed experiments (pareto, poison robustness) with bootstrap CIs.

Proposals are computed once per run; seeds resample the calibration/test split.
"""

from __future__ import annotations

from bench.metrics import pareto_front
from bench.stats import bootstrap_ci
from bench.study import evaluate_split, load_and_propose


def _agg(per_seed_scores, metric):
    """list[dict[name -> Scores]] -> {name: bootstrap CI of ``metric``}."""
    names = per_seed_scores[0].keys()
    return {n: bootstrap_ci([getattr(s[n], metric) for s in per_seed_scores]) for n in names}


def _mondrian_strata_fn(art, defects):
    """Per-column strata if the median column has >= 60 error cells, else cardinality buckets."""
    from care.conformal import by_column, by_cardinality
    import collections as _c
    from care.core.artifact import CellKey as _CK
    col_counts = _c.Counter(_CK.parse(r).col for r in defects.gold)
    median_col = sorted(col_counts.values())[len(col_counts) // 2] if col_counts else 0
    return by_column() if median_col >= 60 else by_cardinality(art)


def _detected_truth(art, defects):
    """Truth for every cell: the gold value on error cells, the current value elsewhere.

    Used by ``scoring="detected"``, which also counts changes to clean cells as errors.
    """
    truth = {}
    for c in art.iter_cells():
        truth[f"{c.row_id}::{c.col}"] = c.value
    truth.update(defects.gold)
    return truth


def pareto(name, client, *, alpha=0.1, alphas=None, delta=0.1, seeds=20, detection="oracle",
           max_cells=None, row_range=None, fast_verify="auto", scoring="errors"):
    """CARE (marginal and Mondrian strata), apply-all and escalate-all per alpha, with the Pareto front.

    ``scoring``: ``"errors"`` scores only true error cells; ``"detected"`` scores every
    proposed cell (see ``_detected_truth``).
    """
    from care.conformal import marginal_strata
    art, defects, repairs, _r = load_and_propose(
        name, client, detection=detection, max_cells=max_cells, row_range=row_range, fast_verify=fast_verify)
    truth = _detected_truth(art, defects) if scoring == "detected" else defects.gold
    alphas = tuple(alphas) if alphas else (alpha,)
    _mondrian = _mondrian_strata_fn(art, defects)
    strata = {"marginal": marginal_strata, "mondrian": _mondrian}

    # Confidence scales can differ by source (e.g. BClean change vs keep decisions),
    # so multi-source logs also get column x source strata.
    from care.conformal import by_product, by_source
    _srcs = {(getattr(r, "evidence", None) or ["?"])[0] for r in repairs.values()}
    if len(_srcs) > 1:
        strata["mondrian_src"] = by_product(_mondrian, by_source({}))
        print(f"  [strata] log carries {len(_srcs)} sources {sorted(_srcs)}; adding "
              f"column x source control alongside column-only")

    rows = []
    for a in alphas:
        block = []
        for mode, sf in strata.items():
            ps = [evaluate_split(repairs, truth, alpha=a, delta=delta, seed=s, strata_fn=sf)
                  for s in range(seeds)]
            hc = _agg(ps, "human_cost")["CARE"]; err = _agg(ps, "realized_error")["CARE"]
            f1 = _agg(ps, "repair_f1")["CARE"]
            block.append({"dataset": name, "baseline": "CARE", "strata": mode, "alpha": a, "scoring": scoring,
                          "human_cost": hc.mean, "realized_error": err.mean, "repair_f1": f1.mean,
                          "err_hi": err.hi, "covered": err.hi <= a})
        base = [evaluate_split(repairs, truth, alpha=a, delta=delta, seed=s)
                for s in range(seeds)]
        for nm in ("ApplyAll", "EscalateAll"):
            hc = _agg(base, "human_cost")[nm]; err = _agg(base, "realized_error")[nm]
            f1 = _agg(base, "repair_f1")[nm]
            block.append({"dataset": name, "baseline": nm, "strata": "-", "alpha": a, "scoring": scoring,
                          "human_cost": hc.mean, "realized_error": err.mean, "repair_f1": f1.mean,
                          "err_hi": err.hi, "covered": err.hi <= a})
        pts = [(round(r["human_cost"], 4), round(r["realized_error"], 4)) for r in block]
        front = pareto_front(pts)
        for r in block:
            r["on_pareto_front"] = (round(r["human_cost"], 4), round(r["realized_error"], 4)) in front
        rows.extend(block)
    return rows


def poison_robustness(name, client, *, poison_fracs=(0.0, 0.2, 0.4), alpha=0.1, delta=0.1,
                      seeds=20, detection="oracle", max_cells=None, row_range=None, fast_verify="auto",
                      trusted_poison=False, scoring="errors"):
    """Realized error of CARE vs apply-all as the poisoned fraction rises.

    ``trusted_poison=False`` attributes poison to a low-trust source, which the gate
    should catch; ``True`` uses a high-trust source, so CARE's error should stay within
    alpha + eps_S(1-alpha). Uses Mondrian strata (column x source for multi-source
    logs). Results are only informative if human_cost < 1 at poison_frac=0.
    """
    from care.conformal import by_product, by_source

    art, defects, repairs, _r = load_and_propose(
        name, client, detection=detection, max_cells=max_cells, row_range=row_range, fast_verify=fast_verify)
    truth = _detected_truth(art, defects) if scoring == "detected" else defects.gold
    strata_fn = _mondrian_strata_fn(art, defects)
    _srcs = {(getattr(r, "evidence", None) or ["?"])[0] for r in repairs.values()}
    if len(_srcs) > 1:
        strata_fn = by_product(strata_fn, by_source({}))
        print(f"  [strata] log carries {len(_srcs)} sources {sorted(_srcs)}; using "
              f"column x source (column-only would certify nothing here)")
    rows = []
    for pf in poison_fracs:
        per_seed = [evaluate_split(repairs, truth, alpha=alpha, delta=delta,
                                   strata_fn=strata_fn,
                                   seed=s, poison_frac=pf, trusted_poison=trusted_poison)
                    for s in range(seeds)]
        for nm in ("CARE", "ApplyAll"):
            err = _agg(per_seed, "realized_error")[nm]
            hc = _agg(per_seed, "human_cost")[nm]
            eps = _agg(per_seed, "applied_contamination")[nm]
            # bound alpha + eps_S(1-alpha) at the realized eps_S
            thm2_bound = alpha + eps.mean * (1.0 - alpha)
            rows.append({"dataset": name, "baseline": nm, "poison_frac": pf,
                         "trusted_poison": trusted_poison, "scoring": scoring,
                         "realized_error": err.mean, "err_lo": err.lo, "err_hi": err.hi,
                         "human_cost": hc.mean, "applied_contamination": eps.mean,
                         "thm2_bound": thm2_bound if nm == "CARE" else "",
                         "within_thm2_bound": (err.mean <= thm2_bound + 1e-9) if nm == "CARE" else ""})
    return rows


__all__ = ["pareto", "poison_robustness"]

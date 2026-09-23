"""bench.experiments -- the paper's experiments (multi-seed, with CIs).

Each function loads ONE set of proposals (an external cleaner's offline log,
replayed through the verifier and the confidence combiner once) and resamples
the calibration/test split over seeds. Returns plain dict rows ready to write
to CSV.
"""

from __future__ import annotations

from bench.metrics import pareto_front
from bench.stats import bootstrap_ci
from bench.study import evaluate_split, load_and_propose


def _agg(per_seed_scores, metric):
    """per_seed_scores: list[dict[name->Scores]] -> {name: Agg(metric)}."""
    names = per_seed_scores[0].keys()
    return {n: bootstrap_ci([getattr(s[n], metric) for s in per_seed_scores]) for n in names}


def _mondrian_strata_fn(art, defects):
    """The adaptive group-conditional stratifier: per-column when columns are fat
    enough to calibrate (>= ~60 error cells each), else coarsen to cardinality
    buckets. The finest-partition-that-retains-mass rule (not per-dataset tuning).
    Shared by pareto and poison_robustness so both exercise the SAME control that
    actually auto-applies a certifiable subset -- marginal (global) trivially
    escalates whenever the global error exceeds alpha, which makes any poison
    experiment vacuous (nothing is auto-applied, so the gate is never exercised)."""
    from care.conformal import by_column, by_cardinality
    import collections as _c
    from care.core.artifact import CellKey as _CK
    col_counts = _c.Counter(_CK.parse(r).col for r in defects.gold)
    median_col = sorted(col_counts.values())[len(col_counts) // 2] if col_counts else 0
    return by_column() if median_col >= 60 else by_cardinality(art)


def _detected_truth(art, defects):
    """Full truth map over EVERY cell: the clean value is the gold repair where the
    cell is a known error, else the cell's current (already-correct) value.

    Why this exists (scoring scope -- a real methodological choice):
    `evaluate_split` derives its evaluated cell set from the truth dict it is given
    (`refs = [r for r in truth if r in repairs]`). Passing `defects.gold` therefore
    scores ONLY true error cells -- the standard repair-F1 population, and what CARE
    evaluates under oracle detection. But it silently IGNORES what a cleaner does to
    the CLEAN cells it also touches: corrupting a clean cell costs nothing, and
    correctly preserving one earns nothing. That blind spot made detection mode have
    no effect on scoring at all (a log-detection run and an oracle run returned
    byte-identical numbers despite 17k vs 509 proposals).

    Passing THIS map instead scores every cell the cleaner proposed on, counting a
    changed clean cell as an error and a preserved one as correct -- the
    deployment-realistic measure, and the one that makes `--detection log:` meaningful.
    """
    truth = {}
    for c in art.iter_cells():
        truth[f"{c.row_id}::{c.col}"] = c.value
    truth.update(defects.gold)          # gold overrides the dirty value on error cells
    return truth


def pareto(name, client, *, alpha=0.1, alphas=None, delta=0.1, seeds=20, detection="oracle",
           max_cells=None, row_range=None, fast_verify="auto", scoring="errors"):
    """(human_cost, realized_error) + Pareto front, with CARE under BOTH 'marginal'
    and 'mondrian' (cardinality-stratified) control. LLMOnly/NoOp are
    stratification-invariant. One shared proposal pass across all alphas/strata/seeds,
    so Mondrian and the alpha sweep add ~0 LLM calls.

    ``scoring``: ``"errors"`` (default, backward-compatible) scores only true error
    cells; ``"detected"`` scores every proposed cell against the full truth map
    (see ``_detected_truth``) so preserving clean cells counts and corrupting them
    is penalised. Report which one you used -- they can differ sharply.
    """
    from care.conformal import marginal_strata
    art, defects, repairs, _r = load_and_propose(
        name, client, detection=detection, max_cells=max_cells, row_range=row_range, fast_verify=fast_verify)
    truth = _detected_truth(art, defects) if scoring == "detected" else defects.gold
    alphas = tuple(alphas) if alphas else (alpha,)
    _mondrian = _mondrian_strata_fn(art, defects)
    strata = {"marginal": marginal_strata, "mondrian": _mondrian}

    # A log that mixes DECISION TYPES (e.g. BClean's "change this cell" alongside
    # "keep the current value") carries confidence scores on incompatible scales:
    # one rates the chosen repair, the other rates the repair that was declined.
    # Pooled in one column stratum they can invert the ranking (Simpson's paradox)
    # even though each source ranks correctly alone, and no single threshold per
    # stratum can be right for both. When we detect more than one source we therefore
    # ALSO report column x source control, so the cost of the finer partition (more
    # strata, thinner calibration, bigger multiplicity penalty) is measured rather
    # than assumed. Engages only when it is needed: a single-source log takes the
    # column-only path and is unaffected.
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
        for nm in ("LLMOnly", "NoOp"):
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
    """Realized error of CARE vs LLMOnly as the poisoned fraction rises.

    ``trusted_poison=False`` (default) attributes poison to a LOW-trust source, so
    the escalation gate removes it -> Theorem 1 exhibit (CARE error stays ~0,
    eps_S ~ 0). ``trusted_poison=True`` attributes it to a HIGH-trust source, so
    the gate passes it -> Theorem 2 exhibit (CARE error tracks alpha + eps_S(1-alpha)).
    Both report ``applied_contamination`` (eps_S) and the Theorem-2 bound, so the
    figure doubles as the empirical validation of the bound.

    Uses the MONDRIAN stratifier: marginal (global) control trivially escalates
    everything whenever the global error exceeds alpha (beers 12% > 10%, rayyan
    60%+), so under marginal the poison gate is never exercised and eps_S is
    always 0 -- a vacuous experiment. Mondrian certifies the clean low-error strata,
    which is what poison can then slip into (trusted) or be caught in (low-trust).

    ``scoring`` MUST match whatever made the proposer certify a non-empty auto-applied
    set in the pareto experiment. **A poison experiment on an empty auto-applied set is
    vacuous**: nothing is auto-applied, so eps_S = 0 and realized error = 0 at EVERY
    poison level, and the gate is never exercised (the same failure mode as using the
    marginal stratifier, from a different cause). Concretely: Jellyfish/beers certifies
    38% under ``scoring="detected"`` but NOTHING under ``scoring="errors"`` (error-cell
    accuracy ~19%), so the Theorem 1/2 exhibits require ``scoring="detected"`` there.
    Sanity check before trusting any poison table: **human_cost must be < 1.0 at
    poison_frac=0**, otherwise the run is vacuous.

    Multi-source logs get column x source, matching ``pareto``. A keep log carries two
    decision types whose confidence scales are not comparable (deviation D23), and for
    such a log column-only Mondrian can certify nothing while column x source certifies
    a substantial fraction -- HoloClean/hospital is 0.00 versus 0.53 at alpha=0.2. Since
    a poison exhibit needs a NON-EMPTY auto-applied set, using the coarser partition
    here would make every multi-source configuration vacuous for a reason that has
    nothing to do with poison.
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
        for nm in ("CARE", "LLMOnly"):
            err = _agg(per_seed, "realized_error")[nm]
            hc = _agg(per_seed, "human_cost")[nm]
            eps = _agg(per_seed, "applied_contamination")[nm]
            # Theorem 2 upper bound evaluated at the realized eps_S: alpha + eps_S(1-alpha).
            thm2_bound = alpha + eps.mean * (1.0 - alpha)
            rows.append({"dataset": name, "baseline": nm, "poison_frac": pf,
                         "trusted_poison": trusted_poison, "scoring": scoring,
                         "realized_error": err.mean, "err_lo": err.lo, "err_hi": err.hi,
                         "human_cost": hc.mean, "applied_contamination": eps.mean,
                         "thm2_bound": thm2_bound if nm == "CARE" else "",
                         "within_thm2_bound": (err.mean <= thm2_bound + 1e-9) if nm == "CARE" else ""})
    return rows


__all__ = ["pareto", "poison_robustness"]

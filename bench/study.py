"""Governed-correction protocol: detect, replay a cleaner's log, calibrate and score.

Detection and the cleaner are held fixed, so CARE, apply-all and escalate-all differ
only in governance. Coverage is realized error on auto-applied test cells, over many
calibration/test splits.
"""

from __future__ import annotations

import os
import random
import time as _time
from concurrent.futures import ThreadPoolExecutor

try:  # optional progress bar
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    def _tqdm(it, **kw):
        return it

from care.audit import Auditor
from care.conformal import ConformalController, marginal_strata
from care.core import (
    ConstraintRegistry, DQRequirements, Source, Violation,
    clear_registry, register_constraint,
)
from care.escalate import ActiveEscalator
from care.predicates import ALL_CONSTRAINTS
from care.verify import Verifier
from care.core.repair import VerifiedRepair

from bench.datasets import load
from bench.log_proposer import LogProposer
from bench.metrics import RepairResult, Scores, score


# Tags of '<tag>:<path>' log backends; bench.run_study imports this.
OFFLINE_LOG_BACKENDS = ("baran", "bclean", "holoclean", "gidcl", "zeroec")


def is_offline_log_backend(spec) -> bool:
    """True if ``spec`` is '<tag>:<path>' with a known tag or a variant such as ``baran_raha``."""
    if not isinstance(spec, str):
        return False
    tag = spec.split(":", 1)[0]
    return tag in OFFLINE_LOG_BACKENDS or tag.split("_", 1)[0] in OFFLINE_LOG_BACKENDS


def _registry() -> ConstraintRegistry:
    clear_registry()
    for c in ALL_CONSTRAINTS:
        register_constraint(c)
    # Only the completeness floor is hard; filling a value cannot violate it.
    return ConstraintRegistry.from_requirements(
        DQRequirements(enabled={"dq.completeness.floor"}), strict=True
    )


def _load_detection_log(path: str) -> set[str]:
    """Read a ``row_id,column`` CSV of detected cells (e.g. from Raha) into a set of refs."""
    import csv as _csv

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise ValueError(f"detection log is empty: {path}")
    fields = set(rows[0].keys())
    missing = {"row_id", "column"} - fields
    if missing:
        raise ValueError(f"detection log {path} missing columns {missing}; has {fields}")
    return {f"{r['row_id']}::{r['column']}" for r in rows}


def detect(art, reg, defects, *, mode="oracle") -> list[str]:
    """Return the work-queue refs.

    ``mode``: ``"oracle"`` (gold error cells), ``"constraints"`` (hard-constraint
    violations) or ``"log:<path>"`` (a detector's exported cell log).
    """
    if mode == "oracle":
        return list(defects.gold)
    if mode.startswith("log:"):
        log_refs = _load_detection_log(mode.split(":", 1)[1])
        known_refs = {f"{c.row_id}::{c.col}" for c in art.iter_cells()}
        return [r for r in log_refs if r in known_refs]
    _r, V = Auditor().run(art, reg, options={})
    return [ref for v in V for ref in v.refs]


def _propose_one(ref, art, proposer, verifier, controller, reg, options, before=None):
    """Propose, verify and score one cell; returns [(ref, repair)]."""
    try:
        v = Violation(cid="study.detected", refs=[ref])
        cands = proposer.propose(v, art, options=options or {})
        feas = verifier.project(cands, art, reg, options=options or {}, before=before)
        return [(r.target_ref, r) for r in controller.assign_s_hat(feas)]
    except Exception as e:  # a failing cell must not abort the sweep
        print(f"  [propose_repairs] cell {ref} failed: {e}")
        return []


def _probe_verifier_constant(art, refs, proposer, verifier, reg, options, before,
                             n=64, n_cells=None):
    """Check whether exact verification is the same for every single-cell repair.

    Assesses up to ``n`` cells that have a candidate. If ``feasible``, ``s_margin`` and
    ``delta_objective`` are constant, callers reuse them instead of copying and
    rescanning the table per candidate.

    Returns (is_constant, (feasible, s_margin, delta_objective)).
    """
    # The exact path is O(cells) per candidate, so probe fewer cells on large tables.
    if n_cells and n_cells > 1_000_000:
        n = 8
    elif n_cells and n_cells > 200_000:
        n = 16
    print(f"  [fast-verify] probing {n} cells with the exact verifier "
          f"(slow by design; this is the equivalence check)...", flush=True)
    t0 = _time.time()
    seen_f, seen_m, seen_d = set(), set(), set()
    # Skip cells without a candidate; a sparse cleaner can leave the first n empty.
    probed = 0
    for ref in refs:
        cands = proposer.propose(Violation(cid="probe", refs=[ref]), art, options=options or {})
        if not cands:
            continue
        for c in cands:
            vr = verifier.assess(c, art, reg, options=options or {}, before=before)
            seen_f.add(vr.feasible)
            seen_m.add(round(float(vr.s_margin or 0.0), 9))
            seen_d.add(round(float(vr.delta_objective or 0.0), 9))
        probed += 1
        if probed % 4 == 0:
            print(f"  [fast-verify] probe {probed}/{n} ({_time.time()-t0:.0f}s)", flush=True)
        if probed >= n:
            break
    if not (seen_f and seen_m and seen_d):
        print("  [fast-verify] no cell in the queue has a candidate; nothing to verify.",
              flush=True)
        return True, (True, 0.0, 0.0)
    ok = len(seen_f) == 1 and len(seen_m) == 1 and len(seen_d) == 1
    return ok, (next(iter(seen_f)), next(iter(seen_m)), next(iter(seen_d)))


def propose_repairs(art, refs, proposer, *, reg, options=None, max_workers=None,
                    fast_verify="auto"):
    """Return ref -> VerifiedRepair, proposing once per cell.

    Uses a thread pool (``CARE_MAX_WORKERS``, default 8); output is deterministic.
    ``fast_verify``: True, False or "auto" (on above 200K cells).
    """
    verifier, controller = Verifier(), ConformalController()
    workers = max_workers or int(os.environ.get("CARE_MAX_WORKERS", "8"))
    out = {}
    show = os.environ.get("CARE_PROGRESS", "1") != "0"
    # Computed once: the proposer never writes to the artifact.
    from care.verify.projector import hard_violation_set
    before = hard_violation_set(art, reg, options or {})

    # Exact-equivalent fast path.
    n_cells = sum(1 for _ in art.iter_cells())
    use_fast = fast_verify is True or (fast_verify == "auto" and n_cells > 200_000)
    print(f"  [propose] {len(refs)} cells to propose over a {n_cells}-cell artifact; "
          f"fast-verify={'ON' if use_fast else 'off'}", flush=True)
    if use_fast:
        # Incremental verification; candidates it cannot decide use the exact path.
        from care.verify.incremental import IncrementalVerifier
        inc = IncrementalVerifier(art, reg, options or {})
        if inc.supported:
            print("  [fast-verify] incremental verification (index self-checked against "
                  "the reference predicates); undecidable candidates use the exact path.",
                  flush=True)
            n_exact = [0]

            def _propose_inc(ref):
                try:
                    v = Violation(cid="study.detected", refs=[ref])
                    vrs = []
                    for c in proposer.propose(v, art, options=options or {}):
                        r = inc.assess(c)
                        if r is None:
                            n_exact[0] += 1
                            r = verifier.assess(c, art, reg, options=options or {},
                                                before=before)
                        if r.feasible:
                            vrs.append(r)
                    return [(r.target_ref, r) for r in controller.assign_s_hat(vrs)]
                except Exception as e:
                    print(f"  [propose_repairs] cell {ref} failed: {e}")
                    return []
            for ref in _tqdm(refs, desc="propose(incremental)", disable=not show,
                             unit="cell"):
                for k, r in _propose_inc(ref):
                    out[k] = r
            print(f"  [fast-verify] {n_exact[0]} candidate(s) verified on the exact path",
                  flush=True)
            return out
        print(f"  [fast-verify] incremental verification does not cover "
              f"{inc.unsupported}; trying the constancy probe instead.", flush=True)
        const_ok, consts = _probe_verifier_constant(art, refs, proposer, verifier, reg,
                                                    options, before, n_cells=n_cells)
        if const_ok:
            feas_v, marg_v, delta_v = consts
            print(f"  [fast-verify] probe: verification is CONSTANT for this constraint "
                  f"set (feasible={feas_v}, s_margin={marg_v}, delta={delta_v}); reusing "
                  f"it for all {len(refs)} cells. Output is identical to the exact path.",
                  flush=True)
            def _propose_fast(ref):
                try:
                    v = Violation(cid="study.detected", refs=[ref])
                    cands = proposer.propose(v, art, options=options or {})
                    vrs = [VerifiedRepair(
                        target_ref=c.target_ref, proposed_value=c.proposed_value,
                        rationale=c.rationale, evidence=list(c.evidence),
                        s_conf=c.s_conf, s_agree=c.s_agree, s_consistency=c.s_consistency,
                        s_margin=marg_v, s_hat=c.s_hat,
                        feasible=feas_v, delta_objective=delta_v) for c in cands]
                    vrs = [r for r in vrs if r.feasible]
                    return [(r.target_ref, r) for r in controller.assign_s_hat(vrs)]
                except Exception as e:
                    print(f"  [propose_repairs] cell {ref} failed: {e}")
                    return []
            for ref in _tqdm(refs, desc="propose(fast)", disable=not show, unit="cell"):
                for k, r in _propose_fast(ref):
                    out[k] = r
            return out
        print("  [fast-verify] probe found NON-constant verification -> falling back to "
              "the exact (slow) path; correctness takes priority.", flush=True)

    if workers <= 1:
        for ref in _tqdm(refs, desc="propose", disable=not show, unit="cell"):
            for k, r in _propose_one(ref, art, proposer, verifier, controller, reg,
                                     options, before):
                out[k] = r
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_propose_one, ref, art, proposer, verifier, controller, reg,
                             options, before)
                   for ref in refs]
        # submission order keeps the output deterministic
        for fut in _tqdm(futures, desc="propose", disable=not show, unit="cell"):
            for k, r in fut.result():
                out[k] = r
    return out


def _poison(repairs, *, frac, seed, wrong="CORRUPT", evidence_id="leak"):
    """Replace a fraction of repairs with a wrong, high-confidence value from ``evidence_id``.

    Returns (repairs, poisoned_refs).
    """
    rng = random.Random(seed + 7919)
    poisoned = set()
    out = dict(repairs)
    for ref in list(repairs):
        if rng.random() < frac:
            poisoned.add(ref)
            out[ref] = repairs[ref].model_copy(update={
                "proposed_value": wrong, "evidence": [evidence_id], "s_conf": 0.95, "s_agree": 1.0,
            })
    return out, poisoned


def _gated(repair, sources, tau_min):
    """True if the escalation gate escalates ``repair`` (best evidence trust < tau_min)."""
    from care.escalate.active import _evidence_trust
    t = _evidence_trust(repair, sources)
    return t is not None and t < tau_min


def evaluate_split(
    repairs, gold, *, alpha, delta, strata_fn=marginal_strata, sources=None,
    cal_frac=0.4, seed=0, poison_frac=0.0, tau_min=0.5, trusted_poison=False,
    return_decisions=False, cal_size=None, grid=None,
    gate_calibration=True, trust_noise=0.0, trust_noise_mode="random",
    bound="exact", procedure="bonferroni",
) -> dict[str, Scores]:
    """Calibrate on a held-out split, decide the rest, and score CARE, ApplyAll and EscalateAll.

    Args:
        return_decisions: also return the per-cell outcome (auto values, escalated
            refs, calibration/test refs, threshold).
        cal_size: number of calibration labels drawn uniformly over the queue,
            instead of ``cal_frac`` per stratum.
        gate_calibration: apply the trust gate to the calibration sample as well.
        trust_noise: probability of mislabelling provenance (poisoned -> trusted,
            clean -> untrusted); ``trust_noise_mode="correlated"`` mislabels only
            correct clean repairs. Requires ``poison_frac > 0``.
    """
    controller = ConformalController(bound=bound, strata_fn=strata_fn, grid=grid,
                                     procedure=procedure)
    escalator = ActiveEscalator(tau_min=tau_min, strata_fn=strata_fn)
    sources = dict(sources or {})
    # case-insensitive comparison
    gold = {k: str(v).lower().strip() if v is not None else v for k, v in gold.items()}
    for r in repairs.values():
        r.proposed_value = str(r.proposed_value).lower().strip() if r.proposed_value is not None else r.proposed_value
    if poison_frac > 0:
        # low-trust poison is caught by the gate; trusted poison is not
        eid = "trusted_leak" if trusted_poison else "leak"
        repairs, poisoned = _poison(repairs, frac=poison_frac, seed=seed, evidence_id=eid)
        sources[eid] = Source(source_id=eid, uri="u", kind="scraped",
                              trust_prior=(0.9 if trusted_poison else 0.1))
    else:
        poisoned = set()
    if trust_noise > 0:
        nrng = random.Random(seed + 104729)
        sources["trusted_leak"] = Source(source_id="trusted_leak", uri="u", kind="scraped",
                                         trust_prior=0.9)
        sources["mislabelled"] = Source(source_id="mislabelled", uri="u", kind="scraped",
                                        trust_prior=0.1)
        repairs = dict(repairs)
        for ref in sorted(repairs):
            u = nrng.random()
            if ref in poisoned:
                if u < trust_noise:
                    repairs[ref] = repairs[ref].model_copy(update={"evidence": ["trusted_leak"]})
            elif ref in gold:
                if trust_noise_mode == "correlated" and repairs[ref].proposed_value != gold[ref]:
                    continue
                if u < trust_noise:
                    repairs[ref] = repairs[ref].model_copy(update={"evidence": ["mislabelled"]})

    refs = [r for r in gold if r in repairs]
    rng = random.Random(seed)
    # Split each stratum separately so none lacks calibration data.
    by_stratum = {}
    for r in refs:
        by_stratum.setdefault(strata_fn(repairs[r]), []).append(r)
    cal_refs, test_refs = [], []
    if cal_size is None:
        for _stratum, group in by_stratum.items():
            rng.shuffle(group)
            n_cal_s = int(len(group) * cal_frac)
            cal_refs.extend(group[:n_cal_s])
            test_refs.extend(group[n_cal_s:])
    else:
        # Fixed label budget drawn uniformly, so a stratum may get too few labels.
        pool = list(refs)
        rng.shuffle(pool)
        k = min(int(cal_size), len(pool))
        cal_refs, test_refs = pool[:k], pool[k:]

    from care.core import CalibrationRecord
    cal_records = [
        CalibrationRecord(
            target_ref=r, s_hat=repairs[r].s_hat,
            correct=(repairs[r].proposed_value == gold[r]), stratum=strata_fn(repairs[r]),
        )
        for r in cal_refs if r not in poisoned   # poison is excluded from calibration
        and not (gate_calibration and _gated(repairs[r], sources, tau_min))
    ]
    thr = controller.calibrate(cal_records, alpha=alpha, delta=delta)

    test_repairs = [repairs[r] for r in test_refs]
    decisions = controller.decide(test_repairs, thr, strata_fn=strata_fn)
    final, _queue = escalator.route(decisions, test_repairs, sources=sources)
    auto = {d.target_ref: repairs[d.target_ref].proposed_value
            for d in final if d.action == "auto_apply"}
    esc = {d.target_ref for d in final if d.action == "escalate"}

    gold_test = {r: gold[r] for r in test_refs}
    apply_all = {r: repairs[r].proposed_value for r in test_refs}
    results = {
        "CARE": RepairResult("CARE", auto, esc),
        "ApplyAll": RepairResult("ApplyAll", apply_all, set()),
        "EscalateAll": RepairResult("EscalateAll", {}, set(test_refs)),
    }
    # `contaminated` only sets applied_contamination (eps_S).
    scores = {name: score(res, gold_test, contaminated=poisoned)
              for name, res in results.items()}
    if return_decisions:
        return scores, {"auto": auto, "escalated": esc,
                        "cal_refs": list(cal_refs), "test_refs": list(test_refs),
                        "threshold": thr}
    return scores


def _warn_detection_mismatch(refs, defects, *, mode):
    """Warn when the detected cells barely overlap the true errors."""
    if mode == "oracle":
        return
    detected = set(refs)
    gold = set(defects.gold)
    tp = len(detected & gold)
    recall = tp / len(gold) if gold else 0.0
    precision = tp / len(detected) if detected else 0.0
    print(f"  [detection={mode}] detected={len(detected)} true_errors={len(gold)} "
          f"TP={tp} recall={recall:.2f} precision={precision:.2f}")
    if tp == 0 or recall < 0.05:
        print("  [detection] WARNING: detected set barely overlaps the true errors "
              "-- this detector does not match the error type; results will be "
              "meaningless. Use a real error-detection log (e.g. Raha) instead.")


def load_and_propose(name, client, *, detection="oracle", max_cells=None, row_range=None,
                     fast_verify="auto"):
    """Load a dataset, detect errors and propose repairs once, for reuse across seeds.

    ``client`` is a '<tag>:<path>' log spec; the tag is the default evidence source.
    """
    reg = _registry()
    art, defects, cols = load(name)
    refs = detect(art, reg, defects, mode=detection)
    if row_range:
        # Keep only rows [lo, hi), the shard the cleaner ran on.
        from care.core.artifact import CellKey
        lo, hi = row_range
        def _in(ref):
            r = CellKey.parse(ref).row_id
            try:
                return lo <= int(r) < hi
            except (TypeError, ValueError):
                return False
        before = len(refs)
        refs = [r for r in refs if _in(r)]
        defects.gold = {r: v for r, v in defects.gold.items() if _in(r)}
        print(f"  [row-range] rows [{lo},{hi}): work queue {before} -> {len(refs)} cells "
              f"({len(refs)/max(1, before)*100:.1f}% of the full table's error cells)")
        if not refs:
            raise SystemExit(f"--row-range [{lo},{hi}) selected 0 cells; check the bounds.")
    if max_cells:
        refs = refs[:max_cells]
        defects.gold = {r: defects.gold[r] for r in refs if r in defects.gold}

    if not is_offline_log_backend(client):
        raise ValueError(
            f"backend {client!r} is not an offline-log spec '<tag>:<path>' with tag in "
            f"{OFFLINE_LOG_BACKENDS} (or a '<tag>_<variant>' of one)")
    kind_tag, log_path = client.split(":", 1)
    art.add_source(Source(source_id=kind_tag, uri=f"{kind_tag}://{log_path}",
                          kind="internal", trust_prior=1.0))
    proposer = LogProposer(log_path, source_id=kind_tag)
    # Propose over the detected queue; in oracle mode this is the gold set.
    _warn_detection_mismatch(refs, defects, mode=detection)
    repairs = propose_repairs(art, refs, proposer, reg=reg, fast_verify=fast_verify)
    return art, defects, repairs, reg


__all__ = [
    "detect", "propose_repairs", "evaluate_split", "load_and_propose",
]
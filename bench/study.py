"""bench.study -- the governed-correction protocol on real corpora.

Design (the controlled comparison for the paper):
  * Detection is held fixed -- oracle (the error cells are given), a real
    detector's exported cell log (``detection="log:<path>"``), or the enabled
    hard constraints (``detection="constraints"``) -- so the experiment isolates
    CARE's contribution: *governed correction*.
  * The proposer is an external cleaner replayed from its offline repair log
    (``<tag>:<path>``, see ``bench.baran_proposer``), held fixed across CARE and
    the ungoverned baseline, so the only variable is the governance: conformal
    gate + tau-aware escalation.
  * Coverage is measured the standard conformal way: split the detected errors
    into a labelled calibration set and a test set, calibrate the threshold on
    calibration, and report realized error on the auto-applied test cells -- vs
    the budget alpha, with CIs over many splits/seeds.

Baselines produced from the same proposals: CARE (gated), LLMOnly (apply all),
NoOp (escalate all).
"""

from __future__ import annotations

import os
import random
import time as _time
from concurrent.futures import ThreadPoolExecutor

try:  # optional progress bar; degrades to a no-op passthrough if absent
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
from bench.baran_proposer import BaranProposer
from bench.metrics import RepairResult, Scores, score


# Backend prefixes handled by the offline-log proposer (`<tag>:<path>`), i.e. any
# external cleaner that exports the log contract row_id,column,value[,confidence][,source].
# SINGLE SOURCE OF TRUTH -- bench.run_study imports this, so adding a new cleaner is a
# one-line change here rather than two hardcoded tuples that drift apart.
OFFLINE_LOG_BACKENDS = ("baran", "retclean", "jellyfish", "bclean", "gpt4o", "holoclean")


def is_offline_log_backend(spec) -> bool:
    """True if `spec` is a '<tag>:<path>' offline-log backend spec.

    Variant tags are accepted: `baran_raha` (Baran re-run under a real detector) and
    `bclean_keep` (BClean with explicit keep-decisions) are the same offline-log
    proposer with a different protocol, so a tag matches on its first `_`-segment
    as well as exactly. `prepare_log.py` names its output after the log file, so
    these tags arrive here verbatim from a copy-pasted hint.
    """
    if not isinstance(spec, str):
        return False
    tag = spec.split(":", 1)[0]
    return tag in OFFLINE_LOG_BACKENDS or tag.split("_", 1)[0] in OFFLINE_LOG_BACKENDS


def _registry() -> ConstraintRegistry:
    clear_registry()
    for c in ALL_CONSTRAINTS:
        register_constraint(c)
    # completeness floor is the only hard constraint here: filling a value can
    # never violate it, so the verifier never spuriously blocks a proposer's correction.
    return ConstraintRegistry.from_requirements(
        DQRequirements(enabled={"dq.completeness.floor"}), strict=True
    )


def _load_detection_log(path: str) -> set[str]:
    """Read a ``row_id,column`` detected-cells CSV (e.g. an exported Raha
    detection run) and return the set of ``"{row_id}::{column}"`` refs.

    This is the non-oracle-detection counterpart to ``BaranProposer``'s offline
    correction log: a real detector runs natively in its own repo/environment and
    exports its flagged cells here, so CARE never imports the detector's code."""
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
    """Return the work-queue refs (error cells).

    ``mode``: ``"oracle"`` (gold error cells, the default), ``"constraints"``
    (Auditor-flagged hard-constraint violations -- only meaningful if the enabled
    constraints actually correlate with the dataset's error type; see
    ``_warn_detection_mismatch``), or ``"log:<path>"`` to use a real detector's
    exported cell log (e.g. Raha) as the work queue.
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
    """Propose+verify+score a single cell. Returns list[(ref, repair)] (possibly empty)."""
    try:
        v = Violation(cid="study.detected", refs=[ref])
        cands = proposer.propose(v, art, options=options or {})
        feas = verifier.project(cands, art, reg, options=options or {}, before=before)
        return [(r.target_ref, r) for r in controller.assign_s_hat(feas)]
    except Exception as e:  # one cell's failure must not abort the sweep
        print(f"  [propose_repairs] cell {ref} failed: {e}")
        return []


def _probe_verifier_constant(art, refs, proposer, verifier, reg, options, before,
                             n=64, n_cells=None):
    """Check whether verification is CONSTANT for this (artifact, constraint set).

    Measured fact: under the benchmark's constraint set (only the completeness
    floor is enabled as hard), `Verifier.assess` returns the same `feasible`,
    `s_margin` and `delta_objective` for every single-cell repair -- a value repair
    cannot flip a floor that is already violated by thousands of nulls. Yet computing
    them costs a DEEP COPY of the artifact plus three full-table constraint scans per
    candidate: 85 ms/cell on hospital (19K cells) and ~25 s/cell on tax (3M cells),
    i.e. ~840 hours for a single tax sweep (121K cells x ~25 s).

    So we probe `n` cells the slow, exact way. If all three fields are constant, the
    remaining cells can reuse those constants -- provably identical output, no
    approximation. If they are NOT constant, we say so and fall back to the exact path
    rather than silently trading correctness for speed.

    Returns (is_constant, (feasible, s_margin, delta_objective)).
    """
    # The probe uses the SLOW exact path, which costs ~O(cells) per candidate. On a
    # 3M-cell artifact that is ~25 s/cell, so a 64-cell probe would stall for ~27
    # minutes with no output. Scale the sample down for large artifacts: a constancy
    # check does not need many samples, and any non-constancy shows up immediately.
    if n_cells and n_cells > 1_000_000:
        n = 8
    elif n_cells and n_cells > 200_000:
        n = 16
    print(f"  [fast-verify] probing {n} cells with the exact verifier "
          f"(slow by design; this is the equivalence check)...", flush=True)
    t0 = _time.time()
    seen_f, seen_m, seen_d = set(), set(), set()
    for i, ref in enumerate(list(refs)[:n], 1):
        for c in proposer.propose(Violation(cid="probe", refs=[ref]), art, options=options or {}):
            vr = verifier.assess(c, art, reg, options=options or {}, before=before)
            seen_f.add(vr.feasible)
            seen_m.add(round(float(vr.s_margin or 0.0), 9))
            seen_d.add(round(float(vr.delta_objective or 0.0), 9))
        if i % 4 == 0:
            print(f"  [fast-verify] probe {i}/{n} ({_time.time()-t0:.0f}s)", flush=True)
    ok = len(seen_f) == 1 and len(seen_m) == 1 and len(seen_d) == 1
    if not (seen_f and seen_m and seen_d):
        return False, None
    return ok, (next(iter(seen_f)), next(iter(seen_m)), next(iter(seen_d)))


def propose_repairs(art, refs, proposer, *, reg, options=None, max_workers=None,
                    fast_verify="auto"):
    """ref -> VerifiedRepair (proposed_value, s_hat, evidence), proposed once.

    Cells are dispatched to a thread pool (workers default to env CARE_MAX_WORKERS,
    else 8); results are collected in submission order, so the output is
    deterministic regardless of the worker count.
    """
    verifier, controller = Verifier(), ConformalController()
    workers = max_workers or int(os.environ.get("CARE_MAX_WORKERS", "8"))
    out = {}
    show = os.environ.get("CARE_PROGRESS", "1") != "0"
    # Hoisted out of the per-cell loop: this scans every cell in the artifact and is
    # invariant while proposing (I1 -- the proposer never writes to the artifact).
    # Recomputing it per cell made tax ~25 s/cell (~840 hours projected over the 121K-cell queue).
    from care.verify.projector import hard_violation_set
    before = hard_violation_set(art, reg, options or {})

    # --- optional exact-equivalent fast path (see _probe_verifier_constant) ---
    n_cells = sum(1 for _ in art.iter_cells())
    use_fast = fast_verify is True or (fast_verify == "auto" and n_cells > 200_000)
    print(f"  [propose] {len(refs)} cells to propose over a {n_cells}-cell artifact; "
          f"fast-verify={'ON' if use_fast else 'off'}", flush=True)
    if use_fast:
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
                        s_llm=c.s_llm, s_agree=c.s_agree, s_consistency=c.s_consistency,
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
        # iterate in submission order (deterministic); bar advances as each future is collected
        for fut in _tqdm(futures, desc="propose", disable=not show, unit="cell"):
            for k, r in fut.result():
                out[k] = r
    return out


def _poison(repairs, *, frac, seed, wrong="CORRUPT", evidence_id="leak"):
    """Mark a fraction of repairs as poisoned: a wrong value with high fake
    confidence, attributed to ``evidence_id``.

    Two regimes, selected by the caller via ``evidence_id`` + the trust it
    registers for that source:
      * low-trust source  (detectable poison)  -> escalation gate removes it
        (Theorem 1: retained validity, eps_S -> 0).
      * high-trust source (undetectable poison) -> gate passes it, so it can be
        auto-applied and contaminates the set (Theorem 2: graceful degradation,
        realized error <= alpha + eps_S(1-alpha)).
    """
    rng = random.Random(seed + 7919)
    poisoned = set()
    out = dict(repairs)
    for ref in list(repairs):
        if rng.random() < frac:
            poisoned.add(ref)
            out[ref] = repairs[ref].model_copy(update={
                "proposed_value": wrong, "evidence": [evidence_id], "s_llm": 0.95, "s_agree": 1.0,
            })
    return out, poisoned


def evaluate_split(
    repairs, gold, *, alpha, delta, strata_fn=marginal_strata, sources=None,
    cal_frac=0.4, seed=0, poison_frac=0.0, tau_min=0.5, trusted_poison=False,
    return_decisions=False, cal_size=None, grid=None,
) -> dict[str, Scores]:
    """Calibrate on a held-out split, then decide the rest; score CARE/LLMOnly/NoOp.

    ``return_decisions=True`` additionally returns the raw per-cell outcome
    (``auto`` values, ``escalated`` refs, and the calibration/test partition) as a
    second element. Nothing else changes. This exists so downstream-utility
    measurement can MATERIALISE the table CARE would actually have written, using
    this exact certifier rather than a second implementation of it that could
    silently drift from the numbers reported everywhere else.
    """
    controller = ConformalController(bound="exact", strata_fn=strata_fn, grid=grid)
    escalator = ActiveEscalator(tau_min=tau_min, strata_fn=strata_fn)
    sources = dict(sources or {})
    # Normalize casing so a proposer is not penalised for case alone
    gold = {k: str(v).lower().strip() if v is not None else v for k, v in gold.items()}
    for r in repairs.values():
        r.proposed_value = str(r.proposed_value).lower().strip() if r.proposed_value is not None else r.proposed_value
    if poison_frac > 0:
        # trusted_poison=False -> low-trust source (gate catches it: Theorem 1).
        # trusted_poison=True  -> high-trust source (gate passes it: Theorem 2).
        eid = "trusted_leak" if trusted_poison else "leak"
        repairs, poisoned = _poison(repairs, frac=poison_frac, seed=seed, evidence_id=eid)
        sources[eid] = Source(source_id=eid, uri="u", kind="scraped",
                              trust_prior=(0.9 if trusted_poison else 0.1))
    else:
        poisoned = set()

    refs = [r for r in gold if r in repairs]
    rng = random.Random(seed)
    # Per-stratum calibration/test split: each stratum contributes its own 40%,
    # so a stratum can't be starved of calibration mass by an unlucky global draw
    # (which would leave its conformal threshold uncalibrated -> escalate-all).
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
        # Fixed LABEL BUDGET rather than a fixed fraction: an operator who can afford
        # `cal_size` reviewed cells labels that many, uniformly at random over the
        # queue, and everything else is decided. The sample is deliberately NOT
        # stratified-proportional -- a real operator drawing n cells cannot guarantee
        # each column its quota, and a stratum that draws too few labels should fail
        # to certify. That failure is the phenomenon being measured, not a nuisance.
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
        for r in cal_refs if r not in poisoned   # injected poison is a safety channel, not calibration
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
        "LLMOnly": RepairResult("LLMOnly", apply_all, set()),
        "NoOp": RepairResult("NoOp", {}, set(test_refs)),
    }
    # `contaminated=poisoned` only populates each Scores.applied_contamination
    # (eps_S) -- for CARE it is the corrupted fraction of the AUTO-applied set (the
    # Theorem 2 quantity); for LLMOnly it is the corrupted fraction of the whole
    # test set (apply-all applies every poison). No other metric is affected.
    scores = {name: score(res, gold_test, contaminated=poisoned)
              for name, res in results.items()}
    if return_decisions:
        return scores, {"auto": auto, "escalated": esc,
                        "cal_refs": list(cal_refs), "test_refs": list(test_refs),
                        "threshold": thr}
    return scores


def _warn_detection_mismatch(refs, defects, *, mode):
    """Loudly flag a detector whose flagged cells barely overlap the true errors.

    Non-oracle detection is only meaningful if the detected set actually contains
    real errors. The completeness-floor constraint, for instance, flags NULL cells
    -- which on datasets whose errors are non-null corruptions (e.g. hospital) has
    ~0 overlap with the true error set, making any downstream number garbage. This
    guard makes that failure loud instead of silent."""
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
    """Load a dataset, detect errors, and propose repairs once (reusable across seeds).

    ``client`` is an offline-log backend spec ``"<tag>:<path>"`` (see
    ``OFFLINE_LOG_BACKENDS``); the tag becomes the default source_id and the log's
    optional per-cell ``source`` column carries finer provenance.
    """
    reg = _registry()
    art, defects, cols = load(name)
    refs = detect(art, reg, defects, mode=detection)
    if row_range:
        # Row sharding. A proposer run on rows [lo, hi) has no opinion about any other
        # row; leaving those rows in the work queue would score "the proposer was never
        # asked" as "the proposer was escalated", driving human cost to ~1.0 for reasons
        # that have nothing to do with the proposer or the certifier.
        #
        # The range comes from the caller (the shard bounds actually executed), NOT from
        # the log's contents -- see the --row-range help text in run_study.py for why
        # that distinction matters.
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
    proposer = BaranProposer(log_path, source_id=kind_tag)
    # Propose over the DETECTED work queue (`refs`), not the oracle gold set;
    # in oracle mode refs == list(defects.gold).
    _warn_detection_mismatch(refs, defects, mode=detection)
    repairs = propose_repairs(art, refs, proposer, reg=reg, fast_verify=fast_verify)
    return art, defects, repairs, reg


__all__ = [
    "detect", "propose_repairs", "evaluate_split", "load_and_propose",
]
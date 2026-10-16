"""IncrementalVerifier.assess agrees with Verifier.assess on random single-cell repairs."""
from __future__ import annotations

import random

import pytest

from care.core.artifact import Cell, DataArtifact, DQRequirements, MetadataTriple
from care.core.registry import ConstraintRegistry, clear_registry, register_constraint
from care.core.repair import RepairCandidate
from care.predicates import ALL_CONSTRAINTS
from care.verify.incremental import IncrementalVerifier
from care.verify.verifier import Verifier

COLS = ["a", "b", "c", "d"]


def _registry(enabled):
    clear_registry()
    for c in ALL_CONSTRAINTS:
        register_constraint(c)
    return ConstraintRegistry.from_requirements(DQRequirements(enabled=set(enabled)),
                                                strict=True)


def _table(rng, n_rows, domain, p_null, p_required):
    art = DataArtifact(artifact_id="t")
    for i in range(n_rows):
        for col in COLS:
            v = None if rng.random() < p_null else rng.choice(domain)
            art.set_cell(Cell(row_id=str(i), col=col, value=v,
                              is_required=rng.random() < p_required))
    return art


def _fds(rng):
    out = []
    for _ in range(rng.randint(1, 3)):
        det = rng.sample(COLS, rng.randint(1, 2))
        rest = [c for c in COLS if c not in det]
        deps = rng.sample(rest, rng.randint(1, len(rest)))
        out.append((det, deps))
    return out


def _compare(art, reg, options, cands):
    exact = Verifier()
    inc = IncrementalVerifier(art, reg, options)
    assert inc.supported
    decided = 0
    for c in cands:
        a = exact.assess(c, art, reg, options=options)
        b = inc.assess(c)
        if b is None:
            continue
        decided += 1
        assert b.feasible == a.feasible, (c.target_ref, c.proposed_value)
        assert b.s_margin == a.s_margin, (c.target_ref, c.proposed_value)
        assert b.delta_objective == a.delta_objective, (c.target_ref, c.proposed_value)
    return decided


@pytest.mark.parametrize("seed", range(60))
def test_incremental_matches_exact(seed):
    rng = random.Random(seed)
    domain = ["x", "y", "z"][: rng.randint(2, 3)]
    art = _table(rng, rng.randint(4, 14), domain, p_null=0.15, p_required=0.6)
    enabled = {"dq.consistency.fd"}
    if rng.random() < 0.7:
        enabled.add("dq.completeness.floor")
    if rng.random() < 0.5:
        enabled.add("dq.completeness")
    reg = _registry(enabled)
    options = {"dq.consistency.fd": {"fds": _fds(rng)}}
    cands = []
    for _ in range(40):
        r, col = rng.randrange(len(art.cells) // len(COLS)), rng.choice(COLS)
        v = None if rng.random() < 0.1 else rng.choice(domain + ["w"])
        cands.append(RepairCandidate(target_ref=f"{r}::{col}", proposed_value=v))
    assert _compare(art, reg, options, cands) == len(cands)


def test_feasibility_actually_varies():
    """The generator produces both feasible and infeasible candidates."""
    seen = set()
    for seed in range(60):
        rng = random.Random(seed)
        art = _table(rng, 10, ["x", "y"], 0.1, 0.5)
        reg = _registry({"dq.consistency.fd"})
        options = {"dq.consistency.fd": {"fds": _fds(rng)}}
        exact = Verifier()
        for _ in range(20):
            r, col = rng.randrange(10), rng.choice(COLS)
            c = RepairCandidate(target_ref=f"{r}::{col}", proposed_value=rng.choice("xyw"))
            seen.add(exact.assess(c, art, reg, options=options).feasible)
    assert seen == {True, False}


def test_falls_back_on_what_it_cannot_decide():
    rng = random.Random(0)
    art = _table(rng, 6, ["x", "y"], 0.1, 0.5)
    reg = _registry({"dq.consistency.fd", "dq.completeness.floor"})
    inc = IncrementalVerifier(art, reg, {"dq.consistency.fd": {"fds": [(["a"], ["b"])]}})
    assert inc.assess(RepairCandidate(target_ref="99::a", proposed_value="x")) is None
    triple = MetadataTriple(subject="t", predicate="license", object="cc", source_id="s")
    assert inc.assess(RepairCandidate(target_ref="meta:t:license",
                                      proposed_value=triple)) is None
    unsupported = IncrementalVerifier(art, _registry({"dq.traceability"}), {})
    assert not unsupported.supported
    assert unsupported.assess(RepairCandidate(target_ref="0::a", proposed_value="x")) is None


def _cfds(rng, domain):
    out = []
    for _ in range(rng.randint(1, 3)):
        det = rng.sample(COLS, rng.randint(1, 2))
        rest = [c for c in COLS if c not in det]
        deps = rng.sample(rest, rng.randint(1, len(rest)))
        # left-hand pattern: constants on some determinant (or other) columns
        lhs = {c: rng.choice(domain) for c in rng.sample(COLS, rng.randint(0, 2))
               if c not in deps}
        # right-hand pattern: a constant on some dependents
        rhs = {d: rng.choice(domain) for d in deps if rng.random() < 0.4}
        out.append((det, deps, lhs, rhs))
    return out


@pytest.mark.parametrize("seed", range(60))
def test_incremental_matches_exact_with_cfds(seed):
    rng = random.Random(10_000 + seed)
    domain = ["x", "y", "z"][: rng.randint(2, 3)]
    art = _table(rng, rng.randint(4, 14), domain, p_null=0.15, p_required=0.6)
    enabled = {"dq.consistency.cfd"}
    options = {"dq.consistency.cfd": {"cfds": _cfds(rng, domain)}}
    if rng.random() < 0.5:
        enabled.add("dq.consistency.fd")
        options["dq.consistency.fd"] = {"fds": _fds(rng)}
    if rng.random() < 0.5:
        enabled.add("dq.completeness.floor")
    reg = _registry(enabled)
    cands = []
    for _ in range(40):
        r, col = rng.randrange(len(art.cells) // len(COLS)), rng.choice(COLS)
        v = None if rng.random() < 0.1 else rng.choice(domain + ["w"])
        cands.append(RepairCandidate(target_ref=f"{r}::{col}", proposed_value=v))
    assert _compare(art, reg, options, cands) == len(cands)


def test_cfd_with_empty_patterns_is_the_fd():
    """A CFD with no constants must give exactly the FD predicate's verdicts."""
    rng = random.Random(7)
    art = _table(rng, 12, ["x", "y"], 0.1, 0.5)
    fds = [(["a"], ["b", "c"]), (["d"], ["a"])]
    reg_fd = _registry({"dq.consistency.fd"})
    fd = Verifier()
    reg_cfd = _registry({"dq.consistency.cfd"})
    for _ in range(40):
        c = RepairCandidate(target_ref=f"{rng.randrange(12)}::{rng.choice(COLS)}",
                            proposed_value=rng.choice("xyw"))
        a = fd.assess(c, art, reg_fd, options={"dq.consistency.fd": {"fds": fds}})
        b = fd.assess(c, art, reg_cfd, options={"dq.consistency.cfd": {
            "cfds": [(d, y, {}, {}) for d, y in fds]}})
        assert (a.feasible, a.s_margin) == (b.feasible, b.s_margin)

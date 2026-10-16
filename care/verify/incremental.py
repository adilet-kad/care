"""Exact incremental verification of single-cell repairs.

Computes the same feasibility, ``s_margin`` and ``delta_objective`` as
``Verifier.assess`` from indexes built once per sweep, without copying or
rescanning the table. Supported: dq.completeness(.floor), dq.consistency.fd and
dq.consistency.cfd (FDs and CFDs are indexed by determinant group, with a count
of each dependent value per group). ``assess`` returns None for metadata repairs,
cell-creating writes and unsupported predicates; the caller then uses the exact
path. At construction the index is checked against ``hard_violation_set``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from care.core.artifact import CellKey, DataArtifact, MetadataTriple
from care.core.constraints import Params
from care.core.registry import ConstraintRegistry
from care.core.repair import RepairCandidate, VerifiedRepair
from care.verify.projector import edit_cost, hard_violation_set

Options = Mapping[str, Mapping[str, Any]]

FLOOR = "dq.completeness.floor"
COMPLETENESS = "dq.completeness"
FD = "dq.consistency.fd"
CFD = "dq.consistency.cfd"
SUPPORTED_HARD = {FLOOR, FD, CFD}
SUPPORTED_SOFT = {COMPLETENESS}


def _params(reg: ConstraintRegistry, cid: str, options: Options) -> Params:
    return Params.for_cid(cid, reg.requirements, **dict(options.get(cid, {})))


def _distinct_after(cnt: Counter, remove: Any = None, add: Any = None) -> int:
    """Number of distinct values in ``cnt`` after removing one occurrence of
    ``remove`` and adding one of ``add`` (None = nothing)."""
    n = len(cnt)
    if remove == add:
        return n
    if remove is not None and cnt.get(remove, 0) == 1:
        n -= 1
    if add is not None and cnt.get(add, 0) == 0:
        n += 1
    return n


class _FDIndex:
    """Group index for one FD ``det -> deps``, or one CFD when patterns are given.

    ``lhs`` restricts the index to rows whose columns equal its constants; a dependent
    column with a constant in ``rhs`` is checked per row instead of per group.
    """

    def __init__(self, cid: str, det: list[str], deps: list[str], rows: dict[str, dict],
                 lhs: dict | None = None, rhs: dict | None = None):
        self.cid = cid
        self.det, self.deps = list(det), list(deps)
        self.lhs, self.rhs = dict(lhs or {}), dict(rhs or {})
        self.var_deps = [d for d in self.deps if d not in self.rhs]
        self.const_deps = [d for d in self.deps if d in self.rhs]
        self.cols = set(self.det) | set(self.deps) | set(self.lhs)
        self.members: dict[tuple, set[str]] = {}
        self.counts: dict[tuple, dict[str, Counter]] = {}
        self.const_bad: set[str] = set()          # refs violating a constant dependent
        for rid, vals in rows.items():
            k = self.key(vals)
            if k is None:
                continue
            self.members.setdefault(k, set()).add(rid)
            cs = self.counts.setdefault(k, {d: Counter() for d in self.var_deps})
            for d in self.var_deps:
                v = vals.get(d)
                if v is not None:
                    cs[d][v] += 1
            for d in self.const_deps:
                if self._const_viol(vals.get(d), d):
                    self.const_bad.add(CellKey(rid, d).key)
        # inconsistent (group, dep) pairs plus constant-violating cells
        self.n_bad = len(self.const_bad) + sum(
            1 for k in self.counts for d in self.var_deps if len(self.counts[k][d]) > 1)

    def key(self, vals: dict) -> tuple | None:
        if any(vals.get(d) is None for d in self.det):
            return None
        if any(vals.get(c) != v for c, v in self.lhs.items()):
            return None
        return tuple(vals.get(d) for d in self.det)

    def _const_viol(self, v: Any, d: str) -> bool:
        return v is not None and v != self.rhs[d]

    def violating_refs(self) -> set[str]:
        out = set(self.const_bad)
        for k, cs in self.counts.items():
            for d in self.var_deps:
                if len(cs[d]) > 1:
                    out.update(CellKey(r, d).key for r in self.members[k])
        return out


class IncrementalVerifier:
    """Exact single-cell verification from indexes built once per sweep.

    Construct once per (artifact, registry, options); ``assess(cand)`` returns the
    same VerifiedRepair as ``Verifier().assess(cand, art, reg, options=...)`` or None
    when the candidate is outside what it can decide.
    """

    def __init__(self, art: DataArtifact, reg: ConstraintRegistry,
                 options: Options | None = None, *, lam: float = 1.0,
                 default_trust: float = 1.0):
        self.art, self.reg = art, reg
        self.options = options or {}
        self.lam, self.default_trust = lam, default_trust
        self.hard = [c.cid for c in reg.hard()]
        self.soft = [c.cid for c in reg.soft()]
        self.unsupported = sorted((set(self.hard) - SUPPORTED_HARD)
                                  | (set(self.soft) - SUPPORTED_SOFT))
        self.supported = not self.unsupported
        if not self.supported:
            return

        cells = list(art.iter_cells())
        self.required_total = sum(1 for c in cells if c.is_required)
        self.missing = {c.key.key for c in cells if c.is_required and c.value is None}

        self.fds: list[_FDIndex] = []
        self.rows: dict[str, dict] = {}
        specs = []
        if FD in self.hard:
            specs += [(FD, det, deps, None, None) for det, deps in
                      _params(reg, FD, self.options).options.get("fds", [])]
        if CFD in self.hard:
            specs += [(CFD, det, deps, lhs, rhs) for det, deps, lhs, rhs in
                      _params(reg, CFD, self.options).options.get("cfds", [])]
        if specs:
            for c in cells:
                self.rows.setdefault(c.row_id, {})[c.col] = c.value
            self.fds = [_FDIndex(cid, det, deps, self.rows, lhs, rhs)
                        for cid, det, deps, lhs, rhs in specs]

        # reference violation set from the full predicates
        self.before = hard_violation_set(art, reg, self.options)
        self._self_check()
        self.n_hard = len(self.hard)
        self._sat_before = {cid: self._satisfied_now(cid) for cid in self.hard}

    # ------------------------------------------------------------------ checks
    def _index_violations(self) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        if FLOOR in self.hard:
            out |= {(FLOOR, r) for r in self.missing}
        for f in self.fds:
            out |= {(f.cid, r) for r in f.violating_refs()}
        return out

    def _self_check(self) -> None:
        mine = self._index_violations()
        if mine != self.before:
            raise RuntimeError(
                "incremental index disagrees with the reference predicates "
                f"({len(mine ^ self.before)} differing violations); refusing to use it")

    def _satisfied_now(self, cid: str) -> bool:
        if cid == FLOOR:
            return not self.missing
        if cid in (FD, CFD):
            return all(f.n_bad == 0 for f in self.fds if f.cid == cid)
        raise KeyError(cid)

    # ------------------------------------------------------------------ assess
    def assess(self, cand: RepairCandidate) -> VerifiedRepair | None:
        if not self.supported or isinstance(cand.proposed_value, MetadataTriple):
            return None
        try:
            key = CellKey.parse(cand.target_ref)
        except ValueError:
            return None
        cell = self.art.get_cell(key)
        if cell is None:
            return None                      # creates a cell: exact path decides
        ref, old, new = key.key, cell.value, cand.proposed_value

        new_viols = False
        sat_after: dict[str, bool] = {}

        # --- completeness floor (H) and completeness (W) share the missing count
        miss_delta = 0
        if cell.is_required:
            miss_delta = (new is None) - (old is None)
        if FLOOR in self.hard:
            if cell.is_required and new is None and (FLOOR, ref) not in self.before:
                new_viols = True
            sat_after[FLOOR] = (len(self.missing) + miss_delta) == 0

        # --- functional and conditional functional dependencies (H)
        for cid in (FD, CFD):
            if cid in self.hard:
                sat_after[cid] = True
        if self.fds:
            fd_new, bad_after = self._fd_effect(key.row_id, key.col, old, new)
            new_viols = new_viols or fd_new
            for cid in (FD, CFD):
                if cid in self.hard:
                    sat_after[cid] = bad_after.get(cid, 0) == 0

        feasible = not new_viols
        margin = (sum(1 for cid in self.hard if sat_after[cid]) / self.n_hard
                  if self.n_hard else 1.0)

        gain = 0.0
        for cid in self.soft:                 # only dq.completeness is supported
            w = self.reg.requirements.weight(cid)
            if self.required_total == 0:
                q0 = q1 = 1.0
            else:
                q0 = 1.0 - len(self.missing) / self.required_total
                q1 = 1.0 - (len(self.missing) + miss_delta) / self.required_total
            gain += w * (q1 - q0)
        delta = gain - self.lam * edit_cost(cand, self.art, self.default_trust)

        return VerifiedRepair(
            target_ref=cand.target_ref, proposed_value=cand.proposed_value,
            rationale=cand.rationale, evidence=list(cand.evidence),
            s_conf=cand.s_conf, s_agree=cand.s_agree, s_consistency=cand.s_consistency,
            s_margin=margin, s_hat=cand.s_hat, feasible=feasible,
            delta_objective=delta)

    def _fd_effect(self, rid: str, col: str, old: Any,
                   new: Any) -> tuple[bool, dict[str, int]]:
        """(introduces a new violation?, violation units per cid after the repair)."""
        any_new = False
        bad_after: dict[str, int] = {}
        vals_before = self.rows.get(rid, {})
        vals_after = dict(vals_before)
        vals_after[col] = new
        for f in self.fds:
            if col not in f.cols or old == new:
                bad_after[f.cid] = bad_after.get(f.cid, 0) + f.n_bad
                continue
            k0, k1 = f.key(vals_before), f.key(vals_after)
            delta_bad = 0
            # constant dependents: only the repaired row's own cell can change
            for d in f.const_deps:
                vb = k0 is not None and f._const_viol(vals_before.get(d), d)
                va = k1 is not None and f._const_viol(vals_after.get(d), d)
                delta_bad += va - vb
                if va and not any_new and (f.cid, CellKey(rid, d).key) not in self.before:
                    any_new = True
            # variable dependents: the group r leaves and the group r joins
            for k in {k for k in (k0, k1) if k is not None}:
                cs = f.counts.get(k, {d: Counter() for d in f.var_deps})
                members_b = f.members.get(k, set())
                in_b, in_a = (k == k0), (k == k1)
                for d in f.var_deps:
                    rem = vals_before.get(d) if in_b else None
                    add = vals_after.get(d) if in_a else None
                    viol_b = len(cs[d]) > 1
                    viol_a = _distinct_after(cs[d], rem, add) > 1
                    delta_bad += viol_a - viol_b
                    if not viol_a or any_new:
                        continue
                    if in_a and (f.cid, CellKey(rid, d).key) not in self.before:
                        any_new = True
                        continue
                    if not viol_b:
                        # group turned inconsistent: every other member is a candidate
                        for m in members_b:
                            if m != rid and (f.cid, CellKey(m, d).key) not in self.before:
                                any_new = True
                                break
            bad_after[f.cid] = bad_after.get(f.cid, 0) + f.n_bad + delta_bad
        return any_new, bad_after


__all__ = ["IncrementalVerifier"]

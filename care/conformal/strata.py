"""care.conformal.strata -- stratification for group-conditional control.

A strata function maps a repair-like object (anything with ``.target_ref``, or a
bare ref string) to a stratum label; the controller calibrates one threshold per
stratum. Calibrating per stratum restores exchangeability under heterogeneity:
if a source, column, or domain is systematically harder, a single marginal
threshold can satisfy the pooled error while quietly violating that subgroup --
group-conditional (Mondrian) control gives a per-stratum guarantee instead.

Strata functions:
  * ``marginal_strata``   -- one global stratum (marginal control).
  * ``by_column``         -- stratum = the repaired column.
  * ``by_domain``         -- stratum = a semantic domain via a column->domain map.
  * ``by_source``         -- stratum = the source backing the repair (a ref->source
                             map, falling back to the repair's evidence).
  * ``by_cardinality``    -- stratum = a label-free column-cardinality bucket.
  * ``by_product``        -- stratum = the tuple of several stratifiers.

The cost of finer strata is thinner calibration data per group, so a finer
partition can certify less even when its scores are cleaner (see ``by_product``).
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from care.core.artifact import CellKey, DataArtifact

StrataFn = Callable[[Any], str]
DEFAULT_STRATUM = "default"


def _ref(r: Any) -> str:
    return r if isinstance(r, str) else r.target_ref


def _is_meta(ref: str) -> bool:
    return ref.startswith("meta:")


def marginal_strata(_r: Any) -> str:
    """Everything in one stratum -> a single global threshold (marginal control)."""
    return DEFAULT_STRATUM


def by_column(*, metadata_stratum: str = "col:__meta__") -> StrataFn:
    """Stratum = the repaired column (metadata refs share one stratum)."""

    def f(r: Any) -> str:
        ref = _ref(r)
        if _is_meta(ref):
            return metadata_stratum
        try:
            return f"col:{CellKey.parse(ref).col}"
        except ValueError:
            return DEFAULT_STRATUM

    return f


def by_domain(
    domain_of: Mapping[str, str] | Callable[[str], str | None],
    *,
    default: str = "other",
) -> StrataFn:
    """Stratum = a semantic domain, via a column->domain map or callable."""
    lookup = domain_of.get if hasattr(domain_of, "get") else domain_of

    def f(r: Any) -> str:
        ref = _ref(r)
        if _is_meta(ref):
            return "domain:__meta__"
        try:
            col = CellKey.parse(ref).col
        except ValueError:
            return f"domain:{default}"
        return f"domain:{lookup(col) or default}"

    return f


def by_source(
    source_of: Mapping[str, str] | Callable[[str], str | None],
    *,
    default: str = "unknown",
) -> StrataFn:
    """Stratum = the originating source of the repaired ref (poison-aware control).

    Falls back to the repair's own ``evidence`` (the source backing the repair)
    when the ref is not in the map, then to ``default``.
    """
    lookup = source_of.get if hasattr(source_of, "get") else source_of

    def f(r: Any) -> str:
        ref = _ref(r)
        sid = lookup(ref)
        if sid is None:
            evidence = getattr(r, "evidence", None)
            sid = evidence[0] if evidence else None
        return f"src:{sid or default}"

    return f


def cardinality_domains(
    art: DataArtifact, *, low: float = 0.05, high: float = 0.40
) -> dict[str, str]:
    """Map each column to a cardinality bucket from the DIRTY data alone (label-free
    -- never reads gold), so it is a legitimate a-priori stratifier.

    key = distinct non-null values / rows observed in the column:
      <= low   -> 'card:low'   (categoricals: State, HospitalType, EmergencyService)
      <= high  -> 'card:med'
      else     -> 'card:high'  (free-text / identifiers: Address, ProviderNumber)
    """
    from collections import defaultdict

    distinct: dict[str, set] = defaultdict(set)
    rows: dict[str, set] = defaultdict(set)
    for c in art.iter_cells():
        rows[c.col].add(c.row_id)
        if c.value is not None:
            distinct[c.col].add(str(c.value))

    out: dict[str, str] = {}
    for col, rset in rows.items():
        ratio = len(distinct[col]) / max(1, len(rset))
        out[col] = "card:low" if ratio <= low else "card:med" if ratio <= high else "card:high"
    return out


def by_product(*fns: StrataFn, sep: str = "&") -> StrataFn:
    """Stratum = the tuple of several stratifiers, e.g. column x source.

    Needed when one log mixes decision types whose scores live on DIFFERENT scales.
    A proposer that emits both "change this cell to X" and "keep the current value"
    scores the two differently -- the first rates the chosen repair, the second rates
    the repair it declined to make -- so a single column stratum containing both can
    exhibit Simpson's paradox: each source ranks correctly on its own while the pooled
    ranking inverts. Conformal control cannot fix that, because it calibrates ONE
    threshold per stratum and there is no threshold that is right for both scales.
    Splitting them into separate strata restores a coherent score inside each.

    The cost is real and must be reported: the number of strata multiplies, every
    stratum pays its own multiplicity penalty, and each holds less calibration data.
    Per the finite-sample rule (a zero-error stratum needs roughly
    ln(|Lambda|/delta)/alpha calibration cells to certify at all), a product
    stratification can certify LESS than the coarser one even though its scores are
    cleaner. Measure both; do not assume the finer partition wins.
    """
    def f(r: Any) -> str:
        return sep.join(fn(r) for fn in fns)

    return f


def by_cardinality(art: DataArtifact, *, low: float = 0.05, high: float = 0.40) -> StrataFn:
    """Mondrian strata by column-cardinality bucket (label-free; ~3 coarse groups, so
    each stratum keeps enough calibration mass to certify -- unlike per-column)."""
    return by_domain(cardinality_domains(art, low=low, high=high))


__all__ = [
    "StrataFn",
    "DEFAULT_STRATUM",
    "marginal_strata",
    "by_column",
    "by_domain",
    "by_source",
    "cardinality_domains",
    "by_cardinality",
    "by_product",
]

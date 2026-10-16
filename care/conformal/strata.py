"""Strata functions for group-conditional (Mondrian) control.

A strata function maps a repair (or a bare ref string) to a stratum label; the
controller calibrates one threshold per stratum. Finer strata give per-group
guarantees but leave less calibration data per group.
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
    """One global stratum (marginal control)."""
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
    """Stratum = the source of the repaired ref; falls back to the repair's first
    evidence source, then to ``default``."""
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
    """Map each column to a cardinality bucket computed from the dirty data only.

    ratio = distinct non-null values / rows:
      <= low   -> 'card:low'   (categorical)
      <= high  -> 'card:med'
      else     -> 'card:high'  (free text, identifiers)
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
    """Stratum = the joined labels of several stratifiers, e.g. column x source.

    Useful when one log mixes score scales (e.g. change vs keep decisions). Each
    product stratum has less calibration data, so it may certify less overall.
    """
    def f(r: Any) -> str:
        return sep.join(fn(r) for fn in fns)

    return f


def type_domains(art: DataArtifact, *, share: float = 0.9,
                 low: float = 0.05) -> dict[str, str]:
    """Map each column to a data type inferred from the dirty data only:

      'type:numeric'      >= ``share`` of non-null values parse as numbers
      'type:date'         >= ``share`` look like dates or times
      'type:categorical'  otherwise, if distinct values / rows <= ``low``
      'type:text'         otherwise (names, addresses, titles, identifiers)
    """
    import re
    from collections import defaultdict

    date = re.compile(r"^\d{1,4}[-/.]\d{1,2}([-/.]\d{1,4})?([ T].*)?$|"
                      r"^\d{1,2}:\d{2}(\s*[ap]\.?m\.?)?$|^[a-z]{3}-\d{1,2}$|^\d{1,2}-[a-z]{3}$",
                      re.I)
    vals: dict[str, list] = defaultdict(list)
    rows: dict[str, set] = defaultdict(set)
    for c in art.iter_cells():
        rows[c.col].add(c.row_id)
        if c.value is not None and str(c.value).strip():
            vals[c.col].append(str(c.value).strip())

    def numeric(v):
        try:
            float(v.replace(",", ""))
            return True
        except ValueError:
            return False

    out: dict[str, str] = {}
    for col, rset in rows.items():
        v = vals[col]
        if v and sum(map(numeric, v)) >= share * len(v):
            out[col] = "type:numeric"
        elif v and sum(bool(date.match(x)) for x in v) >= share * len(v):
            out[col] = "type:date"
        elif len(set(v)) / max(1, len(rset)) <= low:
            out[col] = "type:categorical"
        else:
            out[col] = "type:text"
    return out


def by_type(art: DataArtifact, **kw) -> StrataFn:
    """Mondrian strata by inferred column data type (see ``type_domains``)."""
    return by_domain(type_domains(art, **kw))


def by_cardinality(art: DataArtifact, *, low: float = 0.05, high: float = 0.40) -> StrataFn:
    """Mondrian strata by column-cardinality bucket (see ``cardinality_domains``)."""
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
    "type_domains",
    "by_type",
    "by_product",
]

"""Raha/Baran benchmark corpora: download and load dirty/clean CSV pairs.

Error cells are those where dirty != clean; their clean values form the gold ``Defects``.
"""

from __future__ import annotations

import csv
import os
import urllib.request
from dataclasses import dataclass, field

from care.core.artifact import Cell, DataArtifact, Source

RAW = "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets"
DATA_DIR = os.environ.get("CARE_DATA_DIR", "data")
NULL_TOKENS = {"", "empty", "nan", "null", "n/a", "na", "?"}


@dataclass
class Defects:
    """The gold error cells of a corpus and their clean values."""

    gold: dict[str, object] = field(default_factory=dict)   # "<row>::<col>" -> clean value

    @property
    def refs(self) -> set[str]:
        return set(self.gold)


@dataclass(frozen=True)
class DatasetSpec:
    name: str


# Base corpora; Ni et al. variants are added by ``_register_ni_variants``.
DATASETS: dict[str, DatasetSpec] = {
    "hospital": DatasetSpec("hospital"),
    "flights": DatasetSpec("flights"),
    "beers": DatasetSpec("beers"),
    "rayyan": DatasetSpec("rayyan"),
    "tax": DatasetSpec("tax"),
}


def _register_ni_variants() -> int:
    """Register Ni et al. noise variants listed in data/ni_variants.csv; returns the count.

    The manifest is written by tools/import_ni_variants.py for pairs that pass its checks.
    """
    path = os.path.join(DATA_DIR, "ni_variants.csv")
    if not os.path.exists(path):
        return 0
    n = 0
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                name = r.get("dataset")
                if not name or name in DATASETS:
                    continue
                DATASETS[name] = DatasetSpec(name)
                n += 1
    except (OSError, csv.Error):
        return n
    return n


_register_ni_variants()


def _norm(v: str | None) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return None if s.lower() in NULL_TOKENS else s


# Corpora available upstream; any other name must exist locally.
DOWNLOADABLE = ("hospital", "flights", "beers", "rayyan", "tax")


def download(name: str, *, data_dir: str = DATA_DIR) -> str:
    """Fetch dirty.csv/clean.csv for ``name`` into ``<data_dir>/<name>/``; returns the folder."""
    if name not in DOWNLOADABLE:
        raise SystemExit(
            f"dataset {name!r} has no local copy at {os.path.join(data_dir, name)} and is "
            f"not one of the downloadable upstream corpora {list(DOWNLOADABLE)}.\n"
            f"If it is an imported Ni et al. variant, run:\n"
            f"    python tools/import_ni_variants.py --adr /tmp/adr\n"
            f"If the name looks like a shell pattern, the glob that produced it matched "
            f"no files -- use `shopt -s nullglob`.")
    folder = os.path.join(data_dir, name)
    os.makedirs(folder, exist_ok=True)
    for fn in ("dirty.csv", "clean.csv"):
        dst = os.path.join(folder, fn)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(f"{RAW}/{name}/{fn}", dst)
    return folder


def _read(path: str) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def load(name: str, *, data_dir: str = DATA_DIR, source_trust: float = 0.7):
    """Return (dirty_artifact, defects, columns).

    ``defects.gold`` maps each error cell ``"<rowkey>::<col>"`` to its clean value.
    Columns align by position; the clean header gives the names.
    """
    folder = os.path.join(data_dir, name)
    if not (os.path.exists(os.path.join(folder, "dirty.csv"))):
        folder = download(name, data_dir=data_dir)
    _dh, drows = _read(os.path.join(folder, "dirty.csv"))
    chead, crows = _read(os.path.join(folder, "clean.csv"))

    # Column 0 is the row key only if it looks like an index (tax has none);
    # otherwise use the 0-based row position.
    first_name = chead[0].strip().lower()
    looks_named_index = first_name in {"index", "tid", "id", "row_id", "key", ""}
    first_vals = [r[0] for r in crows if r]
    looks_int_index = bool(first_vals) and all(v.strip().lstrip("-").isdigit() for v in first_vals)
    has_index_col = looks_named_index or looks_int_index

    if has_index_col:
        cols = chead[1:]
        col_start = 1
        def row_key(i, drow):
            return drow[0]
    else:
        cols = chead[:]                 # no key column -- keep every column
        col_start = 0
        def row_key(i, drow):
            return str(i)               # 0-based position, as in Baran's dataframe

    art = DataArtifact(artifact_id=name)
    art.add_source(Source(source_id="ingest", uri=folder, kind="lake", trust_prior=source_trust))
    defects = Defects()
    for i, (drow, crow) in enumerate(zip(drows, crows)):
        key = row_key(i, drow)
        for jj, col in enumerate(cols):
            j = jj + col_start
            dval = _norm(drow[j]) if j < len(drow) else None
            cval = _norm(crow[j]) if j < len(crow) else None
            ref = f"{key}::{col}"
            art.set_cell(Cell(row_id=key, col=col, value=dval, is_required=True))
            if dval != cval:                      # an error cell
                defects.gold[ref] = cval
    return art, defects, cols


__all__ = ["Defects", "DatasetSpec", "DATASETS", "download", "load"]
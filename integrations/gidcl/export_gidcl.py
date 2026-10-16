"""Export GIDCL's released cleaned tables as CARE logs (row_id, column, value, confidence, source).

Reads GEIL_Data/<dataset>/correction/result/correction.csv from the authors' release
and logs every cell that differs from CARE's dirty.csv (checked identical by md5).
Differences due only to null tokens or numeric formatting are not repairs. The
release has no per-cell score, so confidence is empty (see score_gidcl.py).

    python export_gidcl.py hospital beers flights rayyan tax --care ../CARE
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import os
import re
import sys

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(os.path.abspath(__file__))

# Copy of bench.datasets.NULL_TOKENS; the exporter runs without CARE on the path.
NULL_TOKENS = {"", "empty", "nan", "null", "n/a", "na", "?", "none"}
_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def _md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _read(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def _has_index_col(header, rows):
    first = header[0].strip().lower()
    if first in {"index", "tid", "id", "row_id", "key", ""}:
        return True
    vals = [r[0] for r in rows if r]
    return bool(vals) and all(v.strip().lstrip("-").isdigit() for v in vals)


def canon(v):
    """Null tokens to '', '12.0' -> '12', '01907' -> '1907'; used only to decide
    whether a cell changed."""
    s = (v or "").strip().lower()
    if s in NULL_TOKENS:
        return ""
    if _NUM.match(s):
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else repr(f)
        except ValueError:
            return s
    return s


def diff_cells(ds, gidcl_dir, care_dir):
    """Cells GIDCL changed, as (row_key, column, value, row_index), plus absorbed counts."""
    dirty_p = os.path.join(care_dir, "data", ds, "dirty.csv")
    clean_p = os.path.join(care_dir, "data", ds, "clean.csv")
    g_dirty = os.path.join(gidcl_dir, ds, "original", "dirty.csv")
    if not os.path.exists(dirty_p):
        raise SystemExit(f"{ds}: {dirty_p} not found (run CARE once to download the tables)")
    if os.path.exists(g_dirty) and _md5(g_dirty) != _md5(dirty_p):
        raise SystemExit(f"{ds}: GIDCL's dirty.csv differs from CARE's; refusing to diff")
    dh, drows = _read(dirty_p)
    ch, _ = _read(clean_p)
    gh, grows = _read(os.path.join(gidcl_dir, ds, "correction", "result", "correction.csv"))

    if gh and gh[0] == "":                      # pandas' unnamed index column
        gh, grows = gh[1:], [r[1:] for r in grows]
    if len(grows) != len(drows):
        raise SystemExit(f"{ds}: {len(grows)} cleaned rows vs {len(drows)} dirty rows")
    extra = gh[len(dh):]                        # e.g. flights' 'count'

    has_idx = _has_index_col(dh, drows)
    col_start = 1 if has_idx else 0
    cols = ch[col_start:]                       # CARE names columns after the clean header

    absorbed = collections.Counter()
    cells = []
    for i, (dr, gr) in enumerate(zip(drows, grows)):
        key = dr[0] if has_idx else str(i)
        for jj, col in enumerate(cols):
            j = jj + col_start
            dv = dr[j] if j < len(dr) else ""
            gv = gr[j] if j < len(gr) else ""
            if dv.strip().lower() == gv.strip().lower():
                continue
            if canon(dv) == canon(gv):
                absorbed["null" if canon(dv) == "" else "numeric-format"] += 1
                continue
            cells.append((key, col, gv, i))
    return cells, absorbed, extra


def export(ds, gidcl_dir, care_dir, out_dir):
    cells, absorbed, extra = diff_cells(ds, gidcl_dir, care_dir)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"gidcl_{ds}_pred.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "column", "value", "confidence", "source"])
        w.writerows((k, c, v, "", "gidcl") for k, c, v, _ in cells)
    print(f"{ds:8s} {len(cells):6d} repairs  absorbed as serialisation={dict(absorbed)}  "
          f"extra_cols={extra}  -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--gidcl", default=os.path.join(HERE, "GEIL_Data"),
                    help="GIDCL's GEIL_Data folder (default: ./GEIL_Data)")
    ap.add_argument("--care", default=os.path.join(HERE, os.pardir, "CARE"),
                    help="CARE repository root (default: ../CARE)")
    ap.add_argument("--out", default=None, help="default: <care>/csvs")
    a = ap.parse_args()
    for ds in a.datasets:
        export(ds, a.gidcl, a.care, a.out or os.path.join(a.care, "csvs"))


if __name__ == "__main__":
    main()

"""Export ZeroEC runs (run_zeroec.py) as CARE logs (row_id, column, value, confidence, source).

ZeroEC's tables match CARE's row for row, so rows are matched by position. A cell is a
repair if corrections.csv differs from dirty.csv after null and numeric canonicalisation;
cells in the labelled rows are dropped. source is zeroec_llm for cells sent to the LLM
correction prompt, else zeroec_rule. With several seeds, confidence in the primary log is
the fraction of seeds that agree with the primary value; per-seed logs have none.

    python export_zeroec.py hospital beers flights rayyan --care ../CARE

Output: <care>/csvs/zeroec_<ds>_pred.csv and zeroec_s<k>_<ds>_pred.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(os.path.abspath(__file__))
NULL_TOKENS = {"", "empty", "nan", "null", "n/a", "na", "?", "none"}
_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def canon(v):
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


def _read(p):
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def _has_index_col(header, rows):
    first = header[0].strip().lower()
    if first in {"index", "tid", "id", "row_id", "key", "tuple_id", ""}:
        vals = [r[0] for r in rows[:200] if r]
        return all(v.strip().lstrip("-").isdigit() for v in vals)
    return False


def seed_cells(ds, run_dir, care_dir):
    """{(key, column): value} for one seed run, plus counts of what was dropped."""
    dh, drows = _read(os.path.join(care_dir, "data", ds, "dirty.csv"))
    ch, _ = _read(os.path.join(care_dir, "data", ds, "clean.csv"))
    zh, zrows = _read(os.path.join(run_dir, "corrections.csv"))
    if len(zrows) != len(drows):
        raise SystemExit(f"{ds}: {len(zrows)} corrected rows vs {len(drows)} in CARE's table")
    zstart = 1 if zh and zh[0].strip().lower() == "index" else 0
    has_idx = _has_index_col(dh, drows)
    cstart = 1 if has_idx else 0
    cols = ch[cstart:]

    def norm(h):
        return h.strip().lower().replace("-", "_")
    # align columns by name; a CARE row-key column (rayyan's `id`) is skipped
    zpos = {norm(h): k for k, h in enumerate(zh) if k >= zstart}
    missing = [c for c in cols if norm(c) not in zpos]
    if missing:
        raise SystemExit(f"{ds}: CARE columns {missing} not found in ZeroEC's output")
    pairs = [(c, zh[zpos[norm(c)]], zpos[norm(c)], jj + cstart) for jj, c in enumerate(cols)]
    skipped = [h for k, h in enumerate(zh) if k >= zstart and norm(h) not in {norm(c) for c in cols}]
    labelled = {int(x) for x in open(os.path.join(run_dir, "labelled_rows.txt")).read().split()}
    llm = set()
    with open(os.path.join(run_dir, "llm_cells.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            llm.add((int(r["row_idx"]), r["column"]))
    out, dropped = {}, collections.Counter()
    for i, (dr, zr) in enumerate(zip(drows, zrows)):
        key = dr[0] if has_idx else str(i)
        for col, zcol, zj, dj in pairs:
            dv = dr[dj] if dj < len(dr) else ""
            zv = zr[zj] if zj < len(zr) else ""
            if dv.strip().lower() == zv.strip().lower() or canon(dv) == canon(zv):
                continue
            if i in labelled:
                dropped["labelled_row"] += 1
                continue
            src = "zeroec_llm" if (i, zcol) in llm else "zeroec_rule"
            out[(key, col)] = (zv, src)
    if skipped:
        dropped["skipped_columns:" + ",".join(skipped)] = 0
    return out, dropped


def export(ds, runs, care_dir, out_dir, primary, tag="zeroec"):
    seeds = sorted(glob.glob(os.path.join(runs, ds, "seed*")),
                   key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    seeds = [s for s in seeds if os.path.exists(os.path.join(s, "corrections.csv"))]
    if not seeds:
        print(f"{ds}: no finished runs under {runs}/{ds}/seed*")
        return
    per = {}
    for s in seeds:
        k = int(re.sub(r"\D", "", os.path.basename(s)))
        per[k], dropped = seed_cells(ds, s, care_dir)
        n_llm = sum(1 for _, src in per[k].values() if src == "zeroec_llm")
        p = os.path.join(out_dir, f"{tag}_s{k}_{ds}_pred.csv")
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row_id", "column", "value", "confidence", "source"])
            w.writerows((key, col, v, "", src) for (key, col), (v, src) in per[k].items())
        print(f"{ds:8s} seed {k}: {len(per[k]):6d} repairs ({n_llm} via the LLM prompt, "
              f"{len(per[k]) - n_llm} via generated rules); dropped {dict(dropped)} -> {p}")
    if primary not in per:
        primary = min(per)
    base = per[primary]
    many = len(per) > 1
    p = os.path.join(out_dir, f"{tag}_{ds}_pred.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "column", "value", "confidence", "source"])
        for (key, col), (v, src) in base.items():
            if many:
                agree = sum(1 for cells in per.values()
                            if canon(cells.get((key, col), ("",))[0]) == canon(v))
                conf = f"{agree / len(per):.4f}"
            else:
                conf = ""
            w.writerow([key, col, v, conf, src])
    print(f"{ds:8s} primary seed {primary}, {len(per)} seed(s): {len(base)} repairs, "
          f"confidence = {'seed agreement' if many else 'none (unranked)'} -> {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--care", default=os.path.join(HERE, os.pardir, "CARE"))
    ap.add_argument("--out", default=None, help="default: <care>/csvs")
    ap.add_argument("--primary", type=int, default=0)
    ap.add_argument("--tag", default="zeroec",
                    help="file prefix, e.g. zeroec_gen4o for a variant (CARE accepts <tag>_<variant>)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.care, "csvs")
    os.makedirs(out, exist_ok=True)
    for ds in a.datasets:
        export(ds, a.runs, a.care, out, a.primary, a.tag)


if __name__ == "__main__":
    main()

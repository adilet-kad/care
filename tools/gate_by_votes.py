"""Derive a stricter abstention gate from a verify-correct log without a GPU pass.

`export_jellyfish.py --mode verify-correct --verify-threshold 1` corrects every cell that
any of the n verification samples flagged and writes the vote count per cell to a
sidecar `<tag>_<dataset>_votes.csv`. This tool blanks the value of every row whose
yes-votes fall below `--min-yes`, turning those cells into logged abstentions, and
writes a new log under a new tag, so gates at 1, 2, 3, 4 and 5 votes come from one run.

    python tools/gate_by_votes.py hospital --tag jellyfish_vc --min-yes 3 --out-tag jellyfish_vc3
    python prepare_log.py hospital csvs/jellyfish_vc3_hospital_pred.csv --tool jellyfish_vc3
"""
from __future__ import annotations

import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--tag", default="jellyfish_vc")
    ap.add_argument("--min-yes", type=int, required=True)
    ap.add_argument("--out-tag", required=True)
    ap.add_argument("--dir", default="csvs")
    a = ap.parse_args()
    log = os.path.join(a.dir, f"{a.tag}_{a.dataset}_pred.csv")
    votes_path = os.path.join(a.dir, f"{a.tag}_{a.dataset}_votes.csv")
    out = os.path.join(a.dir, f"{a.out_tag}_{a.dataset}_pred.csv")
    votes = {}
    with open(votes_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            votes[(r["row_id"], r["column"])] = int(r["yes"])
    kept = blanked = 0
    with open(log, newline="", encoding="utf-8", errors="replace") as fh, \
            open(out, "w", newline="", encoding="utf-8") as oh:
        rd = csv.DictReader(fh)
        w = csv.DictWriter(oh, fieldnames=rd.fieldnames)
        w.writeheader()
        for r in rd:
            if (r.get("value") or "").strip() and votes.get((r["row_id"], r["column"]), 0) < a.min_yes:
                r["value"] = ""
                r["confidence"] = ""
                blanked += 1
            elif (r.get("value") or "").strip():
                kept += 1
            if r.get("source"):
                r["source"] = a.out_tag
            w.writerow(r)
    print(f"{out}: {kept} proposals kept, {blanked} turned into abstentions (min yes-votes {a.min_yes})")


if __name__ == "__main__":
    main()

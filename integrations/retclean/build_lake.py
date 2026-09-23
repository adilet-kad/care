"""build_lake.py -- build a ZERO-LEAKAGE RetClean data lake from a dirty dataset's
own redundancy, and upload it through RetClean's /index/ endpoint.

Why this exists
---------------
RetClean's contribution is *retrieval-grounded* repair: it answers a missing/dirty
cell by retrieving a related tuple from a data lake and citing it. Running RetClean
with no index (standalone) disables the entire mechanism and benchmarks "an 8B model
guessing blind" -- which is not RetClean and is not a fair comparison.

But the lake must not contain gold. Indexing `clean.csv` would make retrieval return
the ground-truth answer directly: accuracy ~100%, and the result is oracle leakage
rather than a measurement of retrieval.

So this script builds the lake the honest way, from the DIRTY data only:

    group the dirty rows by a key column (e.g. provider_number), and for every other
    attribute take the MAJORITY non-null value within the group.

Errors in these corpora are sparse, independent, minority corruptions, so the
majority vote over a group recovers the consensus value without ever reading
clean/gold. The lake is derived only from the data under repair and never reads
clean/gold values, so it introduces no oracle leakage.

The resulting lake is a legitimate stand-in for an enterprise reference table
bootstrapped from a lake's own redundancy -- and it can be corrupted for the
untrusted-channel experiment, which is impossible in standalone mode.

Usage (from the RetClean repo root, stack running):
    # inspect what would be built, without uploading
    python build_lake.py hospital --key provider_number --dry-run

    # build + upload one lake table keyed on provider_number
    python build_lake.py hospital --key provider_number

    # several tables, one per key (entity-level and measure-level redundancy)
    python build_lake.py hospital --key provider_number --key measure_code

Then export WITH retrieval:
    python export_retclean.py hospital --reasoner "Llama 3.1" \
        --index hospital_lake --index-type semantic
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_URL = os.environ.get("RETCLEAN_BACKEND", "http://localhost:8000")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _has_index_col(header, rows):
    """Mirror bench.datasets.load's index detection EXACTLY.

    A wrong row-key convention silently misaligns every gold comparison."""
    first = header[0].strip().lower()
    if first in {"index", "tid", "id", "row_id", "key", ""}:
        return True
    vals = [r[0] for r in rows if r]
    return bool(vals) and all(v.strip().lstrip("-").isdigit() for v in vals)


def build_majority_table(header, rows, key_col, *, col_start, min_group=2):
    """Group rows by `key_col`; per group emit the majority non-null value of every
    other attribute. Returns (out_header, out_rows, stats).

    Only groups with >= min_group members are emitted: a singleton group has no
    redundancy, so its "majority" is just the (possibly dirty) original row -- that
    would inject errors into the lake rather than consensus.
    """
    key_idx = header.index(key_col)
    other_idx = [j for j in range(col_start, len(header)) if j != key_idx]

    groups = collections.defaultdict(list)
    for r in rows:
        if key_idx >= len(r):
            continue
        k = (r[key_idx] or "").strip()
        if not k:
            continue
        groups[k].append(r)

    out_header = [key_col] + [header[j] for j in other_idx]
    out_rows = []
    n_singleton = 0
    for k, members in sorted(groups.items()):
        if len(members) < min_group:
            n_singleton += 1
            continue
        out = [k]
        for j in other_idx:
            votes = collections.Counter()
            for r in members:
                if j < len(r):
                    v = (r[j] or "").strip()
                    if v and v.lower() not in ("null", "none", "nan", "n/a", "empty"):
                        votes[v] += 1
            out.append(votes.most_common(1)[0][0] if votes else "")
        out_rows.append(out)

    stats = {
        "groups": len(groups),
        "emitted": len(out_rows),
        "skipped_singletons": n_singleton,
        "avg_group_size": (sum(len(m) for m in groups.values()) / len(groups)) if groups else 0.0,
    }
    return out_header, out_rows, stats


def upload_index(index_name, tables, *, recreate):
    """POST the built tables to RetClean's /index/ endpoint (create + populate)."""
    if recreate:
        resp = requests.delete(f"{BACKEND_URL}/index/{index_name}", timeout=120)
        print(f"  delete existing index '{index_name}': {resp.status_code} {resp.text[:200]}")

    files = []
    for table_name, (hdr, rws) in tables.items():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(hdr)
        w.writerows(rws)
        files.append(("files", (f"{table_name}.csv", buf.getvalue(), "text/csv")))

    resp = requests.post(f"{BACKEND_URL}/index/", data={"index_name": index_name},
                         files=files, timeout=3600)
    if resp.status_code != 200:
        raise SystemExit(f"index upload failed: {resp.status_code} {resp.text[:500]}")
    print(f"  uploaded -> {resp.status_code} {resp.text[:200]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--key", action="append", required=True,
                    help="key column to group on; repeat for multiple lake tables "
                         "(e.g. --key provider_number --key measure_code)")
    ap.add_argument("--index", default=None,
                    help="index name (default: <dataset>_lake)")
    ap.add_argument("--min-group", type=int, default=2,
                    help="minimum rows per group to emit (default 2; singletons have "
                         "no redundancy so their 'majority' is just the dirty row)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and report, write CSVs locally, do not upload")
    ap.add_argument("--recreate", action="store_true",
                    help="delete an existing index of the same name first")
    ap.add_argument("--save-dir", default=None,
                    help="also write the built lake tables here (default: lake/<dataset>)")
    args = ap.parse_args()

    index_name = args.index or f"{args.dataset}_lake"
    save_dir = args.save_dir or os.path.join(HERE, "lake", args.dataset)

    dirty_path = os.path.join(HERE, "datasets", args.dataset, "dirty.csv")
    if not os.path.exists(dirty_path):
        raise SystemExit(f"missing {dirty_path}")
    header, rows = _read_csv(dirty_path)
    col_start = 1 if _has_index_col(header, rows) else 0

    print(f"\n========== BUILD LAKE: {args.dataset} ==========")
    print(f"source: {dirty_path} (DIRTY only -- no gold is ever read)")
    print(f"rows: {len(rows)}   index name: {index_name}")

    tables = {}
    os.makedirs(save_dir, exist_ok=True)
    for key_col in args.key:
        if key_col not in header:
            print(f"  !! key column {key_col!r} not in header {header}; skipping",
                  file=sys.stderr)
            continue
        hdr, rws, stats = build_majority_table(header, rows, key_col,
                                               col_start=col_start,
                                               min_group=args.min_group)
        table_name = f"{args.dataset}_by_{key_col}"
        tables[table_name] = (hdr, rws)
        print(f"  [{table_name}] groups={stats['groups']} emitted={stats['emitted']} "
              f"skipped_singletons={stats['skipped_singletons']} "
              f"avg_group_size={stats['avg_group_size']:.1f}")
        out_path = os.path.join(save_dir, f"{table_name}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            w.writerows(rws)
        print(f"      wrote {out_path}")

    if not tables:
        raise SystemExit("no lake tables built (no valid --key columns)")

    if args.dry_run:
        print("\n  --dry-run: built locally, NOT uploaded.")
    else:
        print(f"\n  uploading {len(tables)} table(s) to index '{index_name}' ...")
        upload_index(index_name, tables, recreate=args.recreate)
        print("\nNext:")
        print(f"  python export_retclean.py {args.dataset} --reasoner \"Llama 3.1\" "
              f"--index {index_name} --index-type semantic")
    print("=" * (24 + len(args.dataset)) + "\n")


if __name__ == "__main__":
    main()

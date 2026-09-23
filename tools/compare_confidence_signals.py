"""Compare two confidence signals from the same proposer on the same cells.

Section 8.3 claims that the obvious confidence signal for an LLM cleaner -- mean
token-level log-probability -- is *anti-correlated* with correctness, and that
replacing it with self-consistency over k samples reverses the ordering. That is
the justification for `--confidence agreement` everywhere else in the paper, so it
needs to be checkable rather than asserted.

Both logs must come from the same model on the same dataset, differing only in how
confidence was computed. The exporter writes the same filename in both modes, so
the log-probability run must go to its own directory (csvs/logprob/) -- if the
two paths resolve to one file this script will say so rather than compare a log
with itself.

    python tools/compare_confidence_signals.py hospital \\
        --agreement csvs/jellyfish_hospital_mapped.csv \\
        --logprob   csvs/logprob/jellyfish_hospital_pred.csv

Reports, for each signal: mean confidence on correct vs incorrect proposals, and
accuracy in the top decile against the pool. A signal that ranks has a positive
separation and a top decile above the pool; an inverted one has neither.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _n(v):
    return str(v).strip().lower() if v is not None else ""


def score(path, gold, cur):
    """-> (correct?, confidence) per proposal, judged against gold for error cells
    and against the current value for clean ones (preserving a clean cell is
    correct; rewriting it is not)."""
    out = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            ref = f"{r['row_id']}::{r['column']}"
            v = _n(r.get("value"))
            try:
                c = float(r["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if ref in gold:
                out.append((v == gold[ref], c))
            elif ref in cur:
                out.append((v == cur[ref], c))
    return out


def report(label, rows, path=""):
    ok = [c for good, c in rows if good]
    bad = [c for good, c in rows if not good]
    if not rows:
        # Almost always an unmapped log: the exporter writes the proposer's own column
        # names, so every ref misses CARE's schema and nothing scores. Silent, because
        # "no cell matched" and "no cell was wrong" look identical downstream.
        print(f"  {label:<12} NO CELL MATCHED CARE's schema -- this log is probably "
              f"unmapped.\n  {'':<12} Run:  python prepare_log.py <dataset> {path} "
              f"--tool jellyfish \\\n  {'':<12}         --out {os.path.dirname(path) or '.'}\n"
              f"  {'':<12} then point --{label} at the _mapped.csv it writes.")
        return
    if not ok or not bad:
        print(f"  {label:<12} degenerate: {len(ok)} correct, {len(bad)} incorrect")
        return
    rows = sorted(rows, key=lambda t: -t[1])
    k = max(1, len(rows) // 10)
    top = sum(1 for good, _ in rows[:k] if good) / k
    pool = sum(1 for good, _ in rows if good) / len(rows)
    sep = st.mean(ok) - st.mean(bad)
    verdict = "RANKS" if sep > 0 and top > pool else "INVERTED"
    print(f"  {label:<12} conf(correct)={st.mean(ok):.3f}  conf(incorrect)={st.mean(bad):.3f}  "
          f"separation={sep:+.3f}")
    print(f"  {'':<12} top decile acc={top:.3f}  pool acc={pool:.3f}  -> {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--agreement", required=True)
    ap.add_argument("--logprob", required=True)
    args = ap.parse_args()

    if os.path.abspath(args.agreement) == os.path.abspath(args.logprob):
        raise SystemExit(
            "the two logs are the same file -- the exporter writes the same name in both\n"
            "confidence modes, so the log-probability run must use --out on its own\n"
            "directory: csvs/logprob/.")
    for p in (args.agreement, args.logprob):
        if not os.path.exists(p):
            raise SystemExit(f"log not found: {p}")

    from bench.datasets import load
    art, defects, _ = load(args.dataset)
    gold = {k: _n(v) for k, v in defects.gold.items()}
    cur = {f"{c.row_id}::{c.col}": _n(c.value) for c in art.iter_cells()}

    print(f"===== {args.dataset} =====")
    report("agreement", score(args.agreement, gold, cur), args.agreement)
    report("logprob", score(args.logprob, gold, cur), args.logprob)
    print("\nSection 8.3 expects log-probability to be INVERTED and agreement to RANK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

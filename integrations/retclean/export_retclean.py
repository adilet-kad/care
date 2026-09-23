"""export_retclean.py -- run RetClean over a dataset and export a CARE-ready
offline-repair log (the retrieval-augmented proposer of Table 1).

Same zero-coupling pattern as the Baran and Raha integrations: RetClean runs
natively here (its Docker stack: FastAPI backend + Ollama + Elasticsearch/Qdrant),
and we export a plain CSV log that CARE reads through the offline-log proposer
(`bench.baran_proposer`, via the `retclean:<log>` backend spec). CARE never imports
RetClean's code.

RetClean returns, per cell, {"value", "table_name", "row_number", ...}. The
``table_name`` is RetClean's *citation*: non-null when the value was drawn from a
retrieved data-lake tuple, null when the model produced it with no grounding. We
map that provenance to the log's ``source`` column:
    table_name present -> source = "retclean_lake"   (evidence-backed / grounded)
    table_name absent   -> source = "retclean_model"  (ungrounded model guess)
so RetClean's own provenance flows straight into CARE's trust gate. (For the
untrusted-channel corruption experiment, mark "retclean_lake" low-trust and CARE's
gate escalates every lake-cited repair.)

Prereqs: RetClean's stack must be up (`docker-compose up`) with the backend on
:8000, and -- if you want real retrieval rather than standalone LLM guessing -- an
index built via the /index/ endpoint (see tester.ipynb). Standalone mode
(no --index) is the simplest faithful "governs an LLM cleaner" run; every repair is
then "retclean_model".

Usage (run from the RetClean repo root, stack running):
    python export_retclean.py hospital --reasoner "Llama 3.1"
    python export_retclean.py beers --reasoner "Llama 3.1" --index beers_lake --index-type semantic
    python export_retclean.py hospital --columns City State --reasoner "GPT-4-OpenAI"

Output:
    ../CARE/csvs/retclean_<dataset>_pred.csv   (row_id,column,value,source)
Then in CARE:  python prepare_log.py <dataset> csvs/retclean_<dataset>_pred.csv --tool retclean
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, os.pardir, "CARE", "csvs")
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


def repair_column(dataset, target_col, header, rows, col_idx, other_idx, args):
    """Call RetClean's /repair/ endpoint for one target column; return
    list[(row_id, value, source)] aligned to `rows`.

    Sent in small batches (default 25 rows/request), NOT all rows in one POST.
    The backend loops synchronously over its `target_data` (one blocking Ollama
    call per row, in a single asyncio worker), so a 1000-row single request
    blocks the event loop for the full sequential runtime and appears to trip
    an intermediate idle/connection-reset limit (Docker's userland-proxy or
    similar) well before the client's own --timeout is reached -- symptom:
    ConnectionResetError with zero server-side traceback. Small batches give
    the event loop natural request boundaries, bound the worst-case single
    request duration, and show progress instead of one opaque multi-minute call.
    """
    def row_id(i):
        return rows[i][0] if args._has_index else str(i)

    pivot_names = [header[j] for j in other_idx]
    n = len(rows)
    bs = max(1, args.batch_size)
    out = []
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch_idx = list(range(start, end))
        records_target = [{"id": i, "value": (rows[i][col_idx] if col_idx < len(rows[i]) else None) or None}
                          for i in batch_idx]
        pivot_data = [{"id": i, "values": [(rows[i][j] if j < len(rows[i]) else "") for j in other_idx]}
                      for i in batch_idx]
        payload = {
            "entity_description": f"A row from the {dataset} table.",
            "target_name": target_col,
            "target_data": records_target,
            "pivot_names": pivot_names,
            "pivot_data": pivot_data,
            "reasoner_name": args.reasoner,
            "index_name": args.index,            # None -> standalone LLM (no retrieval)
            "index_type": args.index_type,
            "reranker_type": args.reranker,
        }
        resp = requests.post(f"{BACKEND_URL}/repair/", json=payload, timeout=args.timeout)
        if resp.status_code != 200:
            raise SystemExit(f"RetClean /repair failed for column {target_col!r} "
                             f"rows [{start},{end}): {resp.status_code} {resp.text[:300]}")
        results = resp.json().get("results", [])
        if len(results) != len(batch_idx):
            print(f"  !! column {target_col} rows [{start},{end}): got {len(results)} results "
                  f"for {len(batch_idx)} rows (alignment risk) -- truncating to min", file=sys.stderr)
        for offset, res in enumerate(results[:len(batch_idx)]):
            i = batch_idx[offset]
            val = (res.get("value") if isinstance(res, dict) else None)
            if val in (None, "", "None"):
                continue                              # RetClean abstained
            # `table_name` is only a meaningful citation signal when retrieval
            # actually ran (index_name set). In STANDALONE mode (index_name is
            # None) no lake lookup happened at all -- retrieved_list is empty
            # server-side -- so a non-blank table_name here is the model not
            # following its own "leave this blank with no context" instruction,
            # not a real grounded citation. Force honest provenance: everything
            # is "retclean_model" when there was no retrieval to cite from.
            cited = (args.index is not None and isinstance(res, dict)
                    and res.get("table_name") not in (None, "", "null"))
            # Per-cell confidence from RetClean's OWN provenance signal. Without a
            # confidence channel every repair gets a constant s_hat, so CARE's
            # conformal controller cannot rank cells within a stratum and degenerates
            # to all-or-nothing per stratum. Citation status is the honest, zero-extra-
            # cost signal RetClean actually produces: a repair grounded in a retrieved
            # lake tuple is materially more reliable than an ungrounded model guess.
            # (Values are a fixed two-level prior, not tuned per dataset; the conformal
            # layer calibrates what they mean, so only their ORDER matters.)
            conf = args.conf_lake if cited else args.conf_model
            out.append((row_id(i), str(val), "retclean_lake" if cited else "retclean_model",
                        f"{conf:.3f}"))
        print(f"  ... {target_col}: {end}/{n} rows ({len(out)} non-abstained so far)",
              end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return out


def export_one(dataset, args):
    ds_dir = os.path.join(HERE, "datasets", dataset)
    dirty_path = os.path.join(ds_dir, "dirty.csv")
    if not os.path.exists(dirty_path):
        raise SystemExit(f"missing {dirty_path}")
    header, rows = _read_csv(dirty_path)
    args._has_index = _has_index_col(header, rows)
    col_start = 1 if args._has_index else 0
    data_cols = header[col_start:]

    targets = args.columns or data_cols
    print(f"\n========== RETCLEAN: {dataset} ==========")
    print(f"rows: {len(rows)}  has_index_col: {args._has_index}  reasoner: {args.reasoner}  "
          f"index: {args.index or 'STANDALONE (no retrieval)'}  columns: {targets}")

    log_rows = []
    for target_col in targets:
        if target_col not in header:
            print(f"  !! column {target_col!r} not in header {header}; skipping", file=sys.stderr)
            continue
        col_idx = header.index(target_col)
        other_idx = [j for j in range(col_start, len(header)) if j != col_idx]
        cells = repair_column(dataset, target_col, header, rows, col_idx, other_idx, args)
        for rid, val, src, conf in cells:
            log_rows.append((rid, target_col, val, src, conf))
        print(f"  repaired {target_col}: {len(cells)} non-abstained cells")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"retclean_{dataset}_pred.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "column", "value", "source", "confidence"])
        w.writerows(log_rows)
    n_lake = sum(1 for r in log_rows if r[3] == "retclean_lake")
    print(f"WROTE: {out_path}  ({len(log_rows)} repairs; {n_lake} lake-cited, "
          f"{len(log_rows) - n_lake} model-guessed)")
    print(f"Next (in CARE):  python prepare_log.py {dataset} csvs/retclean_{dataset}_pred.csv --tool retclean")
    print("=" * (22 + len(dataset)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--reasoner", default="Llama 3.1",
                    help="one of the backend's initialized models: 'Llama 3.1', "
                         "'GPT-4-OpenAI', 'GPT-4-AzureOpenAI'")
    ap.add_argument("--columns", nargs="+", default=None,
                    help="target columns to repair (default: all data columns)")
    ap.add_argument("--index", default=None, help="RetClean index name for retrieval (omit = standalone LLM)")
    ap.add_argument("--index-type", default="semantic", choices=["semantic", "syntactic", "both"])
    ap.add_argument("--reranker", default=None)
    ap.add_argument("--conf-lake", type=float, default=0.80,
                    help="confidence written for lake-CITED (grounded) repairs")
    ap.add_argument("--conf-model", type=float, default=0.30,
                    help="confidence written for ungrounded model guesses; only the "
                         "ORDER vs --conf-lake matters, the conformal layer calibrates "
                         "what the values actually mean")
    ap.add_argument("--batch-size", type=int, default=25,
                    help="rows per /repair/ request (default 25; the backend "
                         "calls the reasoner once per row synchronously, so a "
                         "single request covering all rows tends to trip an "
                         "intermediate connection-reset limit before finishing)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    for ds in args.datasets:
        export_one(ds, args)


if __name__ == "__main__":
    main()

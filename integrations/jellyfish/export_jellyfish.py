"""export_jellyfish.py -- run Jellyfish over a dataset and export a CARE-ready
offline-repair log ("CARE governs a local instruction-tuned LLM cleaner").

Zero coupling, same as the Baran / Raha / RetClean integrations: Jellyfish runs
natively here, and we export a plain CSV log that CARE reads through its offline-log
proposer (`jellyfish:<log>` backend). CARE never imports this code.

WHY JELLYFISH (vs RetClean)
---------------------------
RetClean's native task is retrieval-based *imputation from a data lake*; on the
Raha/Baran corruption-correction benchmarks it under-performs and CARE certifies
nothing. Jellyfish is a local LLM instruction-tuned for data preprocessing, free to
run, and -- crucially -- exposes **token logprobs**, giving a GRADED confidence
signal. Graded confidence is what lets the conformal controller RANK repairs within
a stratum; RetClean's binary lake/model flag degenerated to all-or-nothing. This is the best shot at a certifiable LLM automation number.

TASK MAPPING (deliberate)
-------------------------
Jellyfish's trained tasks are error detection, DATA IMPUTATION, schema matching,
entity matching -- there is **no error-correction task**. We repair a cell by posing
it as imputation: withhold the target attribute, ask the model to infer it from the
rest of the record. Faithful use of the model as designed.

CONTAMINATION NOTE
------------------
Jellyfish was instruction-tuned on RAHA-derived data. Per the model card, **Hospital
is a "seen" dataset** for error detection; Flights/Rayyan are "unseen" but from the
same collection. Imputation's seen sets are Buy/Restaurant (not ours), so cell-repair
here isn't directly a trained task -- but the model has plausibly seen these records.
Treat hospital as contamination-suspect.

PERFORMANCE NOTES — tuned for NVIDIA DGX Spark (GB10 Blackwell, 128 GB unified,
273 GB/s)
------------------------------------------------------------------------------
273 GB/s is *low* bandwidth, so token DECODE is bandwidth-bound: a 13B in bf16
(~26 GB of weights) caps near ~10 tok/s for a single stream. Batching amortizes the
weight reads across many sequences, so aggregate throughput scales nearly linearly
with concurrency until it becomes compute-bound. Two consequences drive this script:

  1. **Default engine is vLLM OFFLINE** (`--engine offline`): we build EVERY prompt
     (all columns x all rows) up front and hand the whole list to `llm.generate()`
     in ONE call. vLLM's continuous batching then schedules optimally -- this is
     strictly better than issuing HTTP requests from a thread pool (no per-request
     overhead, no artificial concurrency ceiling, better KV-cache packing).
     Don't hand-pick a batch size; the scheduler beats a fixed number.
  2. **Our workload is prefill-heavy** (~350-token prompts, ~10-token answers), and
     prefill is compute-bound rather than bandwidth-bound -- which is the favourable
     regime for this device. Keep `--max-tokens` small (default 32); every extra
     generated token is paid at bandwidth-bound decode speed.

Also enabled: prefix caching (all prompts share the system+instruction preamble) and
a high `--max-num-seqs`. With 128 GB unified you can serve 8B in bf16 (~16 GB) with a
very large KV cache, or 13B comfortably; add `--quantization fp8` for 13B to cut
bandwidth pressure further on Blackwell.

PREREQS
-------
    pip install vllm                       # server mode also needs: requests
    # offline engine (default) needs NO server. For --engine server:
    #   vllm serve NECOUDBFM/Jellyfish-8B --port 8000

USAGE
-----
    # full dataset, offline engine (recommended on DGX Spark)
    python export_jellyfish.py hospital --model NECOUDBFM/Jellyfish-8B

    # smoke test
    python export_jellyfish.py hospital --columns city state --limit 50

    # self-consistency confidence (k samples per cell, one pass)
    python export_jellyfish.py flights --confidence agreement --n-samples 5

Output: ../CARE/csvs/jellyfish_<dataset>_pred.csv  (row_id,column,value,confidence,source)
Then in CARE:
    python retclean_prepare.py  <ds> csvs/jellyfish_<ds>_pred.csv
    python retclean_precheck.py <ds> csvs/jellyfish_<ds>_pred.csv   # FREE gate
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join(HERE, os.pardir, "CARE", "csvs")
_DATA_CANDIDATES = [
    os.path.join(HERE, "datasets"),
    os.path.join(HERE, os.pardir, "RetClean", "datasets"),
    os.path.join(HERE, os.pardir, "raha", "datasets"),
]
SYSTEM_MESSAGE = ("You are an AI assistant that follows instruction extremely well. "
                  "Help as much as you can.")
_ABSTAIN = {None, "", "nan", "n/a", "none", "null"}


# ----------------------------------------------------------------- data helpers

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


def _find_dataset_dir(dataset, override=None):
    cands = ([os.path.join(override, dataset)] if override else []) + \
            [os.path.join(d, dataset) for d in _DATA_CANDIDATES]
    for c in cands:
        if os.path.exists(os.path.join(c, "dirty.csv")):
            return c
    raise SystemExit(f"could not find datasets/{dataset}/dirty.csv in:\n  " +
                     "\n  ".join(cands) + "\nPass --data-dir.")


def build_prompt(dataset, header, row, col_idx, col_start, mode="correct"):
    """Build the cell prompt.

    ``mode="correct"`` (DEFAULT, and the right choice for error correction): the
    record is shown WITH the target attribute's current (possibly dirty) value, and
    the model is asked to correct it. ``mode="impute"``: the target is withheld and
    the model infers it from the other fields (Jellyfish's literal trained task).

    WHY "correct" IS THE DEFAULT -- a measured finding, not a preference:
    with ``impute`` on hospital, the model scored 17% overall and rewrote 84% of the
    CLEAN cells it touched. Inspecting outputs showed why:
      * The dirty value carries most of the signal for correction ("sheffxeld" ->
        "sheffield" is trivial WITH the value, impossible without). Withholding it,
        the model had no city signal and emitted the state capital ("montgomery")
        for nearly every row -- a framing artifact, not a model failure.
      * Format drift: gold "al" vs generated "alabama" -- semantically right, scored
        wrong. Showing the current value anchors the output format.
    Showing the value is also in-distribution for Jellyfish: its trained
    error-detection prompt presents "Attribute for Verification: [X: value]".
    ``impute`` is retained so the two framings can be compared as an ablation.
    """
    target = header[col_idx]
    cur_val = row[col_idx] if col_idx < len(row) else ""
    others = ", ".join(header[j] for j in range(col_start, len(header)) if j != col_idx)

    if mode == "impute":
        fields = ", ".join(f"{header[j]}: {row[j] if j < len(row) else ''}"
                           for j in range(col_start, len(header)) if j != col_idx)
        user = (
            f"You are presented with a {dataset} record that is missing a specific "
            f"attribute: {target}.\n"
            f"Your task is to deduce or infer the value of {target} using the available "
            f"information in the record.\n"
            f"You may be provided with fields like {others} to help you in the inference.\n"
            f"Record: [{fields}]\n"
            f"Based on the provided record, what would you infer is the value for the "
            f"missing attribute {target}?\n"
            f"Answer only the value of {target}."
        )
    else:  # correct
        fields = ", ".join(f"{header[j]}: {row[j] if j < len(row) else ''}"
                           for j in range(col_start, len(header)))
        user = (
            f"Your task is to correct the value of a specific attribute within the "
            f"{dataset} record provided.\n"
            f"Errors may include, but are not limited to, spelling errors, "
            f"inconsistencies, or values that don't make sense given the context of "
            f"the whole record.\n"
            f"Record: [{fields}]\n"
            f"Attribute to correct: [{target}: {cur_val}]\n"
            f"If the current value is already correct, repeat it exactly unchanged. "
            f"Otherwise, provide the corrected value.\n"
            f"Use exactly the same format, casing and abbreviation style as the data "
            f"(for example, if states appear as two-letter codes, answer with a "
            f"two-letter code).\n"
            f"Answer only the value of {target}."
        )
    return f"{SYSTEM_MESSAGE}\n\n### Instruction:\n\n{user}\n\n### Response:\n\n"


def _clean(text):
    t = (text or "").strip().split("\n")[0].strip()
    return None if t.lower() in _ABSTAIN else t


# ----------------------------------------------------------------- engines
# Both engines take a list of prompts and return, per prompt, a list of
# (text, mean_logprob|None) -- one element for greedy, n for agreement mode.

def _run_offline(prompts, args):
    from vllm import LLM, SamplingParams
    kw = dict(model=args.model, dtype=args.dtype,
              gpu_memory_utilization=args.gpu_util,
              max_num_seqs=args.max_num_seqs,
              enable_prefix_caching=True)
    if args.quantization:
        kw["quantization"] = args.quantization
    if args.max_model_len:
        kw["max_model_len"] = args.max_model_len
    llm = LLM(**kw)
    sp = SamplingParams(
        temperature=args.temperature, top_p=0.9, max_tokens=args.max_tokens,
        n=(args.n_samples if args.confidence == "agreement" else 1),
        logprobs=(1 if args.confidence == "logprob" else None),
        stop=["### Instruction:"],
    )
    outs = llm.generate(prompts, sp)          # ONE call: continuous batching does the rest
    packed = []
    for o in outs:
        per = []
        for c in o.outputs:
            n_tok = max(1, len(c.token_ids))
            mlp = (c.cumulative_logprob / n_tok) if c.cumulative_logprob is not None else None
            per.append((c.text, mlp))
        packed.append(per)
    return packed


def _run_server(prompts, args):
    """Fallback: OpenAI-compatible vLLM server. Slower (HTTP overhead) but useful if
    the model is already served or lives on another host."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    url = os.environ.get("JELLYFISH_URL", "http://localhost:8000")

    def one(p):
        payload = {"model": args.model, "prompt": p, "max_tokens": args.max_tokens,
                   "temperature": args.temperature, "top_p": 0.9,
                   "n": (args.n_samples if args.confidence == "agreement" else 1),
                   "stop": ["### Instruction:"]}
        if args.confidence == "logprob":
            payload["logprobs"] = 1
        r = requests.post(f"{url}/v1/completions", json=payload, timeout=args.timeout)
        r.raise_for_status()
        per = []
        for c in r.json()["choices"]:
            lp = (c.get("logprobs") or {}).get("token_logprobs") or []
            vals = [x for x in lp if x is not None]
            per.append((c.get("text"), (sum(vals) / len(vals)) if vals else None))
        return per

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        return list(ex.map(one, prompts))


# ----------------------------------------------------------------- main flow

def export_one(dataset, args):
    ds_dir = _find_dataset_dir(dataset, args.data_dir)
    header, rows = _read_csv(os.path.join(ds_dir, "dirty.csv"))
    # ---- row sharding (for splitting a huge dataset across machines/nights) ----
    # Every prompt is INDEPENDENT (one self-contained record + one target attribute),
    # and row_id is preserved in the output, so shards can be generated in any order,
    # on any machine, and simply concatenated afterwards. No cross-row state exists,
    # so sharding cannot change any repair. Conformal calibration happens later in
    # CARE over the merged log, so it is unaffected too.
    n_total = len(rows)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[:args.limit]
    row_offset = args.offset          # keeps row_id correct when has_index_col is False
    has_idx = _has_index_col(header, rows)
    col_start = 1 if has_idx else 0
    targets = args.columns or header[col_start:]
    row_id = (lambda i: rows[i][0]) if has_idx else (lambda i: str(i + row_offset))

    print(f"\n========== JELLYFISH: {dataset} ==========")
    shard = (f"  SHARD rows [{args.offset}:{args.offset + len(rows)}) of {n_total}"
             if (args.offset or args.limit) else "")
    print(f"rows: {len(rows)}{shard}  has_index_col: {has_idx}  model: {args.model}  "
          f"engine: {args.engine}  confidence: {args.confidence}")
    print(f"columns: {targets}")

    # Build EVERY prompt up front -> one batched generate call (see PERFORMANCE NOTES).
    keys, prompts = [], []
    for col in targets:
        if col not in header:
            print(f"  !! column {col!r} not in header; skipping", file=sys.stderr)
            continue
        ci = header.index(col)
        for i in range(len(rows)):
            keys.append((i, col))
            prompts.append(build_prompt(dataset, header, rows[i], ci, col_start,
                                        mode=args.mode))
    if not prompts:
        raise SystemExit("no prompts built (check --columns)")
    print(f"total prompts: {len(prompts)}  (single batched pass)")

    runner = _run_offline if args.engine == "offline" else _run_server
    packed = runner(prompts, args)

    log_rows = []
    per_col = collections.Counter()
    for (i, col), per in zip(keys, packed):
        if args.confidence == "agreement":
            vals = [v for v in (_clean(t) for t, _ in per) if v]
            if not vals:
                continue
            top, cnt = collections.Counter(vals).most_common(1)[0]
            value, conf = top, cnt / len(vals)
        else:
            text, mlp = per[0]
            value = _clean(text)
            if not value:
                continue
            conf = math.exp(mlp) if mlp is not None else args.default_confidence
        conf = min(1.0, max(0.0, conf))
        log_rows.append((row_id(i), col, value, f"{conf:.4f}", "jellyfish"))
        per_col[col] += 1

    os.makedirs(args.out, exist_ok=True)
    sfx = f"_{args.offset:07d}_{args.offset + len(rows):07d}" if (args.offset or args.limit) else ""
    out_path = os.path.join(args.out, f"jellyfish_{dataset}_pred{sfx}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "column", "value", "confidence", "source"])
        w.writerows(log_rows)

    for col in targets:
        if col in header:
            print(f"  repaired {col}: {per_col[col]} non-abstained cells")
    if log_rows:
        cs = sorted(float(r[3]) for r in log_rows)
        print(f"WROTE: {out_path}  ({len(log_rows)} repairs; confidence "
              f"min/med/max = {cs[0]:.3f}/{cs[len(cs)//2]:.3f}/{cs[-1]:.3f})")
        print("  ^ a SPREAD here (not all one value) is what lets CARE rank within a "
              "stratum; a flat distribution means no ranking signal.")
    else:
        print(f"WROTE: {out_path} (0 repairs -- model abstained everywhere?)")
    print(f"Next (in CARE):  python retclean_prepare.py {dataset} "
          f"csvs/jellyfish_{dataset}_pred.csv")
    print(f"          then:  python retclean_precheck.py {dataset} "
          f"csvs/jellyfish_{dataset}_pred.csv   # FREE gate before the sweep")
    print("=" * (24 + len(dataset)) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--model", default="NECOUDBFM/Jellyfish-8B",
                    help="7B/8B recommended by the authors for unseen tasks; "
                         "13B scores highest on average but is heavier")
    ap.add_argument("--engine", default="offline", choices=["offline", "server"],
                    help="offline = in-process vLLM, ONE batched generate call "
                         "(recommended); server = OpenAI-compatible HTTP")
    ap.add_argument("--columns", nargs="+", default=None)
    ap.add_argument("--mode", default="correct", choices=["correct", "impute"],
                    help="correct = show the current (dirty) value and ask for a "
                         "correction (DEFAULT; the dirty value carries most of the "
                         "signal for repair); impute = withhold it (Jellyfish's "
                         "literal trained task) -- keep for the ablation")
    ap.add_argument("--confidence", default="logprob", choices=["logprob", "agreement"])
    ap.add_argument("--n-samples", type=int, default=5, help="for --confidence agreement")
    ap.add_argument("--temperature", type=float, default=0.35,
                    help="model card's recommended sampling temperature")
    ap.add_argument("--default-confidence", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=32,
                    help="keep SMALL: decode is bandwidth-bound on GB10")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only N rows (after --offset). Smoke tests + sharding.")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N rows. With --limit this defines a SHARD; "
                         "shards are independent and their output CSVs concatenate "
                         "losslessly (see the sharding note in export_one).")
    # engine tuning (offline)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None,
                    help="e.g. fp8 -- cuts bandwidth pressure for 13B on Blackwell")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256,
                    help="scheduler concurrency ceiling; higher = better amortization "
                         "of weight reads on a bandwidth-limited device")
    ap.add_argument("--max-model-len", type=int, default=None)
    # server-mode only
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    for ds in args.datasets:
        export_one(ds, args)


if __name__ == "__main__":
    main()

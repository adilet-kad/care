"""gpt4o_direct.py -- prompted general-purpose LLM as a repair proposer.

THE SCIENTIFIC PURPOSE (this is a control, not "one more baseline")
-------------------------------------------------------------------
Jellyfish-8B certifies 38% of repairs on beers at alpha=0.1. Three explanations are
confounded in that single number:
    (a) Jellyfish is INSTRUCTION-TUNED for data preprocessing;
    (b) it is an LLM at all (world knowledge / pattern completion);
    (c) the CORRECTION-style prompt (show the dirty value) we found matters enormously
        (17% -> 93% on hospital vs the imputation-style prompt).
This driver isolates (a) by holding (b) and (c) fixed: a general-purpose, untuned model
prompted with the **byte-identical prompt** used for Jellyfish.

    GPT-4o-mini >= Jellyfish  ->  tuning for data prep is NOT what buys certifiability;
                                  prompt framing + calibration are.
    GPT-4o-mini <  Jellyfish  ->  task-specific tuning materially improves certifiable
                                  automation -- a concrete, useful finding.

Either outcome is publishable; without this control the Jellyfish result is anecdotal.

PROMPT PARITY IS THE WHOLE POINT
--------------------------------
We import `build_prompt` from the Jellyfish driver rather than re-implementing it, so
the two systems cannot silently drift apart. If that import fails the script refuses to
run instead of quietly falling back to a near-copy -- a near-copy would invalidate the
comparison.

COST (measured prompt size ~300 input / ~10 output tokens per cell)
-------------------------------------------------------------------
gpt-4o-mini at $0.15/1M input, $0.60/1M output (Aug 2026):
    hospital 17,000 + flights 14,256 + beers 24,100 + rayyan 10,000 ~= 65,000 prompts
    input  65k x 300  = 19.5M tok  -> ~$2.93
    output 65k x 10 x n=5          -> ~$1.95      (n is sampled; input billed once)
    TOTAL  ~= $5 for ALL FOUR datasets with self-consistency.
Full `gpt-4o` is ~17x more expensive (~$80) for no expected gain on this task -- the
default is deliberately gpt-4o-mini. Use --model to override if you want the contrast.

Self-consistency (n samples) is the default confidence signal because token logprobs
were measured to be ANTI-correlated with correctness on this task (the model is most
confident when overwriting a rare-but-valid value with a common one).

SETUP
-----
    pip install openai
    # put your key in CARE/baselines/.env  (git-ignored) as:
    #   OPENAI_API_KEY=sk-proj-...
    # or export OPENAI_API_KEY=... in the shell

USAGE (from the CARE repo root)
-------------------------------
    python baselines/gpt4o_direct.py hospital --limit 50 --columns city state   # smoke
    python baselines/gpt4o_direct.py hospital flights beers rayyan              # full

Output: csvs/gpt4o_<dataset>_pred.csv   (row_id,column,value,confidence,source)
Then:   python prepare_log.py  <ds> csvs/gpt4o_<ds>_pred.csv
        python precheck_log.py <ds> csvs/gpt4o_<ds>_pred.csv --scope touched
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
CARE_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_OUT_DIR = os.path.join(CARE_ROOT, "csvs")
_JELLY = os.path.abspath(os.path.join(CARE_ROOT, os.pardir, "Jellyfish"))
_ABSTAIN = {None, "", "nan", "n/a", "none", "null"}


class RateLimiter:
    """Thread-safe request pacer.

    OpenAI enforces requests-per-minute and tokens-per-minute together, and which one
    binds depends on prompt size -- so the right --rps is a property of your tier AND
    your prompt, not a constant. Compute it rather than guessing:

        tokens/request ~= input + n_samples * max_tokens * 0.45
        rps = min(RPM, TPM / tokens_per_request) / 60, then take ~70% for headroom

    Tier 1 (500 RPM / 200K TPM), ~476 tok/call: TOKENS bind at ~420 RPM = 7 req/s,
        so the historical default of --rps 6 was right.
    Tier 2 (5,000 RPM / 2M TPM), ~370 tok/call: 2M/370 = 5,400 RPM, so REQUESTS bind
        at 5,000 RPM = 83 req/s. --rps 6 then uses 7% of the budget and turns a
        6-minute job into 53 minutes. Use --rps 50 (60% of cap) with --workers 64.

    Tier 1 also imposed a 10,000 requests-per-DAY cap that higher tiers drop; the
    driver detects RPD exhaustion separately and stops rather than retrying (see the
    daily_exhausted flag below), because unlike a per-minute 429 it will not recover.
    """

    def __init__(self, rps):
        self._min_interval = 1.0 / max(rps, 0.1)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._min_interval
        if wait:
            time.sleep(wait)


def _load_done(path):
    """Resume support: cells already written in a previous (interrupted) run.

    Every cell is an independent API call, so a partial run is safely resumable --
    and MUST be resumable, because silently dropping cells that failed during peak
    load biases the output (the dropped set is not random)."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done[(r["row_id"], r["column"])] = r
    return done


def _load_jellyfish_prompt():
    """Import the EXACT prompt builder used for Jellyfish. Refuse to run without it:
    a re-implementation could drift and would invalidate the control."""
    if _JELLY not in sys.path:
        sys.path.insert(0, _JELLY)
    try:
        from export_jellyfish import build_prompt, _read_csv, _has_index_col, _find_dataset_dir
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"could not import the Jellyfish prompt builder from {_JELLY}: {e}\n"
            f"This driver deliberately reuses Jellyfish's prompt verbatim so the two "
            f"systems are directly comparable. Fix the path (expected sibling folder "
            f"'Jellyfish' next to CARE) rather than copying the prompt."
        )
    return build_prompt, _read_csv, _has_index_col, _find_dataset_dir


def _load_env():
    """Read baselines/.env if present (KEY=VALUE lines); never overwrite a real env var."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _clean(text):
    t = (text or "").strip().split("\n")[0].strip()
    return None if t.lower() in _ABSTAIN else t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="default gpt-4o-mini (~$5 for all 4 datasets); full gpt-4o is "
                         "~17x the cost for no expected gain on this task")
    ap.add_argument("--columns", nargs="+", default=None)
    ap.add_argument("--mode", default="correct", choices=["correct", "impute"],
                    help="must match the Jellyfish run being compared against")
    ap.add_argument("--n-samples", type=int, default=5,
                    help="self-consistency samples; confidence = modal-answer share")
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel API requests; the pacer below is the real throttle")
    ap.add_argument("--rps", type=float, default=6.0,
                    help="target requests/sec across ALL workers. The default suits a "
                         "TIER-1 key (500 RPM / 200K TPM), where tokens bind at ~7 "
                         "req/s. On TIER 2 (5,000 RPM / 2M TPM) requests bind instead, "
                         "at ~83 req/s: use --rps 50 --workers 64, which cuts a "
                         "19k-request run from ~53 min to ~6 min. Check your tier's "
                         "limits page rather than assuming this default fits.")
    ap.add_argument("--max-retries", type=int, default=8,
                    help="attempts per cell before recording a hard failure")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="flush the output CSV every N completed cells (atomic "
                         "temp+rename). 0 disables. Protects paid API work from crashes.")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells already present in the output CSV (safe: cells are "
                         "independent) -- use after an interrupted run so you do not pay twice")
    ap.add_argument("--max-requests", type=int, default=None,
                    help="stop after N successful requests ACROSS ALL datasets in this "
                         "invocation, then exit cleanly at a checkpoint. Use this to stay "
                         "inside a daily cap (e.g. --max-requests 9500 on a 10,000 RPD "
                         "key) instead of discovering the cap by collecting 429s: cells "
                         "that fail on an exhausted quota are absent from the log, and "
                         "which ones they are depends on timing rather than on data.")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts + print the cost estimate, make NO API calls")
    args = ap.parse_args()

    build_prompt, _read_csv, _has_index_col, _find_dataset_dir = _load_jellyfish_prompt()
    _load_env()

    # Requests spent across ALL datasets in this invocation, so --max-requests is a
    # single budget rather than a per-dataset one.
    spent = [0]
    budget_stop = threading.Event()

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("pip install openai")
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("no OPENAI_API_KEY (put it in CARE/baselines/.env or export it)")
        client = OpenAI()

    stop_all = {"flag": False}
    for dataset in args.datasets:
        if stop_all["flag"]:
            print(f"\n== skipping {dataset}: quota exhausted or budget spent. "
                  f"Re-run with --resume when it resets. ==")
            continue
        ds_dir = _find_dataset_dir(dataset, args.data_dir)
        header, rows = _read_csv(os.path.join(ds_dir, "dirty.csv"))
        if args.offset:
            rows = rows[args.offset:]
        if args.limit:
            rows = rows[:args.limit]
        has_idx = _has_index_col(header, rows)
        col_start = 1 if has_idx else 0
        targets = args.columns or header[col_start:]
        row_id = (lambda i: rows[i][0]) if has_idx else (lambda i: str(i + args.offset))

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

        # Apply --resume BEFORE estimating, so the printed request count and cost
        # describe the work this invocation will actually do. Reporting the full
        # dataset here would make --dry-run useless for planning against a daily cap.
        out_path_early = os.path.join(args.out, f"gpt4o_{dataset}_pred.csv")
        prior = _load_done(out_path_early) if args.resume else {}
        n_total = len(prompts)
        if prior:
            keep = [(k, p) for k, p in zip(keys, prompts)
                    if (row_id(k[0]), k[1]) not in prior]
            keys, prompts = [k for k, _ in keep], [p for _, p in keep]

        est_in = sum(len(p) for p in prompts) / 4.0
        est_out = len(prompts) * args.max_tokens * args.n_samples * 0.4
        cost = est_in / 1e6 * 0.15 + est_out / 1e6 * 0.60
        print(f"\n========== GPT-DIRECT: {dataset} ==========")
        print(f"rows: {len(rows)}  model: {args.model}  mode: {args.mode}  "
              f"n_samples: {args.n_samples}")
        if prior:
            print(f"  --resume: {len(prior)} cells already in the log, "
                  f"{len(prompts)} of {n_total} remaining")
        print(f"requests this run: {len(prompts)}   est. cost: ~${cost:.2f}")
        if args.max_requests:
            print(f"  budget: --max-requests {args.max_requests} "
                  f"({'covers this dataset' if len(prompts) <= args.max_requests else 'will stop partway'})")
        if args.dry_run:
            print("--dry-run: no API calls made.\n")
            continue
        if not prompts:
            print("  nothing left to do.\n")
            continue

        limiter = RateLimiter(args.rps)
        failures = []
        done = [0]
        # Per-DAY exhaustion is terminal for the run; per-MINUTE is transient. Once the
        # daily cap is hit, every remaining cell is guaranteed to fail, and retrying
        # each one 8 times with 60 s backoff wastes hours to produce nothing. This flag
        # short-circuits the rest of the queue so the run ends promptly at a clean,
        # resumable checkpoint instead of grinding.
        daily_exhausted = threading.Event()

        def _is_daily(msg):
            return "RPD" in msg or "per day" in msg or "requests per day" in msg

        def one(idx_prompt):
            idx, prompt = idx_prompt
            if daily_exhausted.is_set() or budget_stop.is_set():
                return idx, None, None          # quota gone or budget spent
            for attempt in range(int(args.max_retries)):
                try:
                    limiter.acquire()
                    r = client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        n=args.n_samples,
                    )
                    vals = [v for v in (_clean(c.message.content) for c in r.choices) if v]
                    done[0] += 1
                    spent[0] += 1
                    if args.max_requests and spent[0] >= args.max_requests:
                        if not budget_stop.is_set():
                            budget_stop.set()
                            print(f"\n  --max-requests {args.max_requests} reached; "
                                  f"finishing in-flight calls and checkpointing.",
                                  file=sys.stderr)
                    if done[0] % 200 == 0:
                        print(f"  ... {done[0]}/{len(prompts)}", end="\r", file=sys.stderr)
                    if not vals:
                        return idx, None, None
                    top, cnt = collections.Counter(vals).most_common(1)[0]
                    return idx, top, cnt / len(vals)
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    if _is_daily(msg):
                        if not daily_exhausted.is_set():
                            daily_exhausted.set()
                            print("\n  !! DAILY request cap (RPD) reached. Stopping this "
                                  "dataset now rather than retrying every remaining "
                                  "cell -- the quota will not return today.",
                                  file=sys.stderr)
                        failures.append((idx, msg[:160]))
                        return idx, None, None
                    is_rate = "rate_limit" in msg or "429" in msg
                    if attempt == int(args.max_retries) - 1:
                        # Do NOT silently drop the cell: dropped cells are not random
                        # (they are the ones that hit peak load), which biases the log.
                        failures.append((idx, msg[:160]))
                        return idx, None, None
                    # full-jitter exponential backoff; rate-limit windows are ~60s, so
                    # cap high enough to actually outlast one.
                    base = min(60.0, 2.0 ** attempt)
                    time.sleep(random.uniform(0.5 * base, base) if is_rate else base)
            return idx, None, None

        os.makedirs(args.out, exist_ok=True)
        out_path = out_path_early
        collected = {}                                # idx -> (value, confidence)

        def _rows_now():
            """Resumed cells + everything completed so far."""
            rows_out = [(r["row_id"], r["column"], r["value"],
                         r.get("confidence", ""), r.get("source", "gpt4o_direct"))
                        for r in prior.values()]
            for idx, (val, conf) in collected.items():
                i, col = keys[idx]
                rows_out.append((row_id(i), col, val, f"{conf:.4f}", "gpt4o_direct"))
            return rows_out

        def _flush():
            """Atomic checkpoint: write to a temp file then rename, so a crash mid-write
            can never leave a truncated CSV that --resume would then trust."""
            tmp = out_path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["row_id", "column", "value", "confidence", "source"])
                w.writerows(_rows_now())
            os.replace(tmp, out_path)

        # Stream results as they COMPLETE and checkpoint periodically. The previous
        # version only wrote at the end of a dataset, so a crash discarded hours of
        # paid API calls (observed: ~6200 cells lost twice). Now a crash costs at most
        # --checkpoint-every cells, and --resume picks up from the last checkpoint.
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(one, ip) for ip in enumerate(prompts)]
            since = 0
            for fut in as_completed(futs):
                idx, val, conf = fut.result()
                if val:
                    collected[idx] = (val, conf)
                since += 1
                if args.checkpoint_every and since >= args.checkpoint_every:
                    _flush(); since = 0
                    print(f"  [checkpoint] {len(collected)} cells saved -> {out_path}",
                          file=sys.stderr)

        log_rows = _rows_now()
        per_col = collections.Counter(r[1] for r in log_rows)

        if budget_stop.is_set():
            print(f"  stopped on --max-requests after {spent[0]} requests; "
                  f"{len(prompts) - len(collected)} cells remain for this dataset.",
                  file=sys.stderr)
        if failures:
            # Distinguish the per-MINUTE limits from the per-DAY one. They need
            # opposite responses, and advising "lower --rps" for an RPD exhaustion is
            # actively wrong: pacing spreads the same daily quota over more hours
            # without raising it.
            rpd = sum(1 for _i, m in failures if "RPD" in m or "per day" in m)
            print(f"  !! {len(failures)} cells FAILED after {args.max_retries} retries "
                  f"and are ABSENT from the log. Dropped cells are not random (they hit "
                  f"peak load), so this biases the result.", file=sys.stderr)
            if rpd:
                print(f"     {rpd} of them hit the DAILY request cap (RPD), not the "
                      f"per-minute one. Lowering --rps will not help; wait for the "
                      f"daily quota to reset and re-run with --resume.", file=sys.stderr)
            else:
                print("     Re-run with --resume to fill them in; lower --rps if it "
                      "persists.", file=sys.stderr)
            print(f"     example: {failures[0][1][:120]}", file=sys.stderr)

        _flush()
        for col in targets:
            if col in header:
                print(f"  repaired {col}: {per_col[col]} non-abstained cells")
        if log_rows:
            cs = sorted(float(r[3]) for r in log_rows)
            print(f"WROTE: {out_path}  ({len(log_rows)} repairs; confidence "
                  f"min/med/max = {cs[0]:.3f}/{cs[len(cs)//2]:.3f}/{cs[-1]:.3f})")
        if daily_exhausted.is_set() or budget_stop.is_set():
            stop_all["flag"] = True
        print(f"Next:  python prepare_log.py {dataset} csvs/gpt4o_{dataset}_pred.csv")
        print("=" * (26 + len(dataset)) + "\n")


if __name__ == "__main__":
    main()

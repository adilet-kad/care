"""Run ZeroEC's released correction.py on one benchmark table.

Executes correction.py's functions and __main__ body unchanged except for the
hard-coded paths, model names, API settings and embedding device. dashscope's
tokenizer is replaced by tiktoken. Records the cells sent to the LLM correction
prompt, the labelled rows and token usage, and stops at --max-calls / --max-usd.
Detection is perfect detection (clean != dirty), as in perfect_detection.py.

    python run_zeroec.py hospital --seed 0 --dry-run      # no API calls
    python run_zeroec.py hospital --seed 0                # needs OPENAI_API_KEY

Output: runs/<dataset>/seed<k>/{corrections.csv, llm_cells.csv, labelled_rows.txt, usage.json}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))

# (clean, dirty) paths in ZeroEC's datasets/ folder
DATASETS = {
    "hospital": ("datasets/hospital/hospital_clean.csv", "datasets/hospital/hospital_dirty.csv"),
    "flights": ("datasets/flights/flights_clean.csv", "datasets/flights/flights_dirty.csv"),
    "beers": ("datasets/beers/beers_clean.csv", "datasets/beers/beers_dirty.csv"),
    "rayyan": ("datasets/rayyan/rayyan_clean.csv", "datasets/rayyan/rayyan_dirty.csv"),
    "tax": ("datasets/tax/tax_clean.csv", "datasets/tax/tax_dirty.csv"),
}

# USD per 1M tokens (input, output); used for the budget guard and the usage report
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00), "gpt-3.5-turbo": (0.50, 1.50)}


def _load_env():
    p = os.path.join(HERE, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _stub_dashscope():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        encode = enc.encode
    except Exception:                       # pragma: no cover - fallback only
        def encode(s):
            return [0] * (len(s) // 4)
    mod = types.ModuleType("dashscope")
    mod.get_tokenizer = lambda name: types.SimpleNamespace(encode=encode)
    sys.modules["dashscope"] = mod


def build_detection(clean_path, dirty_path, out_path):
    """Perfect detection as in perfect_detection.py: clean != dirty after fillna('null')."""
    import pandas as pd
    clean = pd.read_csv(clean_path, dtype=str, encoding="utf-8").fillna("null")
    dirty = pd.read_csv(dirty_path, dtype=str, encoding="utf-8").fillna("null")
    ind = (clean != dirty).astype(int)
    ind.to_csv(out_path, index=False, encoding="utf-8")
    return ind


def _sub_once(body, pattern, repl):
    new, n = re.subn(pattern, lambda m: repl, body, flags=re.M)
    if n != 1:
        raise SystemExit(f"run_zeroec: expected exactly one match for {pattern!r} in "
                         f"correction.py's __main__, found {n}; the release changed?")
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="the per-cell correction model (the paper: gpt-3.5-turbo)")
    ap.add_argument("--gen-model", default=None,
                    help="model for the three generation stages -- correction chain-of-"
                         "thought, rule code, functional dependencies (the paper: gpt-4o); "
                         "default: same as --model")
    ap.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE",
                                                         "https://api.openai.com/v1"))
    ap.add_argument("--labels", type=int, default=10, help="representative tuples (theta)")
    ap.add_argument("--device", default="cpu", help="embedding device: cpu or cuda")
    ap.add_argument("--rpm", type=float, default=120.0,
                    help="LLM requests per minute cap (120 suits an OpenAI Tier-1 key, whose "
                         "200K tokens/min binds at ~130 calls/min; raise it on higher tiers)")
    ap.add_argument("--max-usd", type=float, default=5.0, help="stop if spend exceeds this")
    ap.add_argument("--max-calls", type=int, default=200_000)
    ap.add_argument("--dry-run", action="store_true",
                    help="embed, select labels and print the call/cost bound; no API calls")
    ap.add_argument("--mock", action="store_true",
                    help="pipeline test without API: every LLM call answers 'no change', so "
                         "the run completes, writes every output and repairs nothing")
    ap.add_argument("--out", default=None, help="default: runs/<dataset>/seed<k>")
    a = ap.parse_args()

    os.chdir(HERE)
    _load_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not (a.dry_run or a.mock) and not key:
        raise SystemExit("OPENAI_API_KEY not set (export it or put it in ZeroEC/.env)")
    _stub_dashscope()
    random.seed(a.seed)
    import numpy as np
    np.random.seed(a.seed)

    out = a.out or os.path.join("runs", a.dataset, f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    clean_p, dirty_p = DATASETS[a.dataset]
    det_p = os.path.join(out, "detection.csv")
    det = build_detection(clean_p, dirty_p, det_p)
    n_err = int(det.values.sum())
    n_cols = int((det.sum(axis=0) > 0).sum())
    print(f"[zeroec] {a.dataset}: {len(det):,} rows, {n_err:,} flagged cells in {n_cols} "
          f"columns; seed {a.seed}; model {a.model}"
          f"{'' if not a.gen_model else ' (generation: ' + a.gen_model + ')'}; -> {out}",
          flush=True)

    # ---- load the release: functions at module level, then its own __main__ body ----
    src = open(os.path.join(HERE, "correction.py"), encoding="utf-8").read()
    head, sep, body = src.partition('if __name__ == "__main__":')
    if not sep:
        raise SystemExit("correction.py has no __main__ block; the release changed?")
    g = {"__name__": "zeroec_correction", "__file__": os.path.join(HERE, "correction.py")}
    exec(compile(head, "correction.py", "exec"), g)

    body = textwrap.dedent(body)
    q = repr
    body = _sub_once(body, r"^human_repair_num = .*$", f"human_repair_num = {a.labels}")
    body = _sub_once(body, r"^clean_data_path = .*$", f"clean_data_path = {q(clean_p)}")
    body = _sub_once(body, r"^dirty_data_path = .*$", f"dirty_data_path = {q(dirty_p)}")
    body = _sub_once(body, r"^detection_path = .*$", f"detection_path = {q(det_p)}")
    body = _sub_once(body, r"^output_path = .*$", f"output_path = {q(out)}")
    body = _sub_once(body, r"open\('rayyan_time_cost\.txt'",
                     f"open({q(os.path.join(out, 'time_cost.txt'))}")
    body = body.replace("prompt_templates-old/", "prompt_templates/")
    gen_model = a.gen_model or a.model
    body = _sub_once(body, r"^MODEL_NAME = .*$", f"MODEL_NAME = {q(a.model)}")
    for var in ("MODEL_NAME_auto_cot", "MODEL_NAME_code_generation", "MODEL_NAME_fd_generation"):
        body = _sub_once(body, rf"^{var} = .*$", f"{var} = {q(gen_model)}")
    for var in ("OPENAI_API_BASE", "OPENAI_API_BASE_auto_cot",
                "OPENAI_API_BASE_code_generation", "OPENAI_API_BASE_fd_generation"):
        body = _sub_once(body, rf"^{var} = .*$", f"{var} = {q(a.api_base)}")
    for var in ("OPENAI_API_KEY", "OPENAI_API_KEY_auto_cot",
                "OPENAI_API_KEY_code_generation", "OPENAI_API_KEY_fd_generation"):
        body = _sub_once(body, rf"^{var} = .*$", f"{var} = os.environ.get('OPENAI_API_KEY', 'dry-run')")
    body = _sub_once(body, r"^EMBEDDING_MODEL_PATH = .*$",
                     f"EMBEDDING_MODEL_PATH = {q(os.path.join(HERE, 'all-MiniLM-L6-v2'))}")

    # ---- embeddings: the release's model and quantisation, on the chosen device ----
    from fast_sentence_transformers import FastSentenceTransformer

    class Embeddings(g["Embeddings"]):
        def __init__(self, modelPath):
            self.model = FastSentenceTransformer(modelPath, device=a.device, quantize=True)

        def embed_documents(self, texts):
            return self.model.encode(texts, normalize_embeddings=True)

        def embed_query(self, text):
            return self.model.encode(text, normalize_embeddings=True)

    g["myEmbeddings"] = Embeddings

    # ---- LLM: count every call, pace it, and stop at the budget ----
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0,
             "models_called": []}
    lock = threading.Lock()
    allowed = {a.model, gen_model}
    gap = 60.0 / a.rpm if a.rpm > 0 else 0.0
    last = [0.0]

    def _stop(msg):
        print(f"\n[zeroec] STOP: {msg}", flush=True)
        with open(os.path.join(out, "usage.json"), "w") as fh:
            json.dump({**usage, "stopped": msg}, fh, indent=1)
        os._exit(3)

    Base = g["ChatOpenAI"]

    def _mock_result(messages):
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
        system = messages[0].content if messages else ""
        text = messages[-1].content if messages else ""
        with lock:
            usage["calls"] += 1
        if "detailed chain of thought for corrections" in system:
            n = text.count("Correction:")
            content = json.dumps([{"chain_of_thought_for_correction": "mock",
                                   "error_type": "Typo"}] * n)
        elif "your task is to repair an erroneous value" in system:
            m = re.search(r'Erroneous Value: \{"(.+?)": "(.*)"\}', text)
            col, val = (m.group(1), m.group(2)) if m else ("x", "")
            content = json.dumps({"chain_of_thought_for_correction": "mock",
                                  "correction": {col: val}})
        elif "functional dependency" in system.lower():
            content = "{'functional_dependency': 'None → None'}"
        elif "Python function" in system:
            content = "no code"
        else:
            content = "[]"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))],
                          llm_output={"token_usage": {}})

    class CountingChatOpenAI(Base):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if a.dry_run:
                _stop("dry run reached the first LLM call")
            if a.mock:
                with lock:
                    if self.model_name not in usage["models_called"]:
                        usage["models_called"].append(self.model_name)
                return _mock_result(messages)
            with lock:
                if self.model_name not in usage["models_called"]:
                    usage["models_called"].append(self.model_name)
                    print(f"[zeroec] first call to model {self.model_name!r}", flush=True)
                    if self.model_name not in allowed:
                        _stop(f"a ZeroEC stage asked for {self.model_name!r}, not {allowed}")
                wait = last[0] + gap - time.time()
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.time()
                if usage["calls"] >= a.max_calls:
                    _stop(f"call budget {a.max_calls} reached")
                usage["calls"] += 1
            res = super()._generate(messages, stop=stop, run_manager=run_manager, **kw)
            tu = (res.llm_output or {}).get("token_usage", {}) or {}
            with lock:
                pin, pout = PRICES.get(self.model_name, PRICES["gpt-4o"])
                usage["prompt_tokens"] += tu.get("prompt_tokens", 0)
                usage["completion_tokens"] += tu.get("completion_tokens", 0)
                usage["usd"] += (tu.get("prompt_tokens", 0) * pin
                                 + tu.get("completion_tokens", 0) * pout) / 1e6
                if usage["calls"] % 200 == 0:
                    print(f"[zeroec] {usage['calls']} calls, ${usage['usd']:.3f}", flush=True)
                if usage["usd"] > a.max_usd:
                    _stop(f"spend ${usage['usd']:.2f} exceeded --max-usd {a.max_usd}")
            return res

    g["ChatOpenAI"] = CountingChatOpenAI

    # ---- record which cells the LLM correction prompt handled ----
    llm_cells = []
    orig_repair_value = g["repair_value"]

    def repair_value(dirty_tuple, column, dirty_value, index_row, index_col, chain):
        with lock:
            llm_cells.append((index_row, column))
        return orig_repair_value(dirty_tuple, column, dirty_value, index_row, index_col, chain)

    g["repair_value"] = repair_value

    if a.dry_run:
        # upper bound: every flagged cell reaches the correction prompt
        # (~1,500 prompt + ~150 completion tokens), plus generation calls per column
        calls = n_err + 3 * n_cols
        pin, pout = PRICES.get(a.model, PRICES["gpt-4o"])
        usd = (calls * 1500 * pin + calls * 150 * pout) / 1e6
        print(f"[zeroec] DRY RUN upper bound: <= {calls:,} LLM calls, <= ${usd:.2f} with "
              f"{a.model} if no generated rule fires (rules usually fix most cells)")

    t0 = time.time()
    exec(compile(body, "correction.py:__main__", "exec"), g)

    # ---- outputs CARE's exporter reads ----
    import csv
    with open(os.path.join(out, "llm_cells.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_idx", "column"])
        w.writerows(sorted(set(llm_cells)))
    with open(os.path.join(out, "labelled_rows.txt"), "w") as fh:
        fh.write("\n".join(str(int(i)) for i in g["repair_list"]) + "\n")
    with open(os.path.join(out, "usage.json"), "w") as fh:
        json.dump({**usage, "seconds": round(time.time() - t0, 1), "model": a.model, "gen_model": gen_model,
                   "labels": a.labels, "seed": a.seed}, fh, indent=1)
    print(f"[zeroec] models called: {usage['models_called']}")
    print(f"[zeroec] done: {usage['calls']} LLM calls, ${usage['usd']:.3f}, "
          f"{len(set(llm_cells))} cells via the correction prompt; {out}/corrections.csv")


if __name__ == "__main__":
    main()

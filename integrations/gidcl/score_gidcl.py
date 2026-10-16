"""Score GIDCL's released repairs with its own detector M_det.

confidence = P_Mdet(clean | row with the cell set to the repaired value), the
softmax score GIDCL uses for repair selection. M_det (roberta-base, Ditto) is
retrained in fp32 from the authors' detector/train.csv with detector_train.ipynb
settings; apex and transformers.AdamW are shimmed in memory. Flights' detector
files were built from dirty_50.csv. Run in a CUDA container (run_gidcl_scores.sh).

    python score_gidcl.py hospital beers rayyan tax [--seed k]

Input:  _care_input/gidcl_<ds>_pred.csv (from export_gidcl.py)
Output: _care_output/gidcl_scored_<ds>_pred.csv (gidcl_scored_s<k>_... for seed k > 0)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import os
import random
import sys
import types

csv.field_size_limit(sys.maxsize)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# leading columns the authors' detector files omit
DROP = {"beers": 2, "rayyan": 1}
# table read mode: "strna" = strings, NA -> ""; "pandas" = type inference (tax: 06330 -> 6330)
READ = {"tax": "pandas"}
# notebook settings (detector_train.ipynb)
HP = dict(batch_size=64, max_len=128, lr=3e-5, n_epochs=20, alpha_aug=0.8, lm="roberta-base")


def _shim():
    """Stub apex and transformers.AdamW, imported by ditto/model.py but unused with fp16=False."""
    import torch
    import transformers                     # before the stub: transformers probes for apex
    if not hasattr(transformers, "AdamW"):
        transformers.AdamW = torch.optim.AdamW
    import importlib.machinery
    apex = types.ModuleType("apex")
    amp = types.ModuleType("apex.amp")
    amp.initialize = lambda model, optimizer=None, **k: (model, optimizer) if optimizer is not None else model

    @contextlib.contextmanager
    def scale_loss(loss, optimizer):
        yield loss
    amp.scale_loss = scale_loss
    apex.amp = amp
    apex.__spec__ = importlib.machinery.ModuleSpec("apex", None)
    amp.__spec__ = importlib.machinery.ModuleSpec("apex.amp", None)
    sys.modules.setdefault("apex", apex)
    sys.modules.setdefault("apex.amp", amp)


def _read_table(path, mode, index_col=None):
    import pandas as pd
    if mode == "pandas":
        return pd.read_csv(path, index_col=index_col, low_memory=False).fillna("").astype(str)
    return pd.read_csv(path, index_col=index_col, dtype=str).fillna("")


def _ser(cols, row):
    return "".join(f"COL {c} VAL {v} " for c, v in zip(cols, row))


def score(ds, args):
    import numpy as np
    import pandas as pd
    import torch
    from export_gidcl import _has_index_col, _read
    from ditto.model import DittoDataset, classify, simple_train

    g = os.path.join(args.gidcl, ds)
    log_in = os.path.join(args.inp, f"gidcl_{ds}_pred.csv")
    if not os.path.exists(log_in):
        raise SystemExit(f"{log_in} missing (run_gidcl_scores.sh copies it from CARE/csvs)")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- serialisation, checked against the authors' test.csv
    mode, k0 = READ.get(ds, "strna"), DROP.get(ds, 0)
    clean_cols = list(pd.read_csv(os.path.join(g, "original", "clean.csv"), nrows=0).columns)
    dirty = _read_table(os.path.join(g, "original", "dirty.csv"), mode).iloc[:, k0:]
    cols = clean_cols[k0:]
    dirty.columns = cols
    train = pd.read_csv(os.path.join(g, "detector", "train.csv"), index_col=0, dtype=str,
                        keep_default_na=False)
    test = pd.read_csv(os.path.join(g, "detector", "test.csv"), index_col=0, dtype=str,
                       keep_default_na=False)
    # Ditto reads the first three columns; tax has an extra attribute-name column
    train, test = train.iloc[:, :3], test.iloc[:, :3]
    train.columns = test.columns = [0, 1, 2]
    ours = set(_ser(cols, r) for r in dirty.itertuples(index=False))
    match = test[0].isin(ours).mean()
    print(f"[{ds}] serialisation matches the authors' detector inputs on {match:.1%} of test rows")
    if match < 0.99:
        print(f"[{ds}] WARNING: the released detector files were not built from dirty.csv "
              f"(flights: dirty_50.csv); the detector is still GIDCL's, trained on its data.")

    # ---- M_det, as in detector_train.ipynb (valid = train, as there)
    hp = types.SimpleNamespace(task=f"{ds}-detector-care", save_model=False,
                               logdir="detector_model/", fp16=False,
                               **dict(HP, n_epochs=args.epochs or HP["n_epochs"],
                                      lm=args.lm or HP["lm"]))
    tr = DittoDataset(train, max_len=hp.max_len, lm=hp.lm)
    te = DittoDataset(test, max_len=hp.max_len, lm=hp.lm)
    te_log = DittoDataset(test.iloc[:256], max_len=hp.max_len, lm=hp.lm)   # per-epoch print only
    model = simple_train(tr, tr, te_log, hp)
    model.eval()
    pred, _, _ = classify(te, model=model, lm=hp.lm, max_len=hp.max_len, threshold=0.5)
    y = test[2].astype(int).to_numpy()
    p = np.asarray(pred)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    print(f"[{ds}] M_det on the authors' test.csv: P={P:.3f} R={R:.3f} "
          f"F1={(2*P*R/(P+R) if P+R else 0):.3f}")

    # ---- the released repairs, each placed in its dirty row
    corr = _read_table(os.path.join(g, "correction", "result", "correction.csv"), mode, index_col=0)
    corr = corr.iloc[:, :dirty.shape[1] + k0].iloc[:, k0:]
    corr.columns = cols
    dh, drows = _read(os.path.join(g, "original", "dirty.csv"))
    has_idx = _has_index_col(dh, drows)
    key2i = {(r[0] if has_idx else str(i)): i for i, r in enumerate(drows)}
    with open(log_in, newline="", encoding="utf-8") as fh:
        log = list(csv.DictReader(fh))

    pairs, where = [], []
    for n, r in enumerate(log):
        i = key2i.get(r["row_id"])
        c = r["column"]
        if i is None or c not in cols:
            continue
        row = list(dirty.iloc[i])
        v = corr.iat[i, cols.index(c)]
        row[cols.index(c)] = v
        pairs.append((_ser(cols, row), f"COL {c} VAL {v} ", "0"))
        where.append(n)
    _, p_dirty, _ = classify(DittoDataset(pd.DataFrame(pairs), max_len=hp.max_len, lm=hp.lm),
                             model=model, lm=hp.lm, max_len=hp.max_len, threshold=0.5)
    conf = {n: 1.0 - float(q) for n, q in zip(where, p_dirty)}

    os.makedirs(args.out, exist_ok=True)
    tag = "gidcl_scored" if args.seed == 0 else f"gidcl_scored_s{args.seed}"
    out = os.path.join(args.out, f"{tag}_{ds}_pred.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "column", "value", "confidence", "source"])
        for n, r in enumerate(log):
            w.writerow([r["row_id"], r["column"], r["value"],
                        f"{conf[n]:.6f}" if n in conf else "", "gidcl"])
    vals = sorted(conf.values())
    print(f"[{ds}] scored {len(conf)}/{len(log)} repairs; confidence min/med/max = "
          f"{vals[0]:.3f}/{vals[len(vals)//2]:.3f}/{vals[-1]:.3f}  -> {out}")

    # ---- diagnostic: mean confidence of correct vs wrong repairs
    from export_gidcl import canon
    clean = _read(os.path.join(g, "original", "clean.csv"))[1]
    ci = {c: j for j, c in enumerate(_read(os.path.join(g, "original", "clean.csv"))[0])}
    ok, bad = [], []
    for n, r in enumerate(log):
        if n not in conf or r["column"] not in ci:
            continue
        right = canon(r["value"]) == canon(clean[key2i[r["row_id"]]][ci[r["column"]]])
        (ok if right else bad).append(conf[n])
    if ok and bad:
        print(f"[{ds}] mean confidence: correct repairs {np.mean(ok):.3f} (n={len(ok)}), "
              f"wrong repairs {np.mean(bad):.3f} (n={len(bad)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--gidcl", default=os.path.join(HERE, "GEIL_Data"))
    ap.add_argument("--inp", default=os.path.join(HERE, "_care_input"))
    ap.add_argument("--out", default=os.path.join(HERE, "_care_output"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lm", default=None, help="default: roberta-base, as the authors")
    ap.add_argument("--epochs", type=int, default=None, help="default: the notebook's 20")
    a = ap.parse_args()
    _shim()
    for ds in a.datasets:
        score(ds, a)


if __name__ == "__main__":
    main()

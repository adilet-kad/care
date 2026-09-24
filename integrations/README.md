# Integrations: exporters for the external cleaners

These scripts are **not part of CARE and are never imported by it.** Each one runs
*inside* its own upstream repository, in that repository's own virtualenv, and writes a
CSV that CARE later reads. They live here so the repository is self-contained: how every
proposer log was produced is visible without cloning five upstream repos first.

That separation is the point, not an accident. CARE's claim is that it governs any
proposer through one contract, so it must never depend on a proposer's code. Keeping the
exporters in a directory CARE does not import preserves that: deleting this folder
changes nothing about CARE's behaviour or its test suite.

## The contract

Every exporter emits the same five columns:

```
row_id, column, value, [confidence], [source]
```

- `ref = "{row_id}::{column}"` must match CARE's `CellKey` format.
- An empty `value` is an **abstention**: CARE escalates that cell rather than treating
  it as a repair. This is the mechanism that lets a proposer say "nothing to fix here",
  and Section 8.2 of the paper shows it is what decides robustness to detection error.
- `confidence` is optional. Without it CARE gets a constant score and can only take a
  stratum all-or-nothing.
- `source` is optional per-cell provenance and feeds the evidence-trust gate.

## Deployment

Copy the file into the sibling repo and run it there:

| directory | copy into | environment |
|---|---|---|
| `raha/` | `raha/` | raha's venv (needs scikit-learn, scipy) |
| `jellyfish/` | `Jellyfish/` | CUDA + vLLM container |
| `bclean/` | `BClean/` | Python ≤3.11 (`src/infer.py` uses `DataFrame.append`) |
| `retclean/` | `RetClean/` | Docker stack up (`docker compose up -d`) |
| `holoclean/` | `holoclean/` | Python 3.6/3.7 + PostgreSQL ≥ 9.4 (upstream pins) |

Exact commands, including which flags matter and why, are in `docs/REPRODUCE.md` §3.

## What each one does

**`raha/export_detection.py`**: runs Raha's real detection ensemble and exports the
flagged cells as a work queue (`row_id,column`, no values). Prints precision/recall
against ground truth so detection quality is known *before* a sweep is spent on it.

**`raha/export_baran_raha.py`**: re-runs Baran with a detector's cells as its input
rather than the oracle diff. Needed because Baran's oracle log contains no row for a
detector's false positive, so the non-oracle result cannot be obtained by re-scoring.
Reports accuracy split into *on true errors* vs *on false positives*.

**`jellyfish/export_jellyfish.py`**: offline vLLM pass over a dataset. Supports
`--confidence agreement` (self-consistency over `n` samples) and `--offset/--limit`
row sharding for tables too large for one run.

**`bclean/export_bclean.py`**: runs BClean with the authors' own user-constraint files
and captures the candidate scores it computes internally but discards. `--emit-keep`
additionally records its implicit "keep this value" decisions, which is what makes the
log scoreable against a detector's queue.

**`holoclean/export_holoclean.py`**: runs HoloClean's factor-graph inference and exports
each repaired cell with its **posterior probability**. It is the only proposer that emits a
true probability rather than a heuristic score, which makes it the sharpest test of whether
a model's own confidence substitutes for a distribution-free bound. Run `--probe` first: the
upstream result schema has moved between revisions, and the script reports which access
path worked instead of guessing silently.

**`retclean/build_lake.py`**: builds the reference data lake by majority vote over the
*dirty* table only, so no ground-truth cell can leak into retrieval.

**`retclean/export_retclean.py`**: runs RetClean against that lake, tagging each repair
with whether the lake or the model produced it.

## Keeping these in sync

They are copies. If you change an exporter in its upstream repo, copy it back here, and
note the change in `docs/REPRODUCE.md`'s deviation register if it affects any reported
number.

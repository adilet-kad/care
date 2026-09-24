# CARE: Reproducibility Guide

Everything needed to regenerate every number in the paper, plus the record of every
protocol deviation. **Read §5 before trusting any table**: it lists each choice that
constrains how a result should be read, and the direction it moves the number.

---

## 1. Layout

Six sibling directories under one parent:

```
CARE/        the governance layer + benchmark harness  (this repo)
raha/        Raha/Baran -- detection + classical-ML correction
Jellyfish/   local instruction-tuned LLM proposer (vLLM)
BClean/      Bayesian-network proposer (ICDE'24)
RetClean/    retrieval-augmented LLM proposer (Docker stack)
holoclean/   factor-graph probabilistic repair (Python 3.6/3.7 + PostgreSQL)
```

Inside CARE:

```
care/           library      bench/       harness (run_study = CLI)
baselines/      hosted-LLM drivers        integrations/  exporters for the repos above
tools/          standalone checks         tests/         unit + property suites
csvs/           proposer logs (contract)  experiments/   result CSVs (the evidence)
paper/          generated figs + tables   docs/          this file
```

`integrations/` holds a **copy** of each exporter so the artifact is self-contained.
They are run inside their own repo, never imported by CARE -- deleting that folder
changes nothing about CARE's behaviour. See `integrations/README.md`.

**Zero-coupling rule:** each external cleaner runs natively in its own repo/venv and
exports a CSV; CARE never imports their code. Adding a cleaner therefore never changes
CARE's runtime, and their dependency conflicts never interact.

**Log contract** (the only interface): `row_id,column,value[,confidence][,source]`
- `ref = "{row_id}::{column}"` must match CARE's CellKey format
- empty `value` = abstention (CARE escalates)
- `confidence` optional; absent ⇒ constant s_hat ⇒ all-or-nothing per stratum
- `source` optional per-cell provenance, feeds the trust gate

---

## 2. Environments

Each repo has its own venv; they are **mutually incompatible by design** (BClean needs
pandas<2, CARE is stdlib-first, Jellyfish needs vLLM/CUDA).

```bash
# CARE  (Python 3.10+, stdlib-first: no numpy/pandas/scipy needed)
cd CARE && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e . && pytest -q     # expect all green

# BClean  (Python <=3.11 REQUIRED: src/infer.py uses DataFrame.append, removed in pandas 2.0)
cd BClean && sudo apt-get install libsuitesparse-dev   # scikit-sparse needs SuiteSparse
python3.10 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
#   NOTE: upstream requirements.txt had a conda-local path
#   (certifi @ file:///croot/...) that breaks pip; ours is repaired, original kept as
#   requirements.txt.orig. torch/bnlearn removed -- verified never imported.

# Jellyfish  (needs a CUDA GPU; we used the NVIDIA vLLM container)
docker run --gpus all -it --rm --ipc=host \
  -v <path-to>/Jellyfish:/workspace -v <path-to>/CARE:/care \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace nvcr.io/nvidia/vllm:26.01-py3 <command>

# GPT-4o baseline (inside CARE's venv)
pip install openai   # key in CARE/baselines/.env  (git-ignored)
```

Hardware used: NVIDIA DGX Spark (GB10 Blackwell, 128 GB unified, 273 GB/s).

---

## 3. Regenerating proposer logs

```bash
# --- Baran (classical ML) : run in raha/, oracle detection + 20 gold labels ---
#     see §5 D1 -- this is Baran's own published protocol
cd CARE && python prepare_log.py <ds> csvs/baran_<ds>_pred.csv --tool baran

# --- BClean (Bayesian) ---
cd BClean && source .venv/bin/activate
python export_bclean.py hospital beers flights --with-confidence
#   uses the AUTHORS' UC_json/<ds>.json verbatim (see §5 D3)
#   tax: START SMALL -- domains are huge (zip 39k distinct). See §4.

# --- Jellyfish (local LLM), offline vLLM engine, one batched pass ---
$DOCKER python export_jellyfish.py <ds> --model NECOUDBFM/Jellyfish-8B \
    --confidence agreement --n-samples 5 --out /care/csvs
#   sharding for big datasets: --offset N --limit M  (shards concatenate losslessly;
#   prompts are independent, no cross-row state)

# --- GPT-4o-mini (prompted LLM control) ---
cd CARE && python baselines/gpt4o_direct.py hospital beers flights rayyan --resume
#   ~$5 total; --rps 6 paces under tier-1 limits (500 RPM / 200K TPM)
#   imports Jellyfish's build_prompt VERBATIM so the control is valid (§5 D5)

# --- RetClean (retrieval LLM), Docker stack + zero-leakage lake ---
cd RetClean && docker compose up -d
python build_lake.py <ds> --key <col> --recreate     # majority vote over DIRTY only
python export_retclean.py <ds> --reasoner "GPT-4-OpenAI" --index <ds>_lake

# --- HoloClean (factor graph), Python 3.6/3.7 + PostgreSQL ---
cd holoclean && python export_holoclean.py <ds>              # -> holoclean_<ds>_pred.csv
python export_holoclean.py <ds> --emit-keep                  # -> holoclean_keep_<ds>_pred.csv
#   beers/flights first need prepare_holoclean_input.py <ds> --care-root <CARE>
#   and the constraint files in integrations/holoclean/constraints/ (see §5 D31)

# --- Raha detection, for the non-oracle runs (§5 D25) ---
cd raha && python export_detection.py hospital beers flights rayyan
python export_baran_raha.py hospital beers flights rayyan    # Baran ON the detector's cells
```

**Baran's five reruns per dataset** (its labelled-tuple sample is unseeded, §5 D34).
Each draw goes to its own directory so nothing overwrites:

```bash
cd raha
for ds in hospital beers flights rayyan tax; do
  for i in 1 2 3 4 5; do
    python export_baran_raha.py $ds --oracle --out <CARE>/csvs/baran_var_$ds/$i
  done
done
```

`reproduce.sh` step 4 then sweeps them, and `audit_baran_median.py` collapses the six
draws (these five plus the shipped log) to the medians the tables report.

**Controlled-noise variants** for the error-rate sweep (§5 D32):

```bash
git clone --depth 1 https://github.com/WelkinNi/Automatic-Data-Repair.git /tmp/adr
python tools/import_ni_variants.py --adr /tmp/adr        # -> data/<ds>_ni_inner<level>/
python tools/import_ni_variants.py --verify
# then Baran on each variant, THREE draws each (unseeded sampler):
#   python export_baran_raha.py <variant> --oracle --out <CARE>/csvs/baran_var_ni/<variant>/$i
python audit_alpha_ceiling.py --logs "csvs/*_ni_*_mapped.csv" --per-stratum   # seal the ceilings BEFORE any sweep
python audit_sweep.py                 # verifies the seal, then joins
```

**Always validate a new log before spending a sweep on it:**
```bash
cd CARE
python prepare_log.py  <ds> csvs/<tool>_<ds>_pred.csv      # maps + validates + reports
python precheck_log.py <ds> csvs/<tool>_<ds>_pred.csv --scope touched
```
`precheck_log.py` predicts certifiability for free and reports whether the confidence
signal *discriminates* (top-k accuracy vs pool). If it says a config will escalate
~100%, believe it: this gate has been correct every time it was used.

---

## 4. Regenerating results

```bash
cd CARE && source .venv/bin/activate

# Pareto / certifiable automation (the headline tables)
python -m bench.run_study --dataset <ds> --backends "<tool>:csvs/<tool>_<ds>_mapped.csv" \
  --experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 --scoring <errors|detected> \
  --fast-verify on --out experiments/results_<tool>

# Robustness / Theorems 1 & 2
python -m bench.run_study --dataset <ds> --backends "<tool>:..." \
  --experiment poison --alphas 0.2 --seeds 20 [--trusted-poison] \   # 0.1 for Jellyfish/beers, see reproduce.sh
  --poison-fracs 0.0 0.05 0.1 0.2 --scoring <...>

# Non-oracle detection (Raha ensemble as the work queue)
  ... --detection "log:csvs/raha_<ds>_detected.csv"
```

### Non-oracle detection, end to end (Table 3)

Detection is generated ONCE and reused by every proposer, so the comparison is
controlled: same detector, same cells, only the proposer differs.

```bash
# 0. detection (once per dataset; skip if csvs/raha_<ds>_detected.csv exists)
cd raha && source .venv/bin/activate
python export_detection.py hospital beers flights rayyan

# 1. Baran RE-RUN on Raha's cells  (cannot be re-scored: its oracle log has no row
#    for a detector false positive -- see D25)
python export_baran_raha.py hospital beers flights rayyan
#    prints precision/recall vs oracle, then accuracy split into
#    "on true errors" vs "on detector FALSE POSITIVES" -- the second is the point

# 2. BClean, full-coverage log (it has NO detection input; --emit-keep makes its
#    implicit "keep this value" explicit over UC-declared columns, so the one log
#    can be scored against either queue)
cd ../BClean && source .venv/bin/activate
python export_bclean.py hospital beers flights --with-confidence --emit-keep

# 3. map + precheck, then sweep. --detection MUST match the log, or CARE queues
#    cells the proposer was never asked about
cd ../CARE && source .venv/bin/activate
for ds in hospital beers flights rayyan; do
  python prepare_log.py  $ds csvs/baran_raha_${ds}_pred.csv --tool baran_raha
  python -m bench.run_study --dataset $ds \
    --backends "baran_raha:csvs/baran_raha_${ds}_mapped.csv" \
    --experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 \
    --scoring detected --fast-verify on \
    --detection log:csvs/raha_${ds}_detected.csv \
    --out experiments/results_raha_baran
done
# BClean: prepare with --tool bclean_keep, then
#   --backends "bclean:csvs/bclean_keep_<ds>_mapped.csv"
#   --out experiments/results_raha_bclean
# (--tool is belt-and-braces: prepare_log.py now infers the tool by splitting on the
#  dataset name, so baran_raha/bclean_keep no longer collapse onto baran/bclean and
#  overwrite the oracle logs. Passing it explicitly documents intent.)

python audit_proposers.py && python make_figures.py   # Table 3 auto-populates
```

`make_figures.tab3()` emits a proposer's rows only when BOTH its oracle and non-oracle
sweeps exist, so the table never shows a half-finished comparison.

**Scoring scope: pick deliberately, and report which.**
- `--scoring errors`: score only true error cells (standard repair-F1 population; what
  oracle detection means). Correct for Baran and BClean, whose logs record only cells
  they CHANGE.
- `--scoring detected`: score every proposed cell against a full truth map, so
  preserving a clean cell counts and corrupting one is penalised. Correct for
  Jellyfish/GPT-4o, which propose on every cell. **The only mode under which
  `--detection` affects scoring at all.**

**Long runs.** tax/Baran is ~121K cells; the propose pass alone is hours. Use
`--seeds 10` first. BClean on tax: **start with `--limit 2000`** and scale up, because BClean
scores every candidate in a column's domain per cell, and tax's domains are
`zip` 39,062 / `city` 17,858 / names 10,000 distinct, versus a few hundred on hospital.

---

## 5. Protocol deviations

Every departure from a default or from an upstream system's own protocol, why it was
made, and what it does to the numbers.

| # | Deviation | Why | Consequence |
|---|---|---|---|
| **D1** | **Baran uses oracle detection + 20 ground-truth labels** (`correction.py`'s `__main__`: `detected_cells = get_actual_errors_dictionary()`, `LABELING_BUDGET=20`) | Baran's own published protocol; we did not modify their repo | Baran numbers are an **upper bound**, not deployment. Its logs contain only true error cells, so `detected` ≡ `errors` for Baran |
| **D2** | **Baran/flights excluded** | 4,920 proposals, 100.0% exactly equal to gold, zero clean cells touched. Cause: D1 + flights has only 100 distinct flights over 2,376 rows, so one labelled tuple propagates via the vicinity model | Degenerate perfect-proposer case. Excluded, or labelled as the limit case |
| **D3** | **BClean uses the authors' UC files verbatim** | Authoring our own user constraints would let us tune BClean up or down | BClean only repairs columns its UC declares (hospital 15/17, beers 6/10, flights 6/6). rayyan has no author UC ⇒ out of scope |
| **D4** | **BClean instrumented in memory** to capture its candidate scores | It computes `P(v\|parents)·P(children\|v)·penalty`, sorts, then discards the scores. CARE needs a graded signal | Patch is **additive only**; the selection loop is untouched, so repairs are identical with/without. Disk never modified |
| **D5** | **GPT-4o-mini reuses Jellyfish's prompt verbatim** (imported, not copied) | It is a *control* isolating tuning from prompting; a near-copy would invalidate it | Driver refuses to run if the import fails rather than falling back |
| **D6** | **Jellyfish saw RAHA data in tuning**: hospital is a "seen" dataset per its model card | Unavoidable; the model is pre-trained | Treat hospital as contamination-suspect; headline LLM claims rest on beers/flights/rayyan |
| **D7** | **RetClean lake built from DIRTY data only** (majority vote per key group) | Indexing `clean.csv` would be oracle leakage | Recovers *sparse* corruption (hospital city/state 87–100%) but not *systematic* (beers `ounces` 0%) |
| **D8** | **CDC (AISTATS'24) not run as a proposer** | Its protocol trains on a CLEAN split; these benchmarks have none. Only options were oracle leakage or crippling the baseline | Cited in the paper's related work, with the guarantee distinction stated there |
| **D9** | **tax sampled for LLM proposers** (if run) | 200K rows × 15 cols × n=5 ≈ 15M generations ≈ 5 days | The sample size is recorded with the result; Baran/BClean can run larger |
| **D10** | **Repair matching is exact after lowercase+strip** | Standard for repair F1 | A semantically-correct-but-differently-formatted repair counts as wrong (e.g. `al` vs `alabama`) |
| **D11** | **`fast_verify` path: auto-engages above 200K cells (tax); `reproduce.sh` forces it on for every sweep after verifying bit-identity** (`bench/study.py`) | `Verifier.assess` deep-copies the artifact + runs 3 full-table constraint scans per candidate: 85 ms/cell on hospital, ~25 s/cell on tax's 3M cells (~840 h projected: 121,193 × 25 s). Under the benchmark's constraint set those outputs are invariant (a value repair cannot flip an already-violated completeness floor) | **Not an approximation.** The code probes N cells with the exact verifier, confirms `feasible`/`s_margin`/`delta_objective` are constant, then reuses them; if non-constant it warns and falls back to the exact path. Verified **bit-identical** on hospital (127 repairs, 0 differing cells). With `--fast-verify auto` it engages only above 200K cells; `reproduce.sh` passes `--fast-verify on` everywhere, which `tools/verify_fastpath_equivalence.py` shows is output-identical. ~840 h → ~2 min on tax |
| **D12** | **Two (proposer, dataset) pairs are degenerate and excluded from all claims** | `audit_proposers.py` measures Baran on flights at **4,920/4,920 = 100.00%** and BClean on hospital at **398/398 = 100.00%** accuracy on error cells. CARE then certifies everything at human cost 0.000 and error 0.000 | These are facts about the benchmark, not about CARE: any correct governance layer certifies a perfect proposer wholesale. Baran/flights is dropped from every figure (`EXCLUDE` in `make_figures.py`); BClean/hospital is kept in Table 1 marked `†` and used only as the right-hand endpoint of Fig. 2(b) (file `fig5_mechanism`). **Never quote either as evidence that CARE works.** |
| **D13** | **GPT-4o-mini control is COMPLETE on all four datasets** | Finished on a Tier-2 key at `--rps 50 --workers 64`. Error-cell coverage: hospital 509/509, beers 3,357/3,357, flights 4,901/4,920 (99.6%), rayyan 944/948 (99.6%). The earlier partial state is superseded | The earlier claim that the gaps formed a **contiguous row suffix was wrong**: measured per-column coverage of the partial log was 92.9–95.6% (flights) and 95–100% (hospital), spread across every column, not concentrated. Moot now that coverage is ~100%, but the lesson stands: check gap STRUCTURE empirically rather than inferring it from queue order. Results in `experiments/results_gpt4o/` |
| **D14** | **Top-$k$% confidence curves break ties at RANDOM, fixed seed** (`precheck_log.py`) | Both originally did `recs.sort(reverse=True)` on `(confidence, correct)`, which orders CORRECT answers first inside a tie group. The signals are coarse (self-consistency over 5 samples = 6 values; **65% of tax cells sit at the maximum**), so top-10% was reporting the best 10% of a tie, not a random 10% | **This inflated three of four logs to top-10% = 1.000.** Fair values: Jellyfish/flights 0.513, BClean/flights 0.806, Jellyfish/beers 0.759, Jellyfish/tax 0.718 (seed spread <±0.01). Two paper claims were corrected: BClean/flights top decile 89%→81%, and "top half by agreement contained no errors" was **removed** (fair value 0.760). Per-agreement-level accuracies are tie-order-invariant and are now the primary statement |
| **D15** | **delta is 0.1 everywhere; figures and prose must match the sweeps** | `bench.run_study --delta` defaults to 0.1 and every reported sweep used the default. An earlier draft of Fig. 2(a) (file `fig5_mechanism`) and Section 7.4 computed Clopper-Pearson bounds at delta=0.05 | **Corrected.** `make_figures.fig5` now pins `DELTA = 0.10` with a comment tying it to the sweeps. Numbers changed: tax certified-column bounds [0.013,0.027]->[0.011,0.023], gap upper 0.264->0.256, finite-sample floor 117->103 cells at alpha=0.05 and 29->26 at alpha=0.2. The result files were unaffected; only the figure and the prose describing them changed |
| **D16** | **The `0.83 at nominal 0.90` under-coverage figure was not reproducible** | It described a pre-correction implementation that selected thresholds from observed calibration values. The shipped API has no such mode, so the exact number cannot be regenerated from the artifact | **Removed from the paper.** Replaced with the reproducible statement: data-dependent selection under-covers where the fixed grid does not, which `tests/property/` asserts. Direction confirmed in simulation (0.883 vs 0.983 at n=400) but the point estimate is gone. The calibration-label-noise remark's Monte Carlo figures WERE verified and kept: symmetric 1.000, adversarial 0.600, deflated 1.000 at eta=0.2, delta=0.1 |
| **D17** | **Pace `gpt4o_direct.py` to your tier; the default is Tier-1** | `--rps` default 6.0 assumes Tier 1 (500 RPM / 200K TPM), where TOKENS bind at ~7 req/s. On Tier 2 (5,000 RPM / 2M TPM) with ~370 tok/call, 2M/370 = 5,400 RPM so **REQUESTS** bind at 5,000 RPM = 83 req/s | At `--rps 6` a 19,155-request run takes 53 min; at `--rps 50 --workers 64` it takes 6.4 min. Tier 1's 10,000 RPD cap is dropped at Tier 2. Recompute per tier: `rps = min(RPM, TPM/tok_per_req)/60 * 0.7`. Does not affect any result, only wall-clock |
| **D18** | **tax detection uses only PVD+RVD, and its non-oracle result is NOT comparable to the other datasets** | `export_detection.py tax --algorithms PVD RVD` (OD and KBVD are intractable at 200K rows). prec=0.808 rec=0.765 vs the 4-strategy default used elsewhere | **The finding is a mechanism, not a comparison.** Raha detects 99.8% of `rate` errors (Baran acc 1.0000) and 18% of `zip` (acc 0.0018), 0% of the other 8 columns -> Baran's accuracy on the DETECTED subset is 0.9391 vs 0.7320 overall. Detectability and repairability are correlated through structural redundancy, so a weak detector can act as a filter and RAISE certifiable automation. The oracle counterpart at matching scoring is now in `results_baran_detected/tax_pareto.csv` (D19 satisfied), so the tax row IS in Table 3: A(0.2) 0.7271 -> 0.9391, R=0.000 both sides. **pi_0 predicts both exactly**: oracle queue pi_0=0.7271 (5 perfect cols), detector queue pi_0=0.9391 (only `rate` survives). The detector column is still not comparable to the other four datasets; the oracle/raha pair within the row is |
| **D19** | **A non-oracle automation number needs an oracle counterpart at the SAME scoring** | The published tax oracle run uses `--scoring errors` (`results_baran_errors`); the non-oracle run uses `--scoring detected`. 0.73 -> 0.94 across those two is NOT a valid comparison | DONE for tax: `results_baran_detected/tax_pareto.csv` (`--scoring detected`) gives A=0.7271, identical to the `--scoring errors` run, as expected, since Baran's log contains only error cells so the two scopes coincide for it (D1). Any future non-oracle number needs the same treatment before a delta is quoted. `make_figures.tab3()` already refuses to emit a row unless both sweeps exist, so the table cannot show the invalid pair |
| **D20** | **Theorem 1/2 exhibits are listed explicitly, not globbed** (`make_figures._poison_series`) | A config where CARE certifies too little for any poisoned cell to reach the applied set reports eps_S=0 at every level, so the 'agreement' is between two zeros. The Jellyfish configurations do exactly this (they certify nothing at these budgets) and are kept only as negative evidence | Theorem 2 spans **5 configs / 2 paradigms** (Baran beers+rayyan+hospital, BClean flights+hospital); worst |obs-pred| = 0.007. Theorem 1 spans 5 configs, eps_S=0.000 throughout. Poison runs MUST use per-proposer `--out` dirs: a single-backend run adds no filename suffix, so two proposers on the same dataset overwrite each other |
| **D21** | **RetClean reported on flights only** | `results_retclean/` (re-run; an earlier result directory predated the scoring-scope fix and had no `scoring` column; it was deleted). Certifies nothing at any budget; apply-all error 0.625 | RetClean/hospital is in `EXCLUDE`: its lake retrieved nothing usable so the log is byte-identical to Jellyfish's (D29), and reporting it would double-count one proposer. Its log carries real `retclean_lake` / `retclean_model` provenance, which auto-triggers `mondrian_src` (D23) |
| **D22** | **raha's error count differs from CARE's on beers; always quote CARE's** | `export_baran_raha.py` prints precision/recall against **raha's own** `get_actual_errors_dictionary()` (beers: 4,362 errors -> precision 0.998). CARE's `defects.gold` has 3,357 -> precision 0.768. Cause: the beers dirty/clean header mismatch (`beer_name` vs `beer-name`) resolves differently in the two loaders | **No result is affected**: logs are always scored CARE-side, and Table 3 reads `experiments/detector_quality.csv` (CARE-side). The raha-side printout is a convenience only. **Never quote it in the paper**; one draft sentence did and was corrected |
| **D23** | **BClean `--emit-keep` mixes two confidence scales; needs `mondrian_src`** | Log carries `source=bclean` (chosen repair, margin = dominance) and `source=bclean_keep` (declined repair, emitted as `1-margin`, the semantically correct direction). flights: changes 47.1% acc, conf ok/bad 0.837/0.704; keeps 87.3% acc, 0.719/0.801; **pooled 0.733/0.751 = inverted** | Simpson's paradox. Fixed by `care.conformal.by_product`; `bench.experiments.pareto` now auto-adds `strata=mondrian_src` (column x source) whenever a log carries >1 source, and `make_figures.care(rows,"best")` prefers it. Single-source logs are untouched, so **no previously reported number changed**. Ablation: hospital a=0.1 col-only certifies 0, colxsrc 0.55 at err 0.000; a=0.2 col-only 0.85 @ 0.065 vs colxsrc 0.79 @ 0.000; beers a=0.2 col-only 0.03 vs colxsrc 0.01 (finer strata can certify LESS) |
| **D24** | **BClean's oracle/non-oracle comparison uses a dedicated oracle sweep** | The published oracle BClean numbers use `--scoring errors` on the change-only log; the non-oracle run uses `--scoring detected` on the keep-log. Comparing those directly would confound scoring scope with detection | `experiments/results_bclean_keep_oracle/` re-runs the SAME keep-log under oracle detection with `--scoring detected`, so Table 3's BClean rows differ in detection only. Verified: both dirs report `scoring=detected`, `backend=bclean_keep` |
| **D25** | **Non-oracle detection requires a log that covers the detector's queue** (Jellyfish natively; Baran re-run under the detector, D18/D24; BClean and HoloClean via keep-logs, D24) | `--detection log:csvs/raha_<ds>_detected.csv --scoring detected`. Raha precision 0.77-0.92, recall 0.75-1.00 (`experiments/detector_quality.csv`). A(0.2) falls 0.29->0.06 hospital, 0.38->0.03 beers, 0.14->0 flights; `covered=True` in all 24 CARE rows | Only Jellyfish/GPT-4o logs cover EVERY cell, so re-slicing them onto a different work queue is valid. Baran/BClean logs record only cells they CHANGED under oracle detection, so they have no entry for a detector's false positives -- scoring that silence would measure the protocol, not the proposer. Extending it required re-running Baran under the detector (`results_raha_baran/`) and exporting keep-logs for BClean and HoloClean (`results_raha_bclean/`, `results_raha_holoclean/`); Table 3 now carries all four proposers. Jellyfish results in `experiments/results_raha_jelly/` |
| **D26** | **`--fast-verify on` used for the non-oracle sweeps** | Default `auto` engages only >200K cells, so hospital/beers/flights/rayyan would run the exact path (~85 ms/cell) | Same self-checking probe as D11: it confirms `feasible`/`s_margin`/`delta_objective` are constant under single-cell repair and falls back with a warning otherwise. The probe PASSED on every non-oracle sweep (printed in the run log). Re-derive with `--fast-verify off` |
| **D27** | **Jellyfish on tax is a 25,000-row shard, not the full table** | 17 GPU-h per 25K rows at `--n-samples 5` (1.875M sampled sequences). Rows `[0,25000)`, all 15 columns, 375,000 cells, no gaps | **A shard, not a truncation.** The range was fixed before the run and processed exhaustively, so exchangeability holds within it; chosen a priori rather than determined by when a failure happened. **The sweep MUST pass `--row-range 0 25000`**, or the 105,591 un-sharded error cells are scored as escalations and human cost reads ~0.87 for reasons unrelated to either system. Verify in the run log: `work queue 121219 -> 15628 cells (12.9%)`. Registered in `audit_proposers.SHARDS` and `make_figures.RESULT_OVERRIDE`. Representativeness: 12.9% of tax's errors vs a 12.5% eight-way mean (ratio 1.031); matches the full table on `rate` and `zip` (95% of errors) but over-represents `state` (2.7% vs 0.1–0.3%) and has no `single_exemp`/`child_exemp` errors, because this benchmark injects rare-column corruptions into localised row ranges |
| **D28** | **The "harm" column is not a controlled paradigm comparison** | `clean_broken/clean_touched`: Jellyfish 0.29–0.55 across four datasets; Baran/BClean undefined or on <200 cells | Baran and BClean run under **oracle detection**, which hands them error locations, so they barely touch clean cells. The contrast with the LLM proposers is **confounded by the detection protocol**, not controlled. State it as the magnitude of risk in the detector-free regime, never as "classical correctors are safe and LLMs are not". Said explicitly in the paper at the point of claim |
| **D29** | **RetClean on hospital is not an independent proposer** | Its lake retrieved no usable evidence there, so every repair fell through to the same Jellyfish backend. `csvs/retclean_hospital_mapped.csv` carries `source=jellyfish` on every row and scores identically: **0.4224 accuracy on all 509 error cells, matching the Jellyfish log to four decimals**, with identical $\pi_0$ and identical certified automation. The two logs are not byte-identical (see D30); they are two samples of one stochastic proposer | Stated in the paper's Experimental Setup. RetClean is reported only on flights, where retrieval does change the outcome (0.377 vs Jellyfish's 0.018 on the same cells). Do **not** count RetClean/hospital as a fifth proposer. |
| **D30** | **The tuned LLM is not cell-level reproducible; its reported statistics are** | `csvs/retclean_hospital_mapped.csv` and `csvs/jellyfish_hospital_mapped.csv` are two independent executions of the same model on the same 17,000 cells (D29 explains why the RetClean run reduces to the Jellyfish backend). They agree on **91.9%** of all cells and **80.7%** of the 509 gold-error cells: the model does not return the same answer twice. Every statistic the paper reports is nevertheless identical: `acc_on_errors` **0.4224** in both, $\pi_0=0.051$ in both, certified automation $0.00$ at every budget in both | No reported claim rests on a particular cell. Every Jellyfish figure is an aggregate over $\ge 509$ cells, and the certified rows sit far enough from every budget that cell-level variation cannot move them. Recorded because anyone executing the exporter will see different cells and should know which quantity reproduces. |
| **D31** | **HoloClean's denial constraints for beers and flights are not HoloClean's: they are the schema-supported subset of Ni et al.'s (PVLDB 2024) rules** | HoloClean ships DCs for hospital only. Its `testdata/flight.csv` is a different extraction (57,246 rows × 8 cols against CARE's 2,376 × 6, `scheduled_dept` against `sched_dep_time`), so its constraints do not apply and its logs would not key against CARE's gold set; beers is absent entirely | D3's "authors' constraints verbatim" holds for hospital and stops holding here. The four flights DCs (`flight → {sched,act}_{dep,arr}_time`) and two beers DCs (`brewery_id → city, state`) were written from the **schema**, before any result, with one structural check: does the determinant repeat? (`brewery_id` 558 distinct over 2,410; `flight` 100 over 2,376; `id` 2,410 over 2,410 → rejected as inert). `city → state` was rejected as false in the US. Every kept rule also appears in Ni et al.'s constraint files; we omit three of their five beers rules (they constrain `brewery_name`/`beer_name`, which CARE's schema spells `brewery-name`/`beer-name`, so `prepare_log.py` would discard those repairs as out-of-schema) and two of their six flights rules (`sched_arr_time → act_arr_time`, which a late flight violates legitimately, and a rule equating a departure column with an arrival column that we read as a slip in the upstream file). Reasoning recorded in `integrations/holoclean/constraints/README.md`. **Consequence:** beers yields only 121 constraint-driven repairs under 757 `ibu` null-imputations and is reported as degenerate, not as a Table 1 row |
| **D32** | **Ni et al.'s noise variants are imported with a synthetic `index` column, and six of their files are refused** | `bench.datasets.load` treats a first column named `id`, `index` or similar, or holding only integers as the row key. Ni's files have none, so beers' `id`, rayyan's `id` and hospital's `ProviderNumber` would each be taken as one, dropped from the scored schema *and* used as a colliding key (hospital inner-30 has 45 distinct `ProviderNumber` values over 1,000 rows). Separately, Ni's flights `clean.csv`/`dirty.csv` differ on **98.6%** of cells (time formatting), which the `outer` and `inner_outer` files inherit | `tools/import_ni_variants.py` prepends `index` = 1..N to both files, so the key is synthetic and every injected attribute stays scored; and it refuses any pair differing on more than 50% of cells, which excludes all six flights `outer`/`inner_outer` files by measurement rather than by assertion. Verified: 38/38 variants load with `len(defects.gold)` equal to the independently measured differing-cell count, and the two anchors reproduce CARE's own tables exactly (beers 3,357, hospital 509). Flights is therefore swept on `inner` variants only |
| **D33** | **Earlier drafts stated $\|\Lambda\|=17$; the shipped grid, and the current paper, use 21 candidates** | `care.conformal.rcps._DEFAULT_GRID = 20` yields $\Lambda=\{0,0.05,\dots,1\}$, and nothing in `bench/` overrides it, so every reported sweep used $m=21$ and a Bonferroni level $\delta/21$ | The paper's floor figures are the 21-grid ones: $\ln(\|\Lambda\|/\delta)/\alpha \approx 107$ at $\alpha=0.05$ and $27$ at $\alpha=0.2$ (exact smallest certifying $n$: 105 / 51 / 24 calibration cells, i.e. 263 / 128 / 60 proposed cells at a 40% calibration split). No result file depends on the prose; `audit_alpha_ceiling.py` reads $\|\Lambda\|$ from the library so the floors cannot drift from the code |
| **D34** | **Baran's active-learning sample is stochastic and unseeded; the certified automation is not a single number; Table 1 reports medians over six draws** | Five reruns per corpus plus the shipped log = six draws (`audit_baran_median.py` → `experiments/baran_median.csv`), on hospital, beers, rayyan and tax (flights is omitted: Baran is exactly correct there in every draw, D12). Spread of $A$, max−min: **hospital** 0.400 marginal / 0.629 column at α=0.2; **beers** 1.000 / 0.213; **rayyan** 1.000 / 0.613. Accuracy on rayyan ranged 0.788–0.956 across draws. On tax accuracy ranged 0.732–0.956 across draws while group-conditional $A(0.1)$ stayed at 0.73 in five of six (`experiments/tax_draws.csv`); marginal control certified all-or-nothing exactly when pooled accuracy cleared $1-\alpha$ (18/18). **Coverage held in all 150 rerun configurations** (5 corpora × 5 reruns × 3 budgets × 2 stratifications), worst realised error 0.152 against α=0.2 | Table 1's Baran rows are column-wise medians over the six draws (the shipped hospital log was the most favourable draw, the shipped tax log the least accurate). Not seeded away: a seed would hide the variance behind a number reproducible only on our build. flights is the control: Baran solves it outright (4,920/4,920) and the spread is exactly zero, so the variance is the proposer's, not the harness's. Baran emits constant confidence, so a single pooled stratum can only certify all or none; when its accuracy straddles $1-\alpha$ a draw lands on one side or the other, which is the bimodality. Neither stratification is reliably steadier (marginal is calmer on hospital, column-wise on beers), so we make no claim that stratification stabilises certification. |

**Reproducing the equivalence check:**
```bash
cd CARE            # compares fast_verify=False vs True on the same cells
python tools/verify_fastpath_equivalence.py   # expects: IDENTICAL: True (differing cells: 0)
```

**Dataset quirk worth knowing:** dirty and clean CSVs do **not** share headers:
beers uses `beer_name`/`brewery_name` in dirty but `beer-name`/`brewery-name` in clean;
hospital uses raha lowercase in dirty and CARE capitalized in clean; flights matches.
Any new adapter must match columns by normalized name (see
`BClean/export_bclean.py::_norm`).

---

## 6. Known-good sanity gates

```bash
cd CARE
pytest -q                                    # full suite must be green
grep -c "grid" care/conformal/rcps.py        # fixed-grid + Bonferroni fix present
```

Red flags that have each caught a real bug in this project:

1. **`covered=True` with `human_cost=1.00`**: escalate-all, not success. Always read
   automation next to coverage.
2. **Two runs byte-identical when they shouldn't be**: caught the scoring-scope bug
   (log-detection and oracle runs matching despite 17k vs 509 proposals).
3. **A poison run with `human_cost=1.00` at poison=0**: vacuous; ε_S is 0 trivially.
   `run_study` now prints a VACUOUS RUN warning.
4. **100% exact-vs-gold with zero clean cells touched**: oracle-derived or degenerate
   log. Caught Baran/flights.
5. **Near-constant confidence** (`distinct<=2` in `prepare_log.py`): no ranking signal;
   CARE can only take strata all-or-nothing.

---

## 7. Result inventory

`experiments/README.md` lists every result directory and audit CSV, what produced it, and
which table or figure reads it. The short version: `results_*/` are `bench.run_study`
sweeps (one CSV per dataset), the top-level CSVs are the outputs of the `audit_*.py`
scripts and `bench.throughput`, and `make_figures.py` turns both into `paper/figures` and `paper/tables`.

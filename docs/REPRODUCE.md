# Reproducing the results

`./reproduce.sh` regenerates every result file, table and figure from the cleaner logs
in `csvs/`; nothing below is needed for that. This file describes how the logs were
produced and records each protocol decision that affects how a number should be read.

## 1. Layout

CARE never imports a cleaner. Each cleaner runs in its own repository and environment and
writes one CSV in the log contract (`row_id, column, value, [confidence], [source]`);
the exporters are kept in `integrations/`. Expected layout:

```
CARE/          this repository
raha/          Raha (detection) and Baran
BClean/        BClean
BCleanplus/    BClean+
holoclean/     HoloClean
GIDCL/         GIDCL (with GEIL_Data.zip)
ZeroEC/        ZeroEC
```

## 2. Environments

| repository | environment |
|---|---|
| CARE | Python ≥ 3.10; `pip install -r requirements.txt && pip install -e .` |
| raha | its own venv (scikit-learn, scipy) |
| BClean | Python ≤ 3.11 (`src/infer.py` uses `DataFrame.append`); `libsuitesparse-dev` for scikit-sparse |
| BClean+ | CPU image from `requirements-cpu.txt`; UC generation (flan-t5-xxl) in a CUDA container |
| HoloClean | Python 3.6/3.7 and PostgreSQL ≥ 9.4 |
| GIDCL | Python 3 standard library for the export; detector retraining in a CUDA container |
| ZeroEC | its own Docker image (`Dockerfile.care`); `OPENAI_API_KEY` in `ZeroEC/.env` |

All runs used one NVIDIA DGX Spark (GB10, 128 GB unified memory).

## 3. Producing the logs

Copy the files from `integrations/<cleaner>/` into the cleaner's repository and run them
there. Each writes `CARE/csvs/<cleaner>_<table>_pred.csv`.

```bash
# Baran: oracle detection, 20 labelled tuples (its published protocol)
cd raha && python export_baran_raha.py hospital beers flights rayyan tax --oracle
# five more draws per table (Baran's labelled-tuple sample is unseeded)
for ds in hospital beers flights rayyan tax; do for i in 1 2 3 4 5; do
  python export_baran_raha.py $ds --oracle --out ../CARE/csvs/baran_var_$ds/$i; done; done

# Raha detection and Baran behind it (real-detector runs)
python export_detection.py hospital beers flights rayyan
python export_detection.py tax --algorithms PVD RVD
python export_baran_raha.py hospital beers flights rayyan tax

# BClean: the authors' constraint files; --emit-keep also logs the cells it keeps
cd ../BClean && python export_bclean.py hospital beers flights --with-confidence
python export_bclean.py hospital beers flights --with-confidence --emit-keep

# BClean+
cd ../BCleanplus && bash run_bclean_plus.sh setup && bash run_bclean_plus.sh best

# HoloClean (beers and flights need prepare_holoclean_input.py and constraints/)
cd ../holoclean && python export_holoclean.py hospital
python export_holoclean.py hospital --emit-keep

# GIDCL: released output; then its detector score, three seeds
cd ../GIDCL && bash run_gidcl.sh && bash run_gidcl_scores.sh

# ZeroEC: gpt-4o-mini, three runs per table
cd ../ZeroEC && bash run_zeroec.sh build
for ds in hospital beers flights rayyan; do bash run_zeroec.sh run $ds 0 1 2; done
bash run_zeroec.sh export hospital beers flights rayyan
```

Then, in CARE, map and check each log before sweeping it:

```bash
python prepare_log.py  <table> csvs/<cleaner>_<table>_pred.csv    # map to the schema, validate
python precheck_log.py <table> csvs/<cleaner>_<table>_pred.csv    # accuracy and confidence signal
bash integrations/sweep_offline_logs.sh <cleaner> <table>...       # map, score, sweep
```

Error-rate variants for the controlled-corruption sweep (Ni et al.):

```bash
git clone --depth 1 https://github.com/WelkinNi/Automatic-Data-Repair.git /tmp/adr
python tools/import_ni_variants.py --adr /tmp/adr && python tools/import_ni_variants.py --verify
# Baran on each variant (three draws), then:
python audit_alpha_ceiling.py --logs "csvs/*_ni_*_mapped.csv" --per-stratum   # ceilings, sealed first
python audit_sweep.py                                                        # checks the seal, joins
```

## 4. Producing the results

`./reproduce.sh` (or `reproduce.ps1`) runs, in order: the oracle sweeps of the six
cleaners, the real-detector sweeps, the poisoning runs, Baran's extra draws, tax, the
error-rate sweep, the audits, throughput and `make_figures.py`, then the checks of
Section 6. A single sweep:

```bash
python -m bench.run_study --dataset <table> --backends "<cleaner>:csvs/<cleaner>_<table>_mapped.csv" \
  --experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 --scoring errors \
  --fast-verify on --out experiments/results_<cleaner>
```

`--scoring errors` scores error cells only (oracle detection); `--scoring detected` scores
every proposed cell, so an overwritten correct cell counts as an error. Add
`--detection log:csvs/raha_<table>_detected.csv` for a real detector's queue; the log
must then cover that queue (a keep-log, or a rerun behind the detector).

## 5. Protocol decisions and deviations

| # | decision | effect on the numbers |
|---|---|---|
| P1 | Baran runs with oracle detection and 20 ground-truth labelled tuples, its published protocol. | Its rows are upper bounds, not deployment figures. Its logs contain only error cells, so `errors` and `detected` scoring coincide. |
| P2 | Baran's labelled-tuple sample is unseeded; Baran rows are medians over six draws (`audit_baran_median.py`). | The spread is reported, not seeded away. |
| P3 | Degenerate pairs: Baran/flights, and BClean, BClean+ and HoloClean on hospital, are 100% accurate on every error cell they touch. | Baran/flights is excluded; the others are marked † and are not evidence for CARE. |
| P4 | BClean uses the authors' constraint files unchanged; BClean+ uses the authors' released UCs. | Only declared columns are repaired; tables without author constraints are not scored. BClean+ on rayyan exceeds 100 GB and is not reported. |
| P5 | BClean is instrumented in memory to keep the candidate scores it computes and discards. | The selection loop is unchanged; repairs are identical with and without it. |
| P6 | HoloClean's denial constraints for beers and flights are the schema-supported subset of Ni et al.'s (`integrations/holoclean/constraints/`). | HoloClean/beers is dominated by null imputations outside the gold standard and is not scored. |
| P7 | GIDCL runs from the authors' released output; its detector is retrained in fp32 with their code and data, three seeds. Its flights detector data come from the 50%-error variant. | GIDCL's confidence is our reconstruction; flights is unranked. |
| P8 | ZeroEC uses gpt-4o-mini for every stage (its paper used GPT-4o and GPT-3.5-turbo), three runs per table; the 10 user-labelled rows are dropped from the log. | Confidence is run agreement, of our construction. tax is not run. |
| P9 | A repair is correct if it equals the clean value after lowercasing and stripping. | Format-only rewrites count as wrong (`tools/audit_overwrite_formats.py` reports them). |
| P10 | Verification uses the incremental index (`--fast-verify on`), exact for the completeness floor, FDs and CFDs. | No approximation: `tools/verify_fastpath_equivalence.py` and `tools/verify_incremental.py` compare it with the exact path. |
| P11 | δ = 0.1 and \|Λ\| = 21 in every reported run except the sensitivity sweep (`audit_sensitivity.py`), which varies both. | — |
| P12 | tax detection uses Raha's PVD and RVD only (OD and KBVD do not scale to 200K rows). | The tax real-detector row is not comparable to the other tables. |
| P13 | A real-detector number is compared only with an oracle run under the same scoring (`results_baran_detected/`, `results_*_keep_oracle/`). | Rows differ in detection only. |
| P14 | Logs with more than one source are stratified by column × source. | Single-source logs are unaffected. |
| P15 | Poisoning results are reported for the listed configurations (`make_figures._poison_series`), each in its own output directory. | Configurations that admit nothing would make the bound hold trivially and are not used. |
| P16 | Ni et al.'s variants get a synthetic `index` key; pairs that differ on more than 50% of cells (flights `outer`, `inner_outer`) are refused. | flights is swept on `inner` variants only. |
| P17 | Error counts are CARE's (`defects.gold`); Raha's own printout differs on beers because of a header mismatch. | No reported number uses the Raha-side count. |

## 6. Checks

```bash
pytest -q
python tools/verify_fastpath_equivalence.py     # incremental and exact verification agree
python tools/verify_incremental.py hospital --sample 0
python tools/verify_precondition_fallback.py
python tools/verify_gate_first.py               # gating the calibration sample changes no result
python tools/verify_results_reproduce.py        # stored sweeps regenerate from their commands
```

Warning signs when adding a cleaner: `covered=True` with `human_cost=1.00` (nothing was
automated); a poison run that escalates everything at zero contamination (the bound holds
trivially, `run_study` warns); 100% accuracy with no correct cell touched (an
oracle-derived or degenerate log); `distinct ≤ 2` confidence values in `prepare_log.py`
(no ranking signal, so a stratum is admitted whole or not at all).

Dirty and clean tables do not always share headers (beers: `beer_name` against
`beer-name`), so exporters match columns by normalised name.

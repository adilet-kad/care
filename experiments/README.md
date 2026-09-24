# Result files

Every number reported for CARE is read from a file in this directory by `make_figures.py`
or by one of the `audit_*.py` scripts; nothing is transcribed by hand. `./reproduce.sh`
regenerates all of it from the proposer logs in `csvs/`, and
`tools/verify_results_reproduce.py` checks that each stored sweep still reproduces from
its documented command.

Directory names follow `results_<experiment>[_<proposer>][_<dataset>]`. A directory holds
one CSV per dataset.

## Sweep results (written by `bench.run_study`)

| directory | what it is | used for |
|---|---|---|
| `results_baran_errors/` | Baran under its published protocol (oracle detection, error-cell scoring), the shipped draw | Table 1, mechanism and Pareto figures |
| `results_baran_detected/` | the same logs under "detected" scoring, so the oracle and real-detector columns of Table 3 are scored alike | Table 3 |
| `results_baran_var_<dataset>/1..5/` | five reruns of Baran per dataset; its labelled-tuple sampler is unseeded, so one log is one draw | Table 1 medians, per-draw markers, Limitations |
| `results_bclean/`, `results_holoclean/` | BClean and HoloClean, change-only logs, oracle detection | Table 1 |
| `results_bclean_keep_oracle/`, `results_holoclean_keep_oracle/` | the same proposers' keep-inclusive logs (their implicit "leave this cell alone" decisions made explicit) | Table 3, stratification section |
| `results_detected/`, `results_gpt4o/`, `results_retclean/` | Jellyfish, GPT-4o-mini and RetClean | Table 1 |
| `results_jelly_tax25k/` | Jellyfish on the tax row shard `[0, 25000)`; swept with `--row-range` so unseen rows are not scored as escalations | Table 1 |
| `results_raha_baran/`, `results_raha_bclean/`, `results_raha_holoclean/`, `results_raha_jelly/` | the same proposer logs behind Raha's real detector instead of the oracle | Table 3, real-detection section |
| `results_constraints_baran/`, `results_constraints_jelly/` | a second detector (a completeness-constraint predicate) to separate detector precision from recall | real-detection section |
| `results_poison_<proposer>_<dataset>/` | corruption injected through an untrusted channel (Theorem 1) and a trusted one (Theorem 2), at poison fractions 0–0.2 | Table 2, robustness section |
| `results_sweep_baran_var/<variant>/<draw>/` | Baran on controlled-noise variants of four corpora, three to four unseeded draws each (draw `0` is the first run, `1`–`3` the repeats) | error-rate sweep figure |

The five `results_poison_baran_*` and `results_poison_bclean_*` directories are the
Theorem 2 exhibits. The two `results_poison_jellyfish_*` directories are kept because the
paper reports *why* an LLM proposer cannot supply one: the gate certifies nothing there,
so no poisoned cell can reach the applied set and the contamination stays at zero.

**Columns.** A `*_pareto.csv` row is one (baseline, stratification, α) configuration:
`human_cost` is the fraction escalated (so automation is `1 − human_cost`),
`realized_error` the error of the auto-applied set, `err_hi` its one-sided bootstrap
upper bound, and `covered` whether that bound is at or below α, the flag the guarantee
is judged on. A `*_poison.csv` row adds `poison_frac`, `applied_contamination` (ε_S),
`thm2_bound` and `within_thm2_bound`.

## Audit outputs

| file | written by | used for |
|---|---|---|
| `proposer_accuracy.csv` | `audit_proposers.py` | Table 1: repairs on error cells, accuracy, π₀, damage to already-correct cells |
| `detector_quality.csv` | `audit_proposers.py` | Table 3: the detector's precision and recall |
| `fp_handling.csv` | `audit_proposers.py` | what each proposer does with the clean cells a detector wrongly flags |
| `baran_median.csv` | `audit_baran_median.py` | Table 1's Baran rows, as medians over the six draws |
| `tax_draws.csv`, `tax_draws.strata.csv` | `audit_alpha_ceiling.py` | per-draw and per-column statistics for tax |
| `calibration.csv`, `calibration_bins.csv` | `audit_calibration.py` | is self-reported confidence a probability? (pooled gap, ECE, per-bin accuracy) |
| `recalibration.csv` | `audit_recalibration.py` | does Platt or isotonic recalibration fix it? |
| `heldout_budget.csv` | `audit_heldout_budget.py` | the naive held-out threshold against the finite-sample bound |
| `sensitivity.csv` | `audit_sensitivity.py` | coverage and automation over the \|Λ\| × δ grid |
| `label_budget.csv` | `audit_label_budget.py` | certifiable automation against the number of labelled cells |
| `error_drop.csv` | `audit_error_drop.py --raha` | error drop rate, unattended apply-all against CARE |
| `drift_probe.csv` | `audit_drift_probe.py` | calibrating on some sources and deploying on others |
| `throughput.csv` | `bench.throughput` | governance cost per repair, exact path against the hoisted one |
| `sweep.csv`, `sweep_median.csv` | `audit_sweep.py` | the joined error-rate sweep and its per-point medians |
| `sweep_predictions.csv`, `.sha256`, `.strata.csv` | `audit_alpha_ceiling.py` | the per-stratum ceilings, sealed before any calibration was run; `audit_sweep.py` checks the digest before joining |

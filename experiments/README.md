# Result files

Every number in the paper is read from a file in this directory, by `make_figures.py`
or by one of the audit scripts. `./reproduce.sh` regenerates all of them from the logs
in `csvs/`, and `tools/verify_results_reproduce.py` checks that each stored sweep
regenerates from its command.

## Sweeps (`bench.run_study`, one CSV per table)

| directory | contents | paper |
|---|---|---|
| `results_baran_errors/` | Baran, oracle detection, error-cell scoring (shipped draw) | Table 4, Figs. 2, 4 |
| `results_baran_var_<table>/1..5/` | five more Baran draws per table (its labelled-tuple sample is unseeded) | Table 4 medians |
| `results_bclean/`, `results_bclean_plus/`, `results_holoclean/` | change-only logs, oracle detection | Table 4 |
| `results_gidcl/` | GIDCL on the oracle queue | Table 8 |
| `results_gidcl_owndet/` | GIDCL with its own detector, scored over every cell it wrote | Table 4, Table 8 |
| `results_gidcl_scored{,_s1,_s2}/` | GIDCL with the retrained detector's score, three seeds | ranking of GIDCL's score |
| `results_zeroec/` | ZeroEC, gpt-4o-mini, three-run agreement as confidence | Table 4 |
| `results_bclean_keep_oracle/`, `results_holoclean_keep_oracle/` | keep-logs (cells the cleaner kept are listed too), oracle detection | Table 8, strata |
| `results_baran_detected/` | Baran under "detected" scoring, the oracle side of Table 8 | Table 8 |
| `results_raha_baran/`, `results_raha_bclean/`, `results_raha_holoclean/` | the same cleaners behind the Raha detector | Table 8 |
| `results_constraints_baran/` | Baran behind a completeness-constraint detector | real detection |
| `results_poison_<cleaner>_<table>/` | corruption through untrusted and trusted sources | Table 7 |
| `results_sweep_baran_var/<variant>/<draw>/` | Baran on Ni et al.'s error-rate variants (draw 0 is the first run) | Fig. 3 |

A `*_pareto.csv` row is one (baseline, strata, α) configuration. `baseline` is `CARE`,
`ApplyAll` (write everything) or `EscalateAll` (review everything); `human_cost` is the
escalated share, so automation is `1 − human_cost`; `realized_error` is the error of the
written set and `err_hi` its one-sided bootstrap upper bound; `covered` is
`err_hi ≤ α`. A `*_poison.csv` row adds `poison_frac`, `applied_contamination` (ε_S)
and the Proposition 2 bound.

## Audits

| file | written by | contents |
|---|---|---|
| `proposer_accuracy.csv` | `audit_proposers.py` | repairs on error cells, accuracy, π₀, overwritten correct cells |
| `detector_quality.csv`, `fp_handling.csv` | `audit_proposers.py` | detector precision/recall; what cleaners do with detector false positives |
| `baran_median.csv` | `audit_baran_median.py` | Baran's rows as medians over six draws |
| `tax_draws.csv`, `tax_draws.strata.csv` | `audit_alpha_ceiling.py` | per-draw, per-column statistics on tax |
| `calibration.csv`, `calibration_bins.csv` | `audit_calibration.py` | mean confidence against accuracy, ECE |
| `recalibration.csv` | `audit_recalibration.py` | Platt and isotonic recalibration |
| `heldout_budget.csv` | `audit_heldout_budget.py` | naive held-out threshold against the bound (Fig. 5) |
| `label_budget.csv` | `audit_label_budget.py` | automation against labelled cells |
| `sensitivity.csv` | `audit_sensitivity.py` | coverage and automation over \|Λ\| × δ |
| `error_drop.csv` | `audit_error_drop.py` | error drop rate, apply-all against CARE |
| `drift_probe.csv` | `audit_drift_probe.py` | calibrating on some flights sources, deploying on the rest |
| `gidcl_confidence.csv` | `audit_gidcl_confidence.py` | AUROC of GIDCL's retrained-detector score |
| `sweep.csv`, `sweep_median.csv` | `audit_sweep.py` | the joined error-rate sweep |
| `sweep_predictions.csv` (+ `.sha256`, `.strata.csv`) | `audit_alpha_ceiling.py` | per-stratum ceilings, sealed before calibration |
| `incremental_verify.csv` | `tools/verify_incremental.py` | incremental vs exact verification with FDs |
| `scaling.csv` | `tools/audit_scaling.py` | build time, index memory, verification and calibration time on 10K–200K tax rows (Table 2; 2-core container) |
| `planner_validation.csv`, `planner_savings.csv`, `planner_breakeven.csv` | `tools/audit_planner.py` | label planner: prediction accuracy, savings (Table 6), break-even sizes |
| `procedures.csv` | `tools/audit_procedures.py` | alternative threshold procedures (Table 5) |
| `partitions.csv` | `tools/audit_partitions.py` | column, data-type, cardinality and planner partitions |
| `trust_noise.csv` | `tools/audit_trust_noise.py` | mislabelled trust, gate before and after calibration |
| `drift_monitor.csv`, `drift_monitor_more.csv` | `tools/audit_drift_monitor.py` | drift between corruption levels and re-certification (beers; flights and hospital) |
| `overwrite_formats.csv` | `tools/audit_overwrite_formats.py` | overwritten correct cells that are format-only |
| `throughput.csv` | `bench.throughput` | cost per repair |

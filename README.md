# CARE: Controlled Automation for Data Repair

CARE is a governance layer between a data cleaner and the database. For each repair a
cleaner proposes, it decides whether the repair can be written without review or must be
escalated to a reviewer, so that the error rate among the written repairs stays below a
budget α with probability at least 1−δ. CARE never runs or modifies the cleaner; it reads
the cleaner's output from one CSV file:

```
row_id, column, value, [confidence], [source]      # an empty value means "nothing to fix"
```

<p align="center"><img src="docs/img/architecture.png" width="720" alt="CARE architecture"></p>

This repository contains the library, the evaluation harness, the logs of the six
cleaners studied in the paper (Baran, BClean, BClean+, HoloClean, GIDCL, ZeroEC) on five
benchmark tables, every result file, and scripts that regenerate every table and figure.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m pytest -q          # unit and property tests
./reproduce.sh --quick       # all results except tax and the error-rate sweep
./reproduce.sh               # everything (tax needs about 16 GB of RAM)
./reproduce.sh --verify      # only check that stored results regenerate
./reproduce.sh --from 6      # resume a run at step 6
```

On Windows use `reproduce.ps1` with `-Quick` or `-Verify`. No GPU or API key is needed:
everything is replayed from the logs in `csvs/`.

The benchmark tables (hospital, beers, flights, rayyan, tax) are the Raha/Baran suite
and are not redistributed; `bench/datasets.py` downloads them into `data/` on first use,
and the full run imports Ni et al.'s error-rate variants the same way (needs `git`).
The admission path needs only `pydantic`; `matplotlib` is needed for figures.

## Layout

| path | contents |
|---|---|
| `care/` | the library: Learn-then-Test threshold search, Clopper–Pearson bounds, strata, constraint verification (exact and incremental), trust gate, label planner |
| `bench/` | evaluation harness: `run_study` (sweeps), the evaluation protocol, dataset loader, throughput |
| `csvs/` | cleaner logs: `<cleaner>_<table>_pred.csv` as exported, `_mapped.csv` after schema mapping; detector queues; Baran's extra draws (`baran_var_*/`); error-rate variants (`*_ni_*`) |
| `experiments/` | result CSVs; every number in the paper is read from these (`experiments/README.md`) |
| `paper/` | the paper source, with figures and tables written by `make_figures.py` (mapping below) |
| `audit_*.py`, `tools/` | audit and verification scripts |
| `integrations/` | exporters that produced each log, run inside each cleaner's own repository |
| `tests/` | unit and property tests |
| `docs/REPRODUCE.md` | environments, how each log was produced, and the register of protocol decisions and deviations that the paper refers to |

## Paper figures and tables

`make_figures.py` writes the generated figures and tables of the paper from
`experiments/`. File names say what a figure or table shows and carry no number, so
they stay valid when the paper's numbering changes. Tables 1 (related work), 2 (how
each cleaner was run), 4 and 5 are written in the paper source itself, the last two
from the CSVs named below. Files named `extra_*` hold results that are quoted in the
text or kept only in this repository.

| in the paper | file | function | reads |
|---|---|---|---|
| Fig. 1, the problem on one run | `paper/fig_problem.tex` (TikZ) | — | numbers from `csvs/baran_var_tax/2/` and `results_baran_var_tax/2/tax_pareto.csv` |
| Fig. 2, the pipeline | `paper/fig_architecture.tex` (TikZ) | — | — |
| Fig. 3, per-column bounds and the accuracy/automation grids | `paper/figures/fig_mechanism.pdf` | `fig_mechanism` | (a) `csvs/baran_var_tax/2/baran_tax_mapped.csv`, the run of Fig. 1; (b, c) `proposer_accuracy.csv`, `results_*_pareto.csv` |
| Fig. 4, Baran under controlled corruption | `paper/figures/fig_corruption.pdf` | `fig_corruption` | `sweep_median.csv` |
| Fig. 5, every pair at the main budget | `paper/figures/fig_budget.pdf` | `fig_budget` | `results_*_pareto.csv` |
| Fig. 6, acceptance without a bound | `paper/figures/fig_naive_threshold.pdf` | `fig_naive` | `heldout_budget.csv` |
| Algorithm 1, CARE on one batch | code, not a file | `bench/study.py` `evaluate_split`, `care/verify/`, `care/conformal/controller.py` (`calibrate`, `decide`), `care/conformal/rcps.py` (`rcps_threshold`), `care/escalate/active.py` (`route`); the audit in `tools/audit_drift_monitor.py` | — |
| Algorithm 2, planning the labels | code, not a file | `care/conformal/planner.py` (`plan`) | pilot labels drawn in `tools/audit_planner.py` |
| Table 3, every scored pair without and with CARE, at budgets of 5% and 20% | `paper/tables/tab_main.tex` | `tab_main` | `results_*_pareto.csv`, `baran_median.csv`, Baran's draws in `results_baran_var_*/` |
| Table 4, alternative threshold procedures | in `paper/care_main.tex` | — | `experiments/procedures.csv` (`tools/audit_procedures.py --tax`) |
| Table 5, human cost with the label planner | in `paper/care_main.tex` | — | `experiments/planner_savings.csv` (`tools/audit_planner.py`) |
| Table 6, oracle versus real detection | `paper/tables/tab_detector.tex` | `tab_detector` | `results_raha_*`, `results_gidcl*`, `results_baran_detected`, `detector_quality.csv` |
| quoted in the text: verifier cost (Section 4.2), sensitivity to the grid and δ (Section 7.3), corruption through trusted sources (Section 7.6) | `paper/tables/extra_scaling.tex`, `experiments/sensitivity.csv`, `paper/tables/extra_poison.tex` | `tab_scaling`, —, `tab_poison` | `scaling.csv`, `sensitivity.csv`, `results_poison_*` |
| repository only: calibration, error drop, error-rate sweep table, GIDCL's score, label budget, planner break-even | `paper/tables/extra_*.tex`, `paper/figures/extra_*.pdf` | `tab_calibration`, `tab_error_drop`, `tab_sweep`, `tab_gidcl_confidence`, `fig_label_budget`, `fig_planner` | see `experiments/README.md` |

## One sweep by hand

```bash
python prepare_log.py tax csvs/baran_tax_pred.csv          # map to CARE's schema and validate
python -m bench.run_study --dataset tax \
  --backends "baran:csvs/baran_tax_mapped.csv" \
  --experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 \
  --scoring errors --fast-verify on --out experiments/results_baran_errors
python make_figures.py
```

Use `--detection log:csvs/raha_<table>_detected.csv` for a real detector's queue instead
of the true error cells. Each result row reports CARE and the two baselines, apply-all
(write everything) and escalate-all (review everything).

## Adding a cleaner

Write an exporter that emits the contract above (`integrations/README.md`), then run
`bash integrations/sweep_offline_logs.sh <cleaner> <table>...`. An empty `value` tells
CARE the cleaner abstains on that cell; `confidence` lets CARE admit part of a column;
`source` feeds the trust gate.

## Hardware

All runs used one NVIDIA DGX Spark (GB10, 128 GB unified memory, Ubuntu 24.04). CARE is
pure Python (3.10 or later).

## License

Apache License 2.0 (`LICENSE`). The benchmark tables and the cleaners keep their own
licenses; this repository contains none of their code, only the logs their runs produced.

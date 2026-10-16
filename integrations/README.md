# Exporters

These scripts run inside each cleaner's own repository and environment and write a log
that CARE reads. CARE never imports them; removing this directory changes nothing in
CARE or its tests.

## Log contract

```
row_id, column, value, [confidence], [source]
```

- `row_id::column` must match CARE's `CellKey`.
- An empty `value` is an abstention: CARE escalates the cell.
- `confidence` is optional; without it a stratum is certified whole or not at all.
- `source` is optional provenance for the trust gate.

## Deployment

| directory | copy into | environment |
|---|---|---|
| `raha/` | `raha/` | raha's venv |
| `bclean/` | `BClean/` | Python ≤ 3.11 |
| `bclean_plus/` | `BCleanplus/` | CPU image from `requirements-cpu.txt`; UC generation in a CUDA container |
| `holoclean/` | `holoclean/` | Python 3.6/3.7, PostgreSQL ≥ 9.4 |
| `gidcl/` | `GIDCL/` | Python 3; detector retraining in a CUDA container |
| `zeroec/` | `ZeroEC/` | Docker image from `Dockerfile.care` |

After a cleaner has written its logs, from the CARE root:

    bash integrations/sweep_offline_logs.sh <cleaner> <table>...

maps each log (`prepare_log.py`), scores it (`tools/score_log.py`) and runs the sweep
into `experiments/results_<cleaner>/`. Commands for every cleaner are in
`docs/REPRODUCE.md`.

## Scripts

- `raha/export_detection.py`: Raha's detection ensemble; writes the flagged cells as a
  work queue and prints precision and recall.
- `raha/export_baran_raha.py`: Baran, either with oracle detection (`--oracle`) or on a
  detector's cells. Baran's oracle log has no row for a detector false positive, so the
  real-detector run is a rerun, not a rescoring.
- `bclean/export_bclean.py`: BClean with the authors' constraint files; keeps the
  candidate scores BClean computes and discards. `--emit-keep` also logs the cells it
  keeps, so the log can be scored against a detector's queue.
- `bclean_plus/`: BClean+ with the authors' released UCs (`run_bclean_plus.sh best`);
  `generate_UC.py` output for other tables (`uc`, `generated`). Confidence as for BClean.
- `holoclean/export_holoclean.py`: HoloClean's repairs with their posterior probability.
  `--probe` reports which result schema the installed revision exposes.
  `prepare_holoclean_input.py` and `constraints/` are needed for beers and flights.
- `gidcl/export_gidcl.py` (`run_gidcl.sh`): every cell GIDCL's released cleaned table
  changes, after an md5 check that its dirty table is CARE's. Serialisation artefacts
  (`12.0` for `12`) are counted, not logged.
- `gidcl/score_gidcl.py` (`run_gidcl_scores.sh`): attaches the score of GIDCL's detector,
  retrained with the authors' settings, as confidence (`gidcl_scored_<table>_pred.csv`).
- `zeroec/` (`run_zeroec.sh`): ZeroEC's released `correction.py` with gpt-4o-mini; see
  `zeroec/README.md`.

If an exporter is changed in its repository, copy it back here.

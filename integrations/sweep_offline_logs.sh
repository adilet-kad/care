#!/usr/bin/env bash
# sweep_offline_logs.sh -- certify a cleaner's repair logs with CARE.
#
# Run from the CARE repository root after the cleaner's runner has written
# csvs/<tool>_<ds>_pred.csv. Maps each log onto CARE's schema, scores it and runs
# the pareto sweep.
#
#   bash integrations/sweep_offline_logs.sh gidcl hospital beers flights rayyan tax
#   bash integrations/sweep_offline_logs.sh bclean_plus hospital beers flights
#
# Output: csvs/<tool>_<ds>_mapped.csv, experiments/results_<tool>/<ds>_pareto.csv
# The tax sweep needs about 16 GB of RAM.

set -uo pipefail
cd "$(dirname "$0")/.."
# interpreter: $PYTHON, else .venv, else python3
if [ -n "${PYTHON:-}" ]; then PY=$PYTHON
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi
$PY -c "import pydantic" 2>/dev/null || { echo "CARE's dependencies not found for $PY (pip install -r requirements.txt)"; exit 1; }
TOOL=${1:?usage: sweep_offline_logs.sh <tool> <dataset>...}; shift
DATASETS=${*:-"hospital beers flights rayyan"}
FAILED=""

for ds in $DATASETS; do
  pred=csvs/${TOOL}_${ds}_pred.csv
  [ -f "$pred" ] || { echo "$pred missing; run the cleaner's runner first"; FAILED="$FAILED $ds"; continue; }
  printf '\n== %s / %s\n' "$TOOL" "$ds"
  $PY prepare_log.py "$ds" "$pred" --tool "$TOOL" | grep -E "alignment|confidence|WROTE"
  $PY tools/score_log.py "$ds" "csvs/${TOOL}_${ds}_mapped.csv"
  $PY -m bench.run_study --dataset "$ds" --backends "$TOOL:csvs/${TOOL}_${ds}_mapped.csv" \
      --experiment pareto --alphas 0.05 0.1 0.2 --seeds 10 --fast-verify on \
      --scoring errors --out "experiments/results_${TOOL}" || FAILED="$FAILED $ds"
done

[ -n "$FAILED" ] && { echo "FAILED:$FAILED"; exit 1; }
echo "done; results in experiments/results_${TOOL}"

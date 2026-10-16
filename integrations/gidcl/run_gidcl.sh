#!/usr/bin/env bash
# run_gidcl.sh -- produce GIDCL repair logs for CARE.
#
# Copy this file and export_gidcl.py into the GIDCL root. Unpacks GEIL_Data.zip if
# needed and diffs each released correction.csv against CARE's dirty table.
#
#   bash run_gidcl.sh                        # hospital beers flights rayyan tax
#   bash run_gidcl.sh hospital rayyan        # a subset
#
# Environment: CARE=../CARE

set -uo pipefail
cd "$(dirname "$0")"
CARE=${CARE:-../CARE}
DATASETS=${*:-"hospital beers flights rayyan tax"}

if [ ! -d GEIL_Data ]; then
  [ -f GEIL_Data.zip ] || { echo "GEIL_Data.zip not found (see GIDCL's README, 'Output and Evaluation')"; exit 1; }
  python3 -m zipfile -e GEIL_Data.zip . || exit 1
fi
python3 export_gidcl.py $DATASETS --gidcl GEIL_Data --care "$CARE" || exit 1
printf '\n== next, in CARE:  bash integrations/sweep_offline_logs.sh gidcl %s\n' "$DATASETS"

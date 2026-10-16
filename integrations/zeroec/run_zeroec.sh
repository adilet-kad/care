#!/usr/bin/env bash
# run_zeroec.sh -- build, test, run and export ZeroEC for CARE.
# Run from the ZeroEC repository root with OPENAI_API_KEY=sk-... in ZeroEC/.env
# (.env and runs/ are added to .git/info/exclude).
#
#   bash run_zeroec.sh build                       # docker image care-zeroec
#   bash run_zeroec.sh mock hospital               # whole pipeline, no API calls
#   bash run_zeroec.sh dry  hospital               # embed + label selection, cost bound
#   bash run_zeroec.sh run  hospital 0 1 2         # three seeds
#   bash run_zeroec.sh export hospital beers ...   # -> ../CARE/csvs/zeroec_*_pred.csv
#
# Extra flags after "--" go to run_zeroec.py, e.g.  run tax 0 -- --max-usd 15
set -euo pipefail
cd "$(dirname "$0")"
IMG=care-zeroec
RUNS=${RUNS:-runs}     # set RUNS and TAG to keep a variant apart
grep -qs '^\.env$' .git/info/exclude || printf '.env\nruns/\n' >> .git/info/exclude 2>/dev/null || true
dock() { docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work -w /work "$IMG" "$@"; }
stage=${1:?usage: run_zeroec.sh build|mock|dry|run|export ...}; shift
case "$stage" in
  build)  docker build -f Dockerfile.care -t "$IMG" . ;;
  mock)   ds=${1:?dataset}; shift; dock python run_zeroec.py "$ds" --mock --out "runs/$ds/mock" "$@" ;;
  dry)    ds=${1:?dataset}; shift; dock python run_zeroec.py "$ds" --dry-run --out "runs/$ds/dry" "$@" ;;
  run)    ds=${1:?dataset}; shift
          seeds=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do seeds+=("$1"); shift; done
          [ "${1:-}" = "--" ] && shift
          [ ${#seeds[@]} -eq 0 ] && seeds=(0)
          for s in "${seeds[@]}"; do
            if [ -f "$RUNS/$ds/seed$s/corrections.csv" ]; then echo "$RUNS/$ds/seed$s done; skipping"; continue; fi
            rm -rf "$RUNS/$ds/seed$s"
            dock python run_zeroec.py "$ds" --seed "$s" --out "$RUNS/$ds/seed$s" "$@" 2>&1 \
              | tee "${RUNS}_${ds}_seed${s}.log" \
              | grep --line-buffered -E '^\[zeroec\]|completed|Precision|Traceback|Error' || true
          done ;;
  export) python3 export_zeroec.py "$@" --runs "$RUNS" --tag "${TAG:-zeroec}" --care ../CARE ;;
  *) echo "unknown stage $stage"; exit 1 ;;
esac

#!/usr/bin/env bash
# run_bclean_plus.sh -- produce BClean+ repair logs for CARE.
#
# Copy this file, export_bclean_plus.py, run_generate_uc.sh, Dockerfile.care and
# requirements-cpu.txt into the BClean+ root and run from there. CARE's tables are
# copied to _care_input/; logs are written to _care_output/ and copied to CARE's csvs/.
#
#   bash run_bclean_plus.sh setup        # build the CPU image (Python 3.11 + SuiteSparse)
#   bash run_bclean_plus.sh best         # hospital, beers, flights with the authors' released UCs
#   bash run_bclean_plus.sh uc           # rayyan, tax: generate UCs with flan-t5-xxl (GPU)
#   bash run_bclean_plus.sh generated    # rayyan, tax with the generated UCs (tax on a 20K-row shard)
#
# Environment: CARE=../CARE  WORKERS=32  TAX_LIMIT=20000 (0 = full table)
#              GPU_IMAGE=nvcr.io/nvidia/vllm:26.01-py3  CPU_IMAGE=care-bclean-plus
#

set -uo pipefail
cd "$(dirname "$0")"
CARE=$(cd "${CARE:-../CARE}" && pwd) || { echo "CARE repository not found"; exit 1; }
WORKERS=${WORKERS:-32}
TAX_LIMIT=${TAX_LIMIT:-20000}
GPU_IMAGE=${GPU_IMAGE:-nvcr.io/nvidia/vllm:26.01-py3}
CPU_IMAGE=${CPU_IMAGE:-care-bclean-plus}
say() { printf '\n== %s\n' "$*"; }

stage_tables() {       # copy CARE's dirty/clean tables into _care_input/data
  for ds in "$@"; do
    mkdir -p "_care_input/data/$ds"
    cp "$CARE/data/$ds/dirty.csv" "$CARE/data/$ds/clean.csv" "_care_input/data/$ds/" \
      || { echo "$CARE/data/$ds missing (run CARE once to download the tables)"; exit 1; }
  done
}

export_logs() {        # uc-mode dataset [args]; runs in the CPU image, copies to CARE
  local uc=$1 ds=$2; shift 2
  docker image inspect "$CPU_IMAGE" >/dev/null 2>&1 || { echo "run 'bash run_bclean_plus.sh setup' first"; exit 1; }
  mkdir -p _care_output
  docker run --rm -v "$PWD":/workspace -w /workspace --user "$(id -u):$(id -g)" \
    -e HOME=/tmp "$CPU_IMAGE" \
    python export_bclean_plus.py "$ds" --uc "$uc" --with-confidence --workers "$WORKERS" \
      --data-dir _care_input/data --out _care_output "$@" || { echo "export failed for $ds"; return 1; }
  cp _care_output/bclean_plus_"$ds"*_pred.csv "$CARE/csvs/" && echo "  copied to $CARE/csvs/"
}

case "${1:-}" in
  setup)
    say "building $CPU_IMAGE from Dockerfile.care"
    docker build -t "$CPU_IMAGE" -f Dockerfile.care . || exit 1 ;;

  best)
    stage_tables hospital beers flights
    for ds in hospital beers flights; do
      say "BClean+ with UC_json/${ds}_schema_best.json: $ds"
      export_logs best "$ds" || exit 1
    done ;;

  uc)
    say "writing CARE's rayyan and tax tables, key column stripped, to _care_input/"
    mkdir -p _care_input
    for ds in rayyan tax; do
      python3 - "$CARE/data/$ds/dirty.csv" "_care_input/${ds}_for_uc.csv" <<'PY'
import csv, sys
csv.field_size_limit(sys.maxsize)
src, dst = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(src, newline="", encoding="utf-8")))
first = rows[0][0].strip().lower()
vals = [r[0] for r in rows[1:] if r]
key = first in {"index", "tid", "id", "row_id", "key", ""} or \
      (bool(vals) and all(v.strip().lstrip("-").isdigit() for v in vals))
with open(dst, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    for r in rows:
        w.writerow(r[1:] if key else r)
print(f"{dst}: {len(rows) - 1} rows, key column {'dropped' if key else 'absent'}")
PY
    done
    say "generate_UC.py inside $GPU_IMAGE (only this folder is mounted)"
    docker run --gpus all --rm --ipc=host \
      -v "$PWD":/workspace -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
      -w /workspace "$GPU_IMAGE" bash run_generate_uc.sh rayyan tax || exit 1 ;;

  generated)
    for ds in rayyan tax; do
      [ -f "UC_json/${ds}_schema.json" ] || { echo "UC_json/${ds}_schema.json missing: run 'uc' first"; exit 1; }
    done
    stage_tables rayyan tax
    say "BClean+ with generated UCs: rayyan"
    export_logs generated rayyan || exit 1
    say "BClean+ with generated UCs: tax (limit=$TAX_LIMIT; 0 = full table)"
    if [ "$TAX_LIMIT" = 0 ]; then export_logs generated tax
    else export_logs generated tax --limit "$TAX_LIMIT"; fi || exit 1 ;;

  *)
    sed -n '2,16p' "$0"; exit 1 ;;
esac

say "next, in CARE:  bash integrations/sweep_offline_logs.sh bclean_plus <datasets>"

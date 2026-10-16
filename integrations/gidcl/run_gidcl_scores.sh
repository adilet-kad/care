#!/usr/bin/env bash
# run_gidcl_scores.sh -- score GIDCL's released repairs with its own detector.
#
# Run from the GIDCL root after run_gidcl.sh, with score_gidcl.py and export_gidcl.py
# copied there. Retrains GIDCL's detector in a GPU container and scores each repair.
#
#   bash run_gidcl_scores.sh                     # hospital beers flights rayyan tax
#   bash run_gidcl_scores.sh hospital rayyan
#
# Environment: CARE=../CARE  IMAGE=nvcr.io/nvidia/vllm:26.01-py3  SEED=0
#   SEED=1 bash run_gidcl_scores.sh hospital beers rayyan   -> gidcl_scored_s1_<ds>_pred.csv

set -uo pipefail
cd "$(dirname "$0")"
CARE=$(cd "${CARE:-../CARE}" && pwd) || { echo "CARE repository not found"; exit 1; }
IMAGE=${IMAGE:-nvcr.io/nvidia/vllm:26.01-py3}
SEED=${SEED:-0}
DATASETS=${*:-"hospital beers flights rayyan tax"}

mkdir -p _care_input _care_output
for ds in $DATASETS; do
  cp "$CARE/csvs/gidcl_${ds}_pred.csv" _care_input/ || { echo "run run_gidcl.sh first"; exit 1; }
done

docker run --gpus all --rm --ipc=host \
  -v "$PWD":/workspace -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -e PYTHONUNBUFFERED=1 -w /workspace "$IMAGE" bash -c "pip install -q scikit-learn 2>&1 | tail -1; \
    python score_gidcl.py $DATASETS --seed $SEED" || exit 1

TAG=gidcl_scored; [ "$SEED" != 0 ] && TAG=gidcl_scored_s$SEED
for ds in $DATASETS; do cp "_care_output/${TAG}_${ds}_pred.csv" "$CARE/csvs/" || exit 1; done
echo "copied to $CARE/csvs/"
printf '\n== next, in CARE:  bash integrations/sweep_offline_logs.sh %s %s\n' "$TAG" "$DATASETS"

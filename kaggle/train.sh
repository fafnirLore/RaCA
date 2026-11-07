#!/usr/bin/env bash
# Train one configuration on Kaggle (1 GPU). Usage: bash kaggle/train.sh <config> <run_name> [extra train.py args]
#   e.g. bash kaggle/train.sh kaggle/configs/raca_full.py raca_full --cfg-options runner.max_epochs=3
# --deterministic (cuDNN deterministic algorithms, no benchmark autotuning) matches tools/dist_train.sh.
# Output: /kaggle/working/runs/<run_name>/ (checkpoints, logs, train_full.log). Progress lines are printed live.
set -o pipefail
CONFIG=$1; RUN=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/tmp/mra_env/bin/python}
OUT=${OUT_ROOT:-/kaggle/working/runs}/$RUN; mkdir -p "$OUT"
# PYTHONPATH: MRA's train.py imports its own 'projects' package (dist_train.sh does the same)
PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=0 timeout ${TIMEOUT:-11h} $PY tools/train.py "$CONFIG" \
    --work-dir "$OUT" --no-validate --seed ${SEED:-0} --deterministic "$@" 2>&1 \
  | tee "$OUT/train_full.log" \
  | grep --line-buffered -E "Epoch \[|Traceback|Error|Saving checkpoint" \
  || { echo "TRAINING FAILED - last lines of the log:"; tail -40 "$OUT/train_full.log"; exit 1; }
ls -la "$OUT"/*.pth

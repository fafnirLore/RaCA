#!/usr/bin/env bash
# Train one configuration on Kaggle (1 GPU). Usage: bash kaggle/train.sh <config> <run_name> [extra train.py args]
#   e.g. bash kaggle/train.sh kaggle/configs/raca_full.py raca_full --cfg-options runner.max_epochs=3
# Starting weights: the config's load_from (ckpts/warm_start.pth) is copied from an attached input or built from SGN-T when missing.
# Seed 0 always. DETERMINISTIC=1 adds --deterministic (as tools/dist_train.sh). Off by default: the runs known to fit a 16 GB T4
#   (11.7-11.8 GB logged) ran without it and its memory cost with the warm start is unmeasured.
# Output: /kaggle/working/runs/<run_name>/ (checkpoints, logs, train_full.log). Progress lines are printed live.
set -o pipefail
CONFIG=$1; RUN=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/tmp/mra_env/bin/python}
OUT=${OUT_ROOT:-/kaggle/working/runs}/$RUN; mkdir -p "$OUT"
DET=""; [ "${DETERMINISTIC:-0}" = "1" ] && DET="--deterministic"
LOAD_FROM=$($PY -c "import sys; from mmcv import Config; print(Config.fromfile(sys.argv[1]).get('load_from') or '')" "$CONFIG")
if [[ " $* " != *" load_from="* ]] && [ -n "$LOAD_FROM" ] && [ ! -s "$LOAD_FROM" ]; then bash kaggle/ensure_warm_start.sh "$LOAD_FROM" || exit 1; fi
echo "run $RUN | seed ${SEED:-0} $DET | load_from ${LOAD_FROM:-none}"
# PYTHONPATH: MRA's train.py imports its own 'projects' package (dist_train.sh does the same)
PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=0 timeout ${TIMEOUT:-11h} $PY tools/train.py "$CONFIG" \
    --work-dir "$OUT" --no-validate --seed ${SEED:-0} $DET "$@" 2>&1 \
  | tee "$OUT/train_full.log" \
  | grep --line-buffered -E "Epoch \[|Traceback|Error|Saving checkpoint|load checkpoint|Start running" \
  || { echo "TRAINING FAILED - last lines of the log:"; tail -40 "$OUT/train_full.log"; exit 1; }
ls -la "$OUT"/*.pth

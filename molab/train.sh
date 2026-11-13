#!/usr/bin/env bash
# Train one configuration on molab (1 GPU). Usage: bash molab/train.sh <config> <run_name> [extra train.py args]
#   e.g. bash molab/train.sh molab/configs/mra_before.py mra_before --cfg-options runner.max_epochs=3
# Same training command as kaggle/train.sh on one GPU (seed 0, validation after every epoch for the evaluation and
# early-stopping hooks, VALIDATE=0 passes --no-validate, DETERMINISTIC=1 adds --deterministic), plus:
#   - resumes from <run>/latest.pth when it exists (molab sessions end after 12 h; RESUME=0 starts again),
#   - BACKUP_TO=<folder or hf://datasets/user/repo/sub> copies the newest checkpoint and the logs there every
#     BACKUP_EVERY seconds (default 600) and once more at the end, because molab clears the machine at shutdown.
# Output: $OUT_ROOT/<run_name>/ (default /tmp/mra_runs/<run_name>/): checkpoints, logs, train_full.log.
set -o pipefail
CONFIG=$1; RUN=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source molab/env.sh
OUT=$OUT_ROOT/$RUN; mkdir -p "$OUT"
DET=""; [ "${DETERMINISTIC:-0}" = "1" ] && DET="--deterministic"
NOVAL=""; [ "${VALIDATE:-1}" = "0" ] && NOVAL="--no-validate"
LOAD_FROM=$($PY -c "import sys; from mmcv import Config; print(Config.fromfile(sys.argv[1]).get('load_from') or '')" "$CONFIG")
if [[ " $* " != *" load_from="* ]] && [ -n "$LOAD_FROM" ] && [ ! -s "$LOAD_FROM" ]; then bash kaggle/ensure_warm_start.sh "$LOAD_FROM" || exit 1; fi
RES=""
if [ "${RESUME:-1}" = "1" ] && [ -e "$OUT/latest.pth" ]; then RES="--resume-from $OUT/latest.pth"; echo "resuming from $(readlink -f "$OUT/latest.pth")"; fi
echo "run $RUN | seed ${SEED:-0} $DET | load_from ${LOAD_FROM:-none} | output $OUT"

BACKUP_PID=""
if [ -n "${BACKUP_TO:-}" ]; then
  ( while sleep "${BACKUP_EVERY:-600}"; do $PY molab/backup.py push "$OUT" "$BACKUP_TO" > "$OUT/backup.log" 2>&1; done ) &
  BACKUP_PID=$!
  trap '[ -n "$BACKUP_PID" ] && kill $BACKUP_PID 2>/dev/null' EXIT
fi

# PYTHONPATH: MRA's train.py imports its own 'projects' package (dist_train.sh does the same)
PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0} timeout ${TIMEOUT:-11h} $PY tools/train.py "$CONFIG" \
    --work-dir "$OUT" $NOVAL --seed ${SEED:-0} $DET $RES "$@" 2>&1 \
  | tee -a "$OUT/train_full.log" \
  | grep --line-buffered -E "Epoch \[|Traceback|Error|Saving checkpoint|load checkpoint|resumed epoch|Start running|EarlyStoppingHook|Epoch.val. \[|Now best"
STATUS=${PIPESTATUS[0]}
if [ -n "${BACKUP_TO:-}" ]; then
  kill $BACKUP_PID 2>/dev/null; BACKUP_PID=""
  $PY molab/backup.py push "$OUT" "$BACKUP_TO" 2>&1 | tail -2
fi
if [ "$STATUS" != "0" ]; then echo "TRAINING FAILED (exit $STATUS) - last lines of the log:"; tail -40 "$OUT/train_full.log"; exit 1; fi
ls -la "$OUT"/*.pth
[ -f "$OUT/early_stopping.json" ] && echo "early stopping record: $OUT/early_stopping.json"

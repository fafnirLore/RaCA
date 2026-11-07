#!/usr/bin/env bash
# Train one configuration on Kaggle (1 or 2 GPUs). Usage: bash kaggle/train.sh <config> <run_name> [extra train.py args]
#   e.g. bash kaggle/train.sh kaggle/configs/raca_full.py raca_full --cfg-options runner.max_epochs=3
# GPUs: GPUS=<n> or, by default, every visible GPU (nvidia-smi). With more than one GPU the run is launched with
#   torch.distributed.launch (--launcher pytorch, one process per GPU, batch 1 per GPU, as tools/dist_train.sh) and the
#   linear scaling rule is applied to the config: optimizer.lr x GPUS, lr_config.warmup_iters / GPUS,
#   data.workers_per_gpu = 2 (Kaggle's T4 x2 machine has 4 CPU cores). SCALE_LR=0 keeps the config values.
#   Your own --cfg-options are merged after these, so they win.
# --deterministic (cuDNN deterministic algorithms, no benchmark autotuning) matches tools/dist_train.sh.
# Validation runs after every epoch (needed by EarlyStoppingHook and for best_*.pth); VALIDATE=0 passes --no-validate.
# Output: /kaggle/working/runs/<run_name>/ (checkpoints, logs, train_full.log). Progress lines are printed live.
set -o pipefail
CONFIG=$1; RUN=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/tmp/mra_env/bin/python}
OUT=${OUT_ROOT:-/kaggle/working/runs}/$RUN; mkdir -p "$OUT"
NOVAL=""; [ "${VALIDATE:-1}" = "0" ] && NOVAL="--no-validate"
GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}; [ "$GPUS" -ge 1 ] 2>/dev/null || GPUS=1

# Split the caller's "--cfg-options k=v ..." out of the extra args: mmcv's DictAction keeps only the LAST --cfg-options
# given, so everything has to be merged into one. Other extra args pass through unchanged.
USER_OPTS=(); OTHER=(); mode=other
for tok in "$@"; do
  if [ "$tok" = "--cfg-options" ]; then mode=opts; continue; fi
  if [ "$mode" = "opts" ] && [[ "$tok" == --* ]]; then mode=other; fi
  if [ "$mode" = "opts" ]; then USER_OPTS+=("$tok"); else OTHER+=("$tok"); fi
done
AUTO_OPTS=()
if [ "$GPUS" -gt 1 ] && [ "${SCALE_LR:-1}" != "0" ]; then
  read -r LR WARM < <($PY - "$CONFIG" "$GPUS" <<'EOF'
import sys; from mmcv import Config
c = Config.fromfile(sys.argv[1]); g = int(sys.argv[2])
print(c.optimizer.lr * g, max(1, int(c.lr_config.get('warmup_iters', 500)) // g))
EOF
)
  AUTO_OPTS=("optimizer.lr=$LR" "lr_config.warmup_iters=$WARM" "data.workers_per_gpu=2")
  echo "GPUS=$GPUS -> linear scaling: ${AUTO_OPTS[*]}"
fi
CFG_ARGS=(); if [ ${#AUTO_OPTS[@]} -gt 0 ] || [ ${#USER_OPTS[@]} -gt 0 ]; then CFG_ARGS=(--cfg-options "${AUTO_OPTS[@]}" "${USER_OPTS[@]}"); fi

if [ "$GPUS" -gt 1 ]; then
  LAUNCH="$PY -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=${PORT:-28707} tools/train.py $CONFIG --launcher pytorch"
  DEVICES=$(seq -s, 0 $((GPUS-1)))
else
  LAUNCH="$PY tools/train.py $CONFIG"
  DEVICES=0
fi
echo "run $RUN | $GPUS GPU(s) | $LAUNCH ${CFG_ARGS[*]} ${OTHER[*]}"
# PYTHONPATH: MRA's train.py imports its own 'projects' package (dist_train.sh does the same)
PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=$DEVICES timeout ${TIMEOUT:-11h} $LAUNCH \
    --work-dir "$OUT" $NOVAL --seed ${SEED:-0} --deterministic "${CFG_ARGS[@]}" "${OTHER[@]}" 2>&1 \
  | tee "$OUT/train_full.log" \
  | grep --line-buffered -E "Epoch \[|Traceback|Error|Saving checkpoint|EarlyStoppingHook|semkitti_SSC_mIoU|Now best|linear scaling" \
  || { echo "TRAINING FAILED - last lines of the log:"; tail -40 "$OUT/train_full.log"; exit 1; }
ls -la "$OUT"/*.pth
[ -f "$OUT/early_stopping.json" ] && echo "early stopping record: $OUT/early_stopping.json" && $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print('best', d['monitor'], d['best'], 'at epoch', d['best_epoch'], '| stopped early:', d['stopped_early'], '| last epoch:', d['last_epoch'])" "$OUT/early_stopping.json"

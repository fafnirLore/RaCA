#!/usr/bin/env bash
# Make sure ckpts/warm_start.pth (the starting weights shared by every Kaggle run) exists. Called by the notebook,
# kaggle/smoke_test.py and kaggle/train.sh, so a run cannot start without it. Order:
#   1. already there                        -> nothing to do
#   2. attached Kaggle input (**/ckpts/warm_start.pth, e.g. the "before" run's output) -> copied, so runs share one file
#   3. otherwise built from SGN-T           -> kaggle/get_weights.sh sgn + kaggle/make_warm_start.py (~1 min)
# Usage: bash kaggle/ensure_warm_start.sh [ckpts/warm_start.pth]
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p ckpts
OUT=${1:-ckpts/warm_start.pth}
PY=${PY:-/tmp/mra_env/bin/python}
if [ -s "$OUT" ]; then echo "starting weights present: $OUT ($(du -h "$OUT" | cut -f1))"; exit 0; fi
SRC=$(find /kaggle/input -path '*ckpts/warm_start.pth' 2>/dev/null | head -1)
if [ -n "$SRC" ]; then cp "$SRC" "$OUT" && echo "starting weights copied from $SRC"; exit $?; fi
echo "building starting weights from SGN-T"
bash kaggle/get_weights.sh sgn || exit 1
PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} $PY kaggle/make_warm_start.py "$OUT" > ckpts/make_warm_start.log 2>&1 \
  || { echo "FAILED: make_warm_start.py - last lines of ckpts/make_warm_start.log:"; tail -30 ckpts/make_warm_start.log; rm -f "$OUT"; exit 1; }
tail -3 ckpts/make_warm_start.log
[ -s "$OUT" ] || { echo "FAILED: $OUT was not written"; exit 1; }

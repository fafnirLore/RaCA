#!/usr/bin/env bash
# Evaluate a checkpoint on the validation frames. Usage: bash kaggle/eval.sh <config> <checkpoint> <out.json> "<label>"
set -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY=${PY:-/tmp/mra_env/bin/python}
F='mmcv - INFO|is the same before|torch.Size|PretrainedInit|XavierInit|WARNING!!!!|UserWarning|floor_divide|return torch|^ *$'
PYTHONPATH=$ROOT CUDA_VISIBLE_DEVICES=0 $PY kaggle/evaluate.py "$1" "$2" "$3" --name "$4" 2>&1 | grep -v -E "$F"

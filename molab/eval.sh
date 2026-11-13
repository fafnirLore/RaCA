#!/usr/bin/env bash
# Evaluate a checkpoint on the validation frames (kaggle/evaluate.py, unchanged) with the molab environment.
# Usage: bash molab/eval.sh <config> <checkpoint> <out.json> "<label>"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source molab/env.sh
exec bash kaggle/eval.sh "$@"

#!/usr/bin/env bash
# Download the starting weights with the molab environment and build ckpts/warm_start.pth (kaggle scripts, unchanged).
# Usage: bash molab/get_weights.sh            ResNet-50 + SGN-T + warm start (what molab/configs/mra_before.py needs)
#        bash molab/get_weights.sh resnet     ResNet-50 only (enough for --cfg-options load_from=None)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source molab/env.sh
command -v wget >/dev/null 2>&1 || { mkdir -p ckpts; [ -s ckpts/resnet50-19c8e357.pth ] || curl -fsSL -o ckpts/resnet50-19c8e357.pth https://download.pytorch.org/models/resnet50-19c8e357.pth; }
if [ "${1:-}" = "resnet" ]; then bash kaggle/get_weights.sh; exit $?; fi
bash kaggle/get_weights.sh sgn && bash kaggle/ensure_warm_start.sh

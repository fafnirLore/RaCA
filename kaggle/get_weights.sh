#!/usr/bin/env bash
# Download the ImageNet ResNet-50 weights MRA starts from (MRA-T's `pretrained`). With the argument `sgn` also
# download SGN-T's SemanticKITTI weights, only needed for the optional warm start (kaggle/make_warm_start.py).
# Writes to <repo>/ckpts/. Usage: bash kaggle/get_weights.sh [sgn]
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; mkdir -p ckpts
GDOWN=${GDOWN:-/tmp/mra_env/bin/gdown}
[ -f ckpts/resnet50-19c8e357.pth ] || wget -q https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpts/resnet50-19c8e357.pth
ls -la ckpts/resnet50-19c8e357.pth && echo "OK: ResNet-50 weights"
[ "${1:-}" = "sgn" ] || exit 0
# SGN-T, SemanticKITTI, from SGN's model zoo (Google Drive)
[ -f ckpts/sgn-T.pth ] || $GDOWN -q 1cDKX6SpyoAHr-cfmjTRDsNF-3_JgtH3p -O ckpts/sgn-T.pth
ls -la ckpts/sgn-T.pth && echo "OK: SGN-T weights" || { echo "FAILED: SGN-T download (Google Drive rate limit?) - retry later"; exit 1; }

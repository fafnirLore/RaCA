#!/usr/bin/env bash
# Install MRA's software stack on a Kaggle GPU notebook (Tesla T4, internet on). About 15 minutes.
# Creates a Python 3.8 environment at /tmp/mra_env with: torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0,
# mmsegmentation 0.14.1, spconv-cu113 2.3.6, torch-scatter 2.0.8, torchmetrics 0.11.4, timm 0.6.13, gdown and
# mmdet3d v0.17.1 installed WITHOUT compiling its CUDA operators (MRA does not use them).
# Idempotent: exits at once if the environment already works. Usage: bash kaggle/install.sh
set -o pipefail
PY=/tmp/mra_env/bin/python
PIP="$PY -m pip"
HERE="$(cd "$(dirname "$0")" && pwd)"

if $PY -c "import mmdet3d, spconv, torch_scatter" 2>/dev/null; then echo "Already installed - skipping"; exit 0; fi

echo "[1/5] micromamba + Python 3.8 environment"
cd /tmp
[ -x /tmp/bin/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
[ -x $PY ] || /tmp/bin/micromamba create -y -q -p /tmp/mra_env -c conda-forge python=3.8 pip > /tmp/mamba.log 2>&1 \
  || { tail -40 /tmp/mamba.log; echo "FAILED (micromamba)"; exit 1; }
$PY --version

echo "[2/5] pip, version pins"
cat > /tmp/constraints.txt <<'EOF'
numpy==1.19.5
setuptools==59.5.0
yapf==0.40.1
torch==1.9.1+cu111
torchvision==0.10.1+cu111
EOF
$PIP install -q --upgrade "pip<24" > /tmp/pip0.log 2>&1 && $PIP install -q setuptools==59.5.0 wheel ninja >> /tmp/pip0.log 2>&1 \
  || { tail -30 /tmp/pip0.log; echo "FAILED (pip)"; exit 1; }

echo "[3/5] PyTorch 1.9.1 for CUDA 11.1 (about 2 GB)"
$PIP install -q -c /tmp/constraints.txt torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html > /tmp/pip1.log 2>&1 || { tail -30 /tmp/pip1.log; echo "FAILED (torch)"; exit 1; }
$PY -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))" || exit 1

echo "[4/5] mmcv-full / mmdet / mmseg / spconv / torch-scatter / torchmetrics / timm / gdown"
# spconv 2.3.6 instead of MRA's 2.1.25: 2.1.25's GPU sort crashes on real-size inputs; same weight layout.
$PIP install -q -c /tmp/constraints.txt mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html > /tmp/pip2.log 2>&1 \
  && $PIP install -q -c /tmp/constraints.txt mmdet==2.14.0 mmsegmentation==0.14.1 >> /tmp/pip2.log 2>&1 \
  && $PIP install -q -c /tmp/constraints.txt spconv-cu113==2.3.6 >> /tmp/pip2.log 2>&1 \
  && $PIP install -q -c /tmp/constraints.txt torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.9.0+cu111.html >> /tmp/pip2.log 2>&1 \
  && $PIP install -q -c /tmp/constraints.txt torchmetrics==0.11.4 timm==0.6.13 gdown >> /tmp/pip2.log 2>&1 \
  || { tail -40 /tmp/pip2.log; echo "FAILED (mm packages)"; exit 1; }

echo "[5/5] mmdet3d v0.17.1 from source, CUDA operators replaced by placeholders"
cd /tmp && rm -rf mmdetection3d
git clone -q https://github.com/open-mmlab/mmdetection3d.git && cd mmdetection3d && git checkout -q v0.17.1
$PY "$HERE/patch_mmdet3d.py" /tmp/mmdetection3d | tail -1
sed -i 's/from numba.errors import/from numba.core.errors import/' mmdet3d/datasets/pipelines/data_augment_utils.py
$PIP install -q -c /tmp/constraints.txt numba==0.53.0 networkx==2.2 plyfile trimesh==2.35.39 scikit-image==0.19.3 \
  nuscenes-devkit==1.1.9 lyft_dataset_sdk==0.0.8 tensorboard==2.11.2 > /tmp/pip3.log 2>&1 \
  || { tail -40 /tmp/pip3.log; echo "FAILED (mmdet3d dependencies)"; exit 1; }
$PIP install -v -e . --no-deps --no-build-isolation > /tmp/mmdet3d_build.log 2>&1 \
  || { tail -60 /tmp/mmdet3d_build.log; echo "FAILED (mmdet3d build)"; exit 1; }
cd /tmp && $PY -c "import mmdet3d, mmdet3d.models, mmdet3d.datasets; print('mmdet3d', mmdet3d.__version__)" && echo "OK: environment ready at /tmp/mra_env"

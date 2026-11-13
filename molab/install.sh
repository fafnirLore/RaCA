#!/usr/bin/env bash
# Install MRA's software stack for molab's NVIDIA RTX PRO 6000 Blackwell (96 GB). About 5-10 minutes.
#
# Why this differs from kaggle/install.sh: Blackwell GPUs (compute capability 12.0, sm_120) only run CUDA code built
# with CUDA >= 12.8, i.e. PyTorch >= 2.7. The Kaggle stack (torch 1.9.1+cu111, mmcv-full 1.4.0, spconv-cu113) fails
# there with "no kernel image is available for execution on the device". So this script builds a Python 3.10
# environment at /tmp/mra_env with:
#   torch 2.7.1+cu128 / torchvision 0.22.1+cu128, numpy 1.23.5
#   mmcv 1.7.2 (the "lite" package; its compiled operators are replaced by placeholders, MRA uses none of them),
#   mmdet 2.14.0, mmsegmentation 0.14.1, mmdet3d v0.17.1 (CUDA operators replaced by placeholders, as on Kaggle)
#   spconv 2.3.8 built for CUDA 12.8 with sm_120 kernels (community wheels, checksums pinned below;
#     falls back to upstream spconv-cu126, which reaches Blackwell through the driver's PTX JIT)
#   torch-scatter 2.1.2 (PyG wheel for torch 2.7 + cu128; falls back to molab/shims/torch_scatter, plain PyTorch)
#   torchmetrics 0.11.4, timm 0.6.13, gdown
# Nothing is compiled, so no gcc / CUDA toolkit is needed on the machine.
# Idempotent: exits at once if the environment already works. Usage: bash molab/install.sh
#   MRA_ENV=/path   environment location (default /tmp/mra_env)
#   CUDA_TAG=cpu    CPU-only build, used to test this script without a GPU
set -o pipefail
ENV=${MRA_ENV:-/tmp/mra_env}
PY=$ENV/bin/python
PIP="$PY -m pip"
HERE="$(cd "$(dirname "$0")" && pwd)"
CUDA_TAG=${CUDA_TAG:-cu128}
TORCH=2.7.1; TVISION=0.22.1
LOG=${MRA_LOG_DIR:-/tmp/mra_install_logs}; mkdir -p "$LOG"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore

if [ -f "$ENV/.mra_ok" ] && $PY -c "import torch, mmdet3d, spconv, torch_scatter" 2>/dev/null; then
  echo "Already installed - skipping ($(cat "$ENV/.mra_ok"))"; exit 0; fi

echo "[1/6] Python 3.10 environment at $ENV"
if [ ! -x "$PY" ]; then
  if command -v uv >/dev/null 2>&1 && uv venv -q --seed --python 3.10 "$ENV" > "$LOG/env.log" 2>&1; then echo "  created with uv"
  else
    rm -rf "$ENV"; mkdir -p /tmp/bin
    if [ ! -x /tmp/bin/micromamba ]; then
      case "$(uname -m)" in aarch64|arm64) MM=linux-aarch64;; *) MM=linux-64;; esac
      (cd /tmp && curl -Ls "https://micro.mamba.pm/api/micromamba/$MM/latest" | tar -xj bin/micromamba) >> "$LOG/env.log" 2>&1
    fi
    if [ -x /tmp/bin/micromamba ] && /tmp/bin/micromamba create -y -q -p "$ENV" -c conda-forge python=3.10 pip >> "$LOG/env.log" 2>&1; then echo "  created with micromamba"
    elif command -v python3.10 >/dev/null 2>&1 && python3.10 -m venv "$ENV" >> "$LOG/env.log" 2>&1; then echo "  created with python3.10 -m venv"
    else tail -40 "$LOG/env.log"; echo "FAILED (no way to create a Python 3.10 environment: tried uv, micromamba, python3.10)"; exit 1; fi
  fi
fi
$PY --version || exit 1

echo "[2/6] pip, version pins"
if [ "$CUDA_TAG" = "cpu" ]; then TORCH_PIN="torch==$TORCH+cpu"; TV_PIN="torchvision==$TVISION+cpu"
else TORCH_PIN="torch==$TORCH+$CUDA_TAG"; TV_PIN="torchvision==$TVISION+$CUDA_TAG"; fi
C=/tmp/mra_constraints.txt
cat > $C <<EOF
numpy==1.23.5
setuptools==59.5.0
yapf==0.40.1
opencv-python==4.8.1.78
opencv-python-headless==4.8.1.78
scipy==1.10.1
matplotlib==3.5.3
pandas==1.5.3
Shapely==1.8.5.post1
$TORCH_PIN
$TV_PIN
EOF
$PIP install -q --upgrade "pip<25" > "$LOG/pip0.log" 2>&1 && $PIP install -q setuptools==59.5.0 wheel >> "$LOG/pip0.log" 2>&1 \
  || { tail -30 "$LOG/pip0.log"; echo "FAILED (pip)"; exit 1; }

echo "[3/6] PyTorch $TORCH for $CUDA_TAG"
$PIP install -q -c $C numpy==1.23.5 > "$LOG/pip1.log" 2>&1 \
  && $PIP install -q -c $C --only-binary :all: torch==$TORCH torchvision==$TVISION \
       --extra-index-url https://download.pytorch.org/whl/$CUDA_TAG >> "$LOG/pip1.log" 2>&1 \
  || { tail -30 "$LOG/pip1.log"; echo "FAILED (torch)"; exit 1; }
if [ "$CUDA_TAG" = "cpu" ]; then $PY -c "import torch; print('torch', torch.__version__, '| CPU build (test mode)')" || exit 1
else
  $PY - <<'EOF' || { echo "FAILED (torch cannot use the GPU - is a GPU attached to the notebook?)"; exit 1; }
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
name = torch.cuda.get_device_name(0); cap = torch.cuda.get_device_capability(0)
x = torch.randn(256, 256, device="cuda"); y = (x @ x).sum().item()          # a real kernel launch on this GPU
conv = torch.nn.Conv2d(3, 8, 3).cuda()(torch.randn(1, 3, 32, 32, device="cuda")).sum().item()   # cuDNN
print(f"torch {torch.__version__} | {name} | sm_{cap[0]}{cap[1]} | {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB | kernels OK")
EOF
fi

echo "[4/6] mmcv / mmdet / mmseg / torchmetrics / timm / gdown"
# mmcv 1.7.2 is the last 1.x release and the one that knows PyTorch 2; mmdet 2.14 / mmseg 0.14 / mmdet3d 0.17 only
# refuse it through a version assert, relaxed in step 6. 'mmcv' (not mmcv-full) = pure Python, nothing to compile.
# opencv headless first: mmcv then keeps it instead of pulling opencv-python, which needs the system library libGL.
$PIP install -q -c $C opencv-python-headless > "$LOG/pip2.log" 2>&1 && $PIP install -q -c $C mmcv==1.7.2 >> "$LOG/pip2.log" 2>&1 \
  && $PIP install -q -c $C mmdet==2.14.0 mmsegmentation==0.14.1 >> "$LOG/pip2.log" 2>&1 \
  && $PIP install -q -c $C torchmetrics==0.11.4 timm==0.6.13 gdown >> "$LOG/pip2.log" 2>&1 \
  || { tail -40 "$LOG/pip2.log"; echo "FAILED (mm packages)"; exit 1; }

echo "[5/6] spconv + torch-scatter"
REL=https://github.com/fafraob/spconv-wheels/releases/download/v2.3.8
CUMM_WHL=cumm_cu128-0.8.2-cp310-cp310-linux_x86_64.whl;   CUMM_SHA=75a7b9645564f602ce8cf6bb4de25c97a9c7874b82a35a65316cd5493cce14b5
SPCONV_WHL=spconv_cu128-2.3.8-cp310-cp310-linux_x86_64.whl; SPCONV_SHA=edea898414cf46606f7795a01e07dfd77343ca8237a8fefabcf1dfd420f63883
spconv_gpu_ok() {   # a real sparse convolution on the GPU, in a fresh process
  $PY - <<'EOF' 2>> "$LOG/spconv_check.log"
import torch, spconv.pytorch as spconv
idx = torch.tensor([[0, 1, 1, 1], [0, 2, 3, 1], [0, 5, 5, 2], [0, 6, 5, 2]], dtype=torch.int32, device="cuda")
x = spconv.SparseConvTensor(torch.randn(4, 8, device="cuda"), idx, spatial_shape=[8, 8, 4], batch_size=1)
net = spconv.SparseSequential(spconv.SubMConv3d(8, 16, 3, indice_key="a", bias=False), spconv.SparseConv3d(16, 16, 3, stride=2, padding=1)).cuda()
out = net(x); out.features.sum().backward(); assert out.features.shape[1] == 16 and torch.isfinite(out.features).all()
EOF
}
SPCONV_KIND=""
if [ "$CUDA_TAG" = "cpu" ]; then
  $PIP install -q -c $C spconv==2.3.6 > "$LOG/pip3.log" 2>&1 || { tail -30 "$LOG/pip3.log"; echo "FAILED (spconv cpu)"; exit 1; }
  SPCONV_KIND="spconv 2.3.6 (cpu)"
else
  if [ "$CUDA_TAG" = "cu128" ] && mkdir -p /tmp/mra_wheels && cd /tmp/mra_wheels \
     && curl -fsSL -o $CUMM_WHL $REL/$CUMM_WHL && curl -fsSL -o $SPCONV_WHL $REL/$SPCONV_WHL \
     && echo "$CUMM_SHA  $CUMM_WHL" | sha256sum -c - > "$LOG/pip3.log" 2>&1 && echo "$SPCONV_SHA  $SPCONV_WHL" | sha256sum -c - >> "$LOG/pip3.log" 2>&1 \
     && $PIP install -q -c $C ./$CUMM_WHL ./$SPCONV_WHL >> "$LOG/pip3.log" 2>&1 && spconv_gpu_ok; then
    SPCONV_KIND="spconv 2.3.8 cu128, native sm_120 kernels"
  else
    echo "  native-Blackwell spconv wheels unavailable or failed their GPU check - falling back to upstream spconv-cu126 (PTX JIT)"
    $PIP uninstall -q -y spconv-cu128 cumm-cu128 >> "$LOG/pip3.log" 2>&1
    $PIP install -q -c $C spconv-cu126 >> "$LOG/pip3.log" 2>&1 && spconv_gpu_ok \
      || { tail -30 "$LOG/pip3.log"; tail -30 "$LOG/spconv_check.log"; echo "FAILED (spconv: no build runs a sparse convolution on this GPU)"; exit 1; }
    SPCONV_KIND="spconv-cu126 (PTX JIT on Blackwell)"
  fi
fi
cd /tmp
echo "  $SPCONV_KIND"

SITE=$($PY -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
scatter_ok() {
  $PY - <<'EOF' 2>> "$LOG/scatter_check.log"
import torch, torch_scatter
dev = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.tensor([[1., 2.], [3., 4.], [5., 0.]], device=dev, requires_grad=True); i = torch.tensor([0, 0, 2], device=dev)
m = torch_scatter.scatter_mean(x, i, dim=0); mx = torch_scatter.scatter_max(x, i, dim=0)[0]; s = torch_scatter.scatter_add(x, i, dim=0)
assert m.tolist() == [[2., 3.], [0., 0.], [5., 0.]] and mx.tolist() == [[3., 4.], [0., 0.], [5., 0.]] and s.tolist() == [[4., 6.], [0., 0.], [5., 0.]]
(m.sum() + mx.sum() + s.sum()).backward()
EOF
}
rm -f "$SITE/mra_scatter_shim.pth"
if $PIP install -q -c $C torch-scatter==2.1.2 --only-binary :all: -f https://data.pyg.org/whl/torch-2.7.0+$CUDA_TAG.html > "$LOG/pip4.log" 2>&1 && scatter_ok; then
  echo "  torch-scatter 2.1.2 (PyG wheel)"
else
  echo "  torch-scatter wheel unavailable or failed its check - using molab/shims/torch_scatter (plain PyTorch, same results)"
  $PIP uninstall -q -y torch-scatter >> "$LOG/pip4.log" 2>&1
  echo "$HERE/shims" > "$SITE/mra_scatter_shim.pth"
  scatter_ok || { tail -30 "$LOG/scatter_check.log"; echo "FAILED (torch_scatter shim)"; exit 1; }
fi

echo "[6/6] mmdet3d v0.17.1 from source (operators replaced by placeholders) + compatibility patches"
rm -rf /tmp/mmdetection3d
git clone -q --depth 1 -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdetection3d > "$LOG/mmdet3d.log" 2>&1 \
  || { tail -20 "$LOG/mmdet3d.log"; echo "FAILED (clone mmdetection3d)"; exit 1; }
cd /tmp/mmdetection3d
$PY "$HERE/../kaggle/patch_mmdet3d.py" /tmp/mmdetection3d | tail -1
sed -i 's/from numba.errors import/from numba.core.errors import/' mmdet3d/datasets/pipelines/data_augment_utils.py
$PIP install -q -c $C numba==0.56.4 networkx==2.8.8 plyfile trimesh==3.23.5 scikit-image==0.19.3 scikit-learn==1.2.2 \
    pyquaternion cachetools descartes fire tqdm tensorboard==2.11.2 "protobuf<4" > "$LOG/pip5.log" 2>&1 \
  && $PIP install -q -c $C --no-deps nuscenes-devkit==1.1.9 lyft_dataset_sdk==0.0.8 >> "$LOG/pip5.log" 2>&1 \
  || { tail -40 "$LOG/pip5.log"; echo "FAILED (mmdet3d dependencies)"; exit 1; }
$PIP install -q -e . --no-deps --no-build-isolation > "$LOG/mmdet3d_build.log" 2>&1 \
  || { tail -60 "$LOG/mmdet3d_build.log"; echo "FAILED (mmdet3d install)"; exit 1; }
$PY "$HERE/patch_stack.py" /tmp/mmdetection3d || { echo "FAILED (patch_stack.py)"; exit 1; }
cd /tmp && TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 $PY -W ignore -c "
import torch, mmcv, mmdet, mmseg, mmdet3d, mmdet3d.models, mmdet3d.datasets, spconv, torch_scatter, numpy
print('numpy', numpy.__version__, '| mmcv', mmcv.__version__, '| mmdet', mmdet.__version__, '| mmseg', mmseg.__version__, '| mmdet3d', mmdet3d.__version__)" \
  || { echo "FAILED (import check)"; exit 1; }
echo "torch $TORCH+$CUDA_TAG | $SPCONV_KIND" > "$ENV/.mra_ok"
echo "OK: environment ready at $ENV"

# Running the experiments on Kaggle (RaCA)

Everything needed to reproduce the small-scale runs on a free Kaggle GPU (one Tesla T4, 16 GB) lives in this folder.
The notebook `kaggle/kaggle_raca.ipynb` strings the steps together; each step is also a script that can be run by hand.

## Datasets to attach (right-hand panel → *Add Input*)

| Dataset | Where it comes from | What this folder reads from it |
|---|---|---|
| **scenes** (SemanticKITTI subset, 400 train / 150 val frames) | Output of the step-2 "building-subset" notebook (`daniyalimran/building-subset`), saved compactly to fit Kaggle's 20 GB output limit | `**/kitti_subset/selection.json` (summary of the split) and the `dataset/` tree next to it |
| **mra-before-run** (output of the original-MRA run) | Output of the "before" notebook (`daniyalimran/mra-before-run`) | `**/runs/before/results_trained.json` only (for `compare.py`); its `warm_start.pth` is no longer used |

The compact scenes layout and how it is expanded back to MRA's format are documented in `expand_subset.py`:
labels `labels/XX/<frame>_1_1.npy` are stored as uint8 and rewritten as float32 (values 0–19, 255), point clouds
`sequences_msnet3d_lidar/sequences/XX/<frame>.bin` are stored as x,y,z and rewritten as x,y,z,1 (as MRA's `depth2lidar.py`
writes them), and images, voxels, calib, poses and `*_1_2.npy` are symlinked. Expansion writes to `/tmp/kitti_full/`
(about 3 minutes).

Weights downloaded by `get_weights.sh`: ImageNet ResNet-50 (`ckpts/resnet50-19c8e357.pth`, PyTorch model zoo), the
only pretrained weights MRA-T uses. `bash kaggle/get_weights.sh sgn` additionally fetches SGN-T SemanticKITTI
(`ckpts/sgn-T.pth`, SGN model zoo on Google Drive, id `1cDKX6SpyoAHr-cfmjTRDsNF-3_JgtH3p`) for the optional warm start.

## Steps and commands (run from the repository root)

| Step | Command | Notes |
|---|---|---|
| Install the software | `bash kaggle/install.sh` | ~15 min; Python 3.8 env at `/tmp/mra_env`, torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0, mmseg 0.14.1, spconv-cu113 2.3.6, torch-scatter 2.0.8, mmdet3d v0.17.1 without compiled CUDA ops (`patch_mmdet3d.py`) |
| Download weights | `bash kaggle/get_weights.sh` | ResNet-50 (add `sgn` for the optional SGN-T warm start) |
| Rebuild the data | `python kaggle/expand_subset.py <scenes>/kitti_subset/ /tmp/kitti_full/` | both paths end with `/` |
| Starting weights | nothing to do | MRA's own initialisation: ImageNet ResNet-50 for the backbone, default init elsewhere, identical for every config. Optional SGN-T warm start: `python kaggle/make_warm_start.py ckpts/warm_start.pth` and train with `--cfg-options load_from=ckpts/warm_start.pth` |
| Smoke test | `python kaggle/smoke_test.py kaggle/configs/raca_full.py --steps 2 --expect-head RaCAHead --expect-keys loss_align_directed,loss_scan` | 2 real steps, prints loss keys and peak memory |
| Train | `bash kaggle/train.sh kaggle/configs/<run>.py <run>` | one GPU, seed 0, `--deterministic`, `--no-validate`; output in `/kaggle/working/runs/<run>/`; add `--cfg-options runner.max_epochs=3` etc. to override the config |
| Evaluate | `bash kaggle/eval.sh kaggle/configs/<run>.py /kaggle/working/runs/<run>/latest.pth out.json "label"` | MRA's `SSCMetrics` + four disjoint 12.8 m range bands, per-class IoU with GT voxel counts, pooled VRU IoU, wIoU |
| Compare | `python kaggle/compare.py before.json after.json` | before/after table: overall, rare classes, per band, all classes |

Scripts use `/tmp/mra_env/bin/python` by default (`PY=... bash kaggle/train.sh ...` to change). `train.sh` honours `SEED`, `TIMEOUT` (default 11h) and `OUT_ROOT`.

## Configurations

All Kaggle configs inherit a repository config and change only the data root (`/tmp/kitti_full/`), the schedule (6 epochs,
learning rate 5e-5 = 2e-4 / 4 for batch 1 on one GPU) and checkpoint/log settings. Initial weights are not changed:
every run starts from MRA-T's own initialisation (`pretrained` ImageNet ResNet-50, default init for the rest).

| Config | Trains |
|---|---|
| `kaggle/configs/mra_before.py` | original MRA-T (the 'before' run) |
| `kaggle/configs/raca_c2c3c4.py` | RaCA, training-only components C2 + C3 + C4 (the 'after' run) |
| `kaggle/configs/raca_full.py` | RaCA, full method C1 + C2 + C3 + C4 |
| `kaggle/configs/raca_c1.py` | RaCA, C1 depth-axis scan only (both grids) |
| `kaggle/configs/raca_c1_fine.py` | RaCA, C1 on the fine grid only |
| `kaggle/configs/raca_c2.py` | RaCA, C2 only |
| `kaggle/configs/raca_c2c3.py` | RaCA, C2 + C3 |
| `kaggle/configs/raca_c4.py` | RaCA, C4 only |

## Initial weights and determinism

- **Initial weights.** The configs set no `load_from`; the model is built and initialised exactly as MRA-T builds it
  (ImageNet ResNet-50 for the image backbone, PyTorch / mmcv default init for everything else). All configs therefore
  start from the same point for a given seed. The SGN-T warm start used in the first preliminary runs remains available
  as an opt-in (`make_warm_start.py`, `--cfg-options load_from=ckpts/warm_start.pth`).
- **Determinism.** `train.sh` passes `--seed 0 --deterministic`, as `tools/dist_train.sh` does. In mmdet 2.14 this calls
  `set_random_seed(seed, deterministic=True)`: Python, NumPy and torch seeds, `cudnn.deterministic = True`,
  `cudnn.benchmark = False`. It removes cuDNN algorithm selection as a source of run-to-run variation. It does not make
  the run bit-exact: spconv and torch-scatter kernels use atomic adds whose summation order is not fixed. Repeats with
  the same seed should agree closely but not necessarily to the last digit; `SEED=1 bash kaggle/train.sh ...` changes the seed.

## Observed on a T4 (400 training frames, batch 1)

| Item | Time / memory |
|---|---|
| install.sh | ~15 min |
| expand_subset.py | ~3 min |
| MRA-T epoch | ~40 min, 11.7 GB (measured with the earlier warm start and without `--deterministic`; cuDNN deterministic mode can be slower) |
| RaCA C2+C3+C4 epoch | ~53 min, 11.7 GB |
| evaluate.py (150 frames) | ~5 min |
| RaCA with C1 | not yet run; expect roughly +0.5–1 GB and a longer epoch |

Kaggle sessions are limited to 12 h, so one 6-epoch run fits per session; use *Save & Run All (Commit)* to run in the background.


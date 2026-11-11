# Running the baseline on Kaggle (original MRA, branch `mra_Og`)

This branch is the unmodified MRA code (upstream commit `d6b560f`) plus this `kaggle/` folder. It contains no RaCA code:
the RaCA head, utilities and configs live on branch `farquleet_add_module` of the same fork.

Everything needed to reproduce the small-scale runs on a free Kaggle GPU (one Tesla T4, 16 GB) lives in this folder.
The notebook `kaggle/kaggle_before.ipynb` strings the steps together; each step is also a script that can be run by hand.

## Datasets to attach (right-hand panel → *Add Input*)

| Dataset | Where it comes from | What this folder reads from it |
|---|---|---|
| **scenes** (SemanticKITTI subset, 400 train / 150 val frames) | Output of the step-2 "building-subset" notebook (`daniyalimran/building-subset`), saved compactly to fit Kaggle's 20 GB output limit | `**/kitti_subset/selection.json` (summary of the split) and the `dataset/` tree next to it |
| **mra-before-run** (output of the original-MRA run) | Output of the "before" notebook (`daniyalimran/mra-before-run`) | `**/ckpts/warm_start.pth` (shared starting weights, copied when attached; otherwise rebuilt from SGN-T) and `**/runs/before/results_trained.json` (for `compare.py`) |

The compact scenes layout and how it is expanded back to MRA's format are documented in `expand_subset.py`:
labels `labels/XX/<frame>_1_1.npy` are stored as uint8 and rewritten as float32 (values 0–19, 255), point clouds
`sequences_msnet3d_lidar/sequences/XX/<frame>.bin` are stored as x,y,z and rewritten as x,y,z,1 (as MRA's `depth2lidar.py`
writes them), and images, voxels, calib, poses and `*_1_2.npy` are symlinked. Expansion writes to `/tmp/kitti_full/`
(about 3 minutes).

Weights downloaded by `get_weights.sh`: ImageNet ResNet-50 (`ckpts/resnet50-19c8e357.pth`, PyTorch model zoo), the
only pretrained weights MRA-T itself uses. `bash kaggle/get_weights.sh sgn` additionally fetches SGN-T SemanticKITTI
(`ckpts/sgn-T.pth`, SGN model zoo on Google Drive, id `1cDKX6SpyoAHr-cfmjTRDsNF-3_JgtH3p`) for the shared warm start.

## Steps and commands (run from the repository root)

| Step | Command | Notes |
|---|---|---|
| Install the software | `bash kaggle/install.sh` | ~15 min; Python 3.8 env at `/tmp/mra_env`, torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0, mmseg 0.14.1, spconv-cu113 2.3.6, torch-scatter 2.0.8, mmdet3d v0.17.1 without compiled CUDA ops (`patch_mmdet3d.py`) |
| Download weights | `bash kaggle/get_weights.sh` | ResNet-50; with `sgn` also SGN-T for the warm start (the notebook uses `sgn`) |
| Rebuild the data | `python kaggle/expand_subset.py <scenes>/kitti_subset/ /tmp/kitti_full/` | both paths end with `/` |
| Starting weights | `bash kaggle/ensure_warm_start.sh` (also run by `train.sh` when the file is missing) | copies an attached `**/ckpts/warm_start.pth`, else runs `get_weights.sh sgn` + `make_warm_start.py`: SGN-T weights + SGN's main branch copied into MRA's coarse branch; 15 tensors keep their initial values. Required on a T4, see below |
| Train | `bash kaggle/train.sh kaggle/configs/<run>.py <run>` | one GPU, seed 0, `--no-validate` (`DETERMINISTIC=1` adds `--deterministic`); output in `/kaggle/working/runs/<run>/`; add `--cfg-options runner.max_epochs=3` etc. to override the config |
| Evaluate | `bash kaggle/eval.sh kaggle/configs/<run>.py /kaggle/working/runs/<run>/latest.pth out.json "label"` | MRA's `SSCMetrics` + four disjoint 12.8 m range bands, per-class IoU with GT voxel counts, pooled VRU IoU, wIoU |

Scripts use `/tmp/mra_env/bin/python` by default (`PY=... bash kaggle/train.sh ...` to change). `train.sh` honours `SEED`, `TIMEOUT` (default 11h) and `OUT_ROOT`.

## Configurations

All Kaggle configs inherit a repository config and change only the data root (`/tmp/kitti_full/`), the schedule (6 epochs,
learning rate 5e-5 = 2e-4 / 4 for batch 1 on one GPU) checkpoint/log settings and `load_from = 'ckpts/warm_start.pth'` (SGN-T warm start shared with the RaCA runs, see below).

| Config | Trains |
|---|---|
| `kaggle/configs/mra_before.py` | original MRA-T (the 'before' run) |

## Initial weights and determinism

- **Initial weights: SGN-T warm start, because MRA's own initialisation does not fit a T4.** MRA-T starts from ImageNet
  ResNet-50 and default init for the rest. On a 16 GB T4 (about 13.3 GB usable by PyTorch) that ran out of memory on the
  first training step, with `--deterministic` on or off (measured on the RaCA branch, whose baseline path is this code):
  the head seeds its sparse branch with every voxel whose occupancy logit is positive (`proposal = (pts_occ > 0)`), and a
  randomly initialised occupancy head proposes about half the grid. With the SGN-T warm start the same code trained at a
  logged 11.7 GB. The warm start is the same file for the baseline here and for the RaCA runs, so the comparison starts
  from one point; it is a deviation from MRA's recipe. `--cfg-options load_from=None` restores MRA's own initialisation
  (larger GPU needed).
- **Determinism.** `train.sh` passes `--seed 0`. `DETERMINISTIC=1 bash kaggle/train.sh ...` adds `--deterministic`, as
  `tools/dist_train.sh` does (`cudnn.deterministic = True`, `cudnn.benchmark = False`). It is off by default because the
  runs known to fit a T4 ran without it and its memory cost with the warm start has not been measured. Neither setting
  is bit-exact: spconv and torch-scatter kernels use atomic adds whose summation order is not fixed.

## Observed on a T4 (400 training frames, batch 1)

| Item | Time / memory |
|---|---|
| install.sh | ~15 min |
| expand_subset.py | ~3 min |
| MRA-T epoch | ~40 min, 11.7 GB (logged by mmcv with the SGN-T warm start, `--seed 0`, no `--deterministic`) |
| RaCA C2+C3+C4 epoch | ~53 min, 11.7 GB |
| evaluate.py (150 frames) | ~5 min |
| RaCA with C1 | not yet run; expect roughly +0.5–1 GB and a longer epoch |

Kaggle sessions are limited to 12 h, so one 6-epoch run fits per session; use *Save & Run All (Commit)* to run in the background.
The notebook clones branch `mra_Og` of `https://github.com/fafnirLore/RaCA.git`; the upstream MRA repository does not contain these files.

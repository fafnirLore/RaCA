# Running the experiments on Kaggle (RaCA)

Everything needed to reproduce the small-scale runs on a free Kaggle GPU session (Tesla T4 x1 or x2, 16 GB each) lives in this folder.
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
| Smoke test | `python kaggle/smoke_test.py kaggle/configs/raca_full.py --steps 2 --expect-head RaCAHead --expect-keys loss_align_directed,loss_scan` | 2 real training steps (loss keys, peak memory) + the validation path on 2 val frames (`dataset.evaluate`, `EarlyStoppingHook`); `--val-frames 0` skips the latter |
| Train | `bash kaggle/train.sh kaggle/configs/<run>.py <run>` | all visible GPUs (`GPUS=1` to force one), seed 0, validation after every epoch with early stopping (below; `VALIDATE=0` turns validation off); output in `/kaggle/working/runs/<run>/`; add `--cfg-options runner.max_epochs=3` etc. to override the config |
| Evaluate | `bash kaggle/eval.sh kaggle/configs/<run>.py /kaggle/working/runs/<run>/best_semkitti_SSC_mIoU_epoch_N.pth out.json "label"` (or `latest.pth`) | MRA's `SSCMetrics` + four disjoint 12.8 m range bands, per-class IoU with GT voxel counts, pooled VRU IoU, wIoU |
| Compare | `python kaggle/compare.py before.json after.json` | before/after table: overall, rare classes, per band, all classes |

Scripts use `/tmp/mra_env/bin/python` by default (`PY=... bash kaggle/train.sh ...` to change). `train.sh` honours `SEED`, `TIMEOUT` (default 11h), `OUT_ROOT`, `VALIDATE` (default 1), `GPUS` (default: all visible), `SCALE_LR` (default 1), `DETERMINISTIC` (default 0; 1 adds `--deterministic`) and `PORT`.

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

## One or two GPUs (T4 x1 / T4 x2)

`train.sh` uses every GPU that `nvidia-smi` lists. On one GPU it runs `tools/train.py` directly. On two it launches
`python -m torch.distributed.launch --nproc_per_node=2 tools/train.py <config> --launcher pytorch`, i.e. MRA's own
distributed path (`tools/dist_train.sh`, `MMDistributedDataParallel`, `CustomDistEvalHook` for validation), with batch 1
per GPU as MRA does. Because the total batch doubles, the script applies the linear scaling rule to the config before
launching (printed as `GPUS=2 -> linear scaling: ...`):

| setting | 1 GPU (config value) | 2 GPUs (applied by `train.sh`) | rule |
|---|---|---|---|
| `optimizer.lr` | 5e-5 | 1e-4 | x GPUs (MRA: 2e-4 for 4 GPUs x batch 1) |
| `lr_config.warmup_iters` | 500 | 250 | / GPUs, so the warm-up covers the same number of samples |
| `data.workers_per_gpu` | 4 | 2 | the T4 x2 machine has 4 CPU cores |
| iterations per epoch | 400 | 200 | 400 training frames / total batch |

`SCALE_LR=0` keeps the config values; your own `--cfg-options` are merged after the automatic ones and win. Epochs, early
stopping, seed and determinism are unchanged, so a 2-GPU run is the same protocol at half the wall-clock time per epoch,
not a bit-identical reproduction of a 1-GPU run (different total batch, different data order across ranks). Keep the
whole ablation on one GPU count. Validation on 2 GPUs goes through `CustomDistEvalHook`, which gathers per-GPU partial
results through `<run>/.eval_hook/`; `EarlyStoppingHook` takes its decision on rank 0 and broadcasts it. If the data
loader dies with a shared-memory ("bus error") message, add `--cfg-options data.workers_per_gpu=1`.

## Validation and early stopping

`train.sh` runs validation on the 150 validation frames after every epoch (about 5 min each). Every Kaggle config sets
`evaluation = dict(interval=1, save_best='semkitti_SSC_mIoU', rule='greater')`, so the evaluation hook keeps the best
checkpoint as `best_semkitti_SSC_mIoU_epoch_N.pth` next to `latest.pth`, and
`custom_hooks = [dict(type='EarlyStoppingHook', monitor='semkitti_SSC_mIoU', patience=2, min_delta=0.0, min_epochs=3, priority=80)]`,
which ends training once the validation mIoU has not improved for two consecutive epochs (never before epoch 3). The hook
logs `EarlyStoppingHook: semkitti_SSC_mIoU = ... (best ... at epoch N, k/2 epochs without improvement)` after each
evaluation and writes `<run>/early_stopping.json` (best value, best epoch, whether it stopped early, full history).
Override from the command line, e.g. `--cfg-options custom_hooks.0.patience=3 custom_hooks.0.min_delta=0.1`; the
cosine learning-rate schedule still spans the configured `max_epochs`, so an early stop ends at a non-zero learning rate.
The notebook evaluates the best checkpoint when one exists. Note that the stopping decision is taken on the same 150
frames that are reported as the result, so early stopping is a protocol choice applied identically to every config,
not an unbiased model selection.

Single-GPU validation did not work in the released MRA code (mmdet's `EvalHook` cannot read MRA's dict outputs);
`mmdet_train.py` now uses mmcv's `EvalHook` with MRA's own `custom_single_gpu_test` on one GPU. The multi-GPU path
(`CustomDistEvalHook`) is unchanged.

## Initial weights and determinism

- **Initial weights.** The configs set no `load_from`; the model is built and initialised exactly as MRA-T builds it
  (ImageNet ResNet-50 for the image backbone, PyTorch / mmcv default init for everything else). All configs therefore
  start from the same point for a given seed. The SGN-T warm start used in the first preliminary runs remains available
  as an opt-in (`make_warm_start.py`, `--cfg-options load_from=ckpts/warm_start.pth`).
- **Seed and determinism.** `train.sh` always passes `--seed 0` (Python, NumPy, torch and data-loader seeds, as in
  `tools/train.py`). It does **not** pass `--deterministic` by default. That flag (used by `tools/dist_train.sh`) sets
  `cudnn.deterministic = True`, which restricts cuDNN to deterministic convolution algorithms whose workspaces are larger:
  on a T4 the C2 + C3 + C4 configuration went from a logged 11.7 GB (the earlier notebooks, `--seed 0` only) to 13.0 GB on
  the first step and ran out of memory on the second, where PyTorch has about 13.3 GB of the card. The flag never made the
  run bit-exact anyway: spconv and torch-scatter kernels use atomic adds whose summation order is not fixed. Repeats with the
  same seed therefore agree closely but not to the last digit, with or without the flag. `DETERMINISTIC=1 bash kaggle/train.sh ...`
  restores it for GPUs with more memory; `SEED=1 ...` changes the seed. The smoke test uses the same default and prints the
  cuDNN flags and peak memory per step (`--deterministic` to measure the other setting).

## Observed on a T4 (400 training frames, batch 1)

| Item | Time / memory |
|---|---|
| install.sh | ~15 min |
| expand_subset.py | ~3 min |
| validation per epoch | ~5 min |
| MRA-T epoch, 1 GPU | ~40 min, 11.7 GB (logged by mmcv in the earlier notebooks: `--seed 0`, SGN-T warm start; with `--deterministic` the same config needs >13 GB and does not fit) |
| any config, 2 GPUs | not yet measured; expect roughly half the epoch time, same memory per GPU plus NCCL buffers |
| RaCA C2+C3+C4 epoch | ~53 min, 11.7 GB |
| evaluate.py (150 frames) | ~5 min |
| RaCA with C1 | not yet run; expect roughly +0.5–1 GB and a longer epoch |

Kaggle sessions are limited to 12 h, so one 6-epoch run fits per session; use *Save & Run All (Commit)* to run in the background.


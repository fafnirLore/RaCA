# Running the experiments on Kaggle (RaCA)

Everything needed to reproduce the small-scale runs on a free Kaggle GPU session (Tesla T4 x1 or x2, 16 GB each) lives in this folder.
The notebook `kaggle/kaggle_raca.ipynb` strings the steps together; each step is also a script that can be run by hand.

## Datasets to attach (right-hand panel → *Add Input*)

| Dataset | Where it comes from | What this folder reads from it |
|---|---|---|
| **scenes** (SemanticKITTI subset, 400 train / 150 val frames) | Output of the step-2 "building-subset" notebook (`daniyalimran/building-subset`), saved compactly to fit Kaggle's 20 GB output limit | `**/kitti_subset/selection.json` (summary of the split) and the `dataset/` tree next to it |
| **mra-before-run** (output of the original-MRA run) | Output of the "before" notebook (`daniyalimran/mra-before-run`) | `**/ckpts/warm_start.pth` (shared starting weights; otherwise rebuilt from SGN-T) and `**/runs/before/results_trained.json` (for `compare.py`) |

The compact scenes layout and how it is expanded back to MRA's format are documented in `expand_subset.py`:
labels `labels/XX/<frame>_1_1.npy` are stored as uint8 and rewritten as float32 (values 0–19, 255), point clouds
`sequences_msnet3d_lidar/sequences/XX/<frame>.bin` are stored as x,y,z and rewritten as x,y,z,1 (as MRA's `depth2lidar.py`
writes them), and images, voxels, calib, poses and `*_1_2.npy` are symlinked. Expansion writes to `/tmp/kitti_full/`
(about 3 minutes).

Weights downloaded by `get_weights.sh sgn`: ImageNet ResNet-50 (`ckpts/resnet50-19c8e357.pth`, PyTorch model zoo; MRA-T's
`pretrained`) and SGN-T SemanticKITTI (`ckpts/sgn-T.pth`, SGN model zoo on Google Drive, id `1cDKX6SpyoAHr-cfmjTRDsNF-3_JgtH3p`)
for the shared warm start (without the `sgn` argument only ResNet-50 is fetched).

## Steps and commands (run from the repository root)

| Step | Command | Notes |
|---|---|---|
| Install the software | `bash kaggle/install.sh` | ~15 min; Python 3.8 env at `/tmp/mra_env`, torch 1.9.1+cu111, mmcv-full 1.4.0, mmdet 2.14.0, mmseg 0.14.1, spconv-cu113 2.3.6, torch-scatter 2.0.8, mmdet3d v0.17.1 without compiled CUDA ops (`patch_mmdet3d.py`) |
| Download weights | `bash kaggle/get_weights.sh sgn` | ResNet-50 and SGN-T |
| Rebuild the data | `python kaggle/expand_subset.py <scenes>/kitti_subset/ /tmp/kitti_full/` | both paths end with `/` |
| Starting weights | `bash kaggle/ensure_warm_start.sh` (also run automatically by `smoke_test.py` and `train.sh` when the file is missing; copies an attached `**/ckpts/warm_start.pth`, else runs `get_weights.sh sgn` + `make_warm_start.py`) | SGN-T weights + SGN's main branch copied into MRA's coarse branch; 15 tensors stay random (fusion conv, CDA heads). Skip when `warm_start.pth` from the before run is attached: copy it instead, so all runs share one start. Required on a T4, see below |
| Smoke test | `python kaggle/smoke_test.py kaggle/configs/raca_full.py --steps 2 --expect-head RaCAHead --expect-keys loss_align_directed,loss_scan` | 2 real training steps (loss keys, peak memory) + the validation path on 2 val frames (`dataset.evaluate`, `EarlyStoppingHook`); `--val-frames 0` skips the latter |
| Train | `bash kaggle/train.sh kaggle/configs/<run>.py <run>` | all visible GPUs (`GPUS=1` to force one), seed 0, validation after every epoch with early stopping (below; `VALIDATE=0` turns validation off); output in `/kaggle/working/runs/<run>/`; add `--cfg-options runner.max_epochs=3` etc. to override the config |
| Evaluate | `bash kaggle/eval.sh kaggle/configs/<run>.py /kaggle/working/runs/<run>/best_semkitti_SSC_mIoU_epoch_N.pth out.json "label"` (or `latest.pth`) | MRA's `SSCMetrics` + four disjoint 12.8 m range bands, per-class IoU with GT voxel counts, pooled VRU IoU, wIoU |
| Compare | `python kaggle/compare.py before.json after.json` | before/after table: overall, rare classes, per band, all classes |

Scripts use `/tmp/mra_env/bin/python` by default (`PY=... bash kaggle/train.sh ...` to change). `train.sh` honours `SEED`, `TIMEOUT` (default 11h), `OUT_ROOT`, `VALIDATE` (default 1), `GPUS` (default: all visible), `SCALE_LR` (default 1), `DETERMINISTIC` (default 0; 1 adds `--deterministic`) and `PORT`.

## Configurations

All Kaggle configs inherit a repository config and change only the data root (`/tmp/kitti_full/`), the schedule (6 epochs,
learning rate 5e-5 = 2e-4 / 4 for batch 1 on one GPU) checkpoint/log settings and `load_from = 'ckpts/warm_start.pth'` (shared SGN-T warm start, see below).

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

`raca_c2c3c4.py` currently differs from the other configs: 24 epochs and early stopping on the total training loss
(`monitor='loss', source='train', rule='less'`, patience 3, not before epoch 10). With `source='train'` the hook averages
the logged training variable over the epoch; `loss` is the sum of all loss terms. There is no validation loss, because
MRA's `forward_train` updates the EMA teacher and cannot be run on validation frames without side effects. The best
checkpoint is still chosen by validation mIoU.

Single-GPU validation did not work in the released MRA code (mmdet's `EvalHook` cannot read MRA's dict outputs);
`mmdet_train.py` now uses mmcv's `EvalHook` with MRA's own `custom_single_gpu_test` on one GPU. The multi-GPU path
(`CustomDistEvalHook`) is unchanged.

## Initial weights and determinism

- **Initial weights: shared SGN-T warm start, because MRA's own initialisation does not fit a T4.** MRA-T starts from
  ImageNet ResNet-50 only. Its sparse branch takes every voxel whose point-cloud occupancy logit is positive as a seed
  (`raca_head.py`: `proposal = (pts_occ > 0)`), and a randomly initialised occupancy head gives logits around zero, so about
  half of the 128x128x16 grid becomes seeds for the sparse convolutions and the semantic header. Measured with the smoke test
  on a T4 (PyTorch gets about 13.3 GB of the card): `raca_c2c3c4` from MRA's own init allocates 13.0 GB on step 0 and runs out
  of memory, with `--deterministic` on or off; the same code with the SGN-T warm start trained at a logged 11.7 GB. The warm
  start (`make_warm_start.py`: SGN-T's 674 matching tensors, SGN's main branch copied into MRA's coarse branch, 15 tensors random)
  is therefore the Kaggle protocol for every config, including `mra_before`; it is a documented deviation from MRA's recipe and it
  is identical for all runs. `--cfg-options load_from=None` reproduces MRA's own start on a larger GPU.
- **Seed and determinism.** `train.sh` always passes `--seed 0` (Python, NumPy, torch and data-loader seeds). `--deterministic`
  (`cudnn.deterministic = True`, as `tools/dist_train.sh`) is off by default: the configuration known to fit the T4 ran without
  it, and its memory cost on this model has not been measured (the earlier attribution of the out-of-memory to this flag was
  wrong; the cause is the initialisation above). `DETERMINISTIC=1 bash kaggle/train.sh ...` turns it on; the smoke test takes
  `--deterministic` and prints peak memory, so its cost can be measured in two minutes. Neither setting is bit-exact: spconv and
  torch-scatter kernels use atomic adds. `SEED=1 ...` changes the seed.

## molab, one 96 GB GPU (git tag `molab-96gb`)

[molab](https://molab.marimo.io) offers one NVIDIA RTX Pro 6000 Blackwell (96 GB), 4 CPUs, 32 GB RAM, 12-hour sessions.
Open the notebook from GitHub:
`https://molab.marimo.io/github/fafnirLore/RaCA/blob/farquleet_add_module/kaggle/molab_raca.py`

| File | Purpose |
|---|---|
| `kaggle/molab_raca.py` | marimo notebook with the same steps as `kaggle_raca.ipynb` (install, GPU check, data, smoke test, train, evaluate); outputs go to `~/raca_work/runs/<run>/` |
| `kaggle/configs/molab_raca_full.py` | `raca_full.py` with `load_from = None`: full method C1-C4 from MRA's own initialisation |
| `kaggle/configs/molab_mra_before.py` | `mra_before.py` with `load_from = None`: the baseline under the same protocol |
| `kaggle/check_gpu.py` | runs a dense and a sparse convolution on the GPU with the installed PyTorch and stops with an explanation when that fails |

What the larger GPU changes: the SGN-T warm start is dropped (it existed only because MRA's own initialisation needs
> 13 GB on the first steps), `--deterministic` is on (`DETERMINISTIC=1`), and C1 fits. Batch size, learning rate, data,
epochs and seed are those of the Kaggle configs. These runs are comparable with each other, not with the warm-started
Kaggle runs.

**Not yet run, and one known risk.** Nothing in this section has been executed on molab. MRA's stack is pinned to
PyTorch 1.9.1 built for CUDA 11.1, whose binaries carry GPU kernels only up to the RTX 30 generation; Blackwell GPUs
need a much newer PyTorch, which mmcv-full 1.4.0 / mmdet3d 0.17.1 do not support. `check_gpu.py` (step 1b of the
notebook) tests exactly this before any data or training time is spent. If it fails, the stack has to be ported; no
config can fix it. Also unverified: that molab's container allows the micromamba install to `/tmp`, and where the
scenes subset is mounted (`SCENES` in the notebook).

## Observed on a T4 (400 training frames, batch 1)

| Item | Time / memory |
|---|---|
| install.sh | ~15 min |
| expand_subset.py | ~3 min |
| validation per epoch | ~5 min |
| MRA-T epoch, 1 GPU | ~40 min, 11.7 GB (logged by mmcv: `--seed 0`, SGN-T warm start; from MRA's own initialisation the same config needs > 13 GB and does not fit) |
| any config, 2 GPUs | not yet measured; expect roughly half the epoch time, same memory per GPU plus NCCL buffers |
| RaCA C2+C3+C4 epoch | ~53 min, 11.7 GB |
| evaluate.py (150 frames) | ~5 min |
| RaCA with C1 | not yet run; expect roughly +0.5–1 GB and a longer epoch |

Kaggle sessions are limited to 12 h, so one 6-epoch run fits per session; use *Save & Run All (Commit)* to run in the background.


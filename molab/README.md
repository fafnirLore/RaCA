# Running the baseline on molab (original MRA, branch `mra_Og_molab`)

This branch is `mra_Og` (the unmodified MRA code plus the `kaggle/` folder) plus this `molab/` folder. It moves the
"before" run from a Kaggle T4 to [molab](https://molab.marimo.io) with its **NVIDIA RTX PRO 6000 Blackwell (96 GB)**.
No MRA file and no `kaggle/` file is modified; the molab scripts call the Kaggle ones wherever they can.

The notebook `molab/molab_before.ipynb` strings the steps together. Each step is also a script that can be run by hand.

## Why a new software stack was needed

The Kaggle stack cannot run on this GPU. Blackwell cards have compute capability 12.0 (`sm_120`), which only CUDA
12.8 and newer can target, so PyTorch 2.7 is the oldest release with kernels for them. PyTorch 1.9.1 + CUDA 11.1 stops
with `CUDA error: no kernel image is available for execution on the device`. Everything built against PyTorch had to
move with it:

| Package | Kaggle (`kaggle/install.sh`) | molab (`molab/install.sh`) | Note |
|---|---|---|---|
| Python | 3.8 | 3.10 | PyTorch 2.7 needs 3.9 or newer |
| torch / torchvision | 1.9.1+cu111 / 0.10.1 | 2.7.1+cu128 / 0.22.1 | |
| numpy | 1.19.5 | 1.23.5 | newest release that still has `np.int` / `np.float`, used by the old OpenMMLab code |
| mmcv | mmcv-full 1.4.0 | mmcv 1.7.2 ("lite") | 1.7.2 is the 1.x release that supports PyTorch 2. Its compiled operators are replaced by placeholders (`patch_stack.py`): MRA calls none of them |
| mmdet / mmseg / mmdet3d | 2.14.0 / 0.14.1 / 0.17.1 | same | their "mmcv <= 1.4.0" checks are relaxed by `patch_stack.py`; mmdet3d's CUDA operators are placeholders, as on Kaggle |
| spconv | spconv-cu113 2.3.6 | spconv 2.3.8 for CUDA 12.8 with `sm_120` kernels | community build of the unmodified upstream source ([fafraob/spconv-wheels](https://github.com/fafraob/spconv-wheels)), SHA-256 pinned in `install.sh`. Fallback: upstream `spconv-cu126` (works on Blackwell through the driver's PTX JIT) |
| torch-scatter | 2.0.8 | 2.1.2 (PyG wheel for torch 2.7 + cu128) | Fallback: `molab/shims/torch_scatter`, the three functions MRA uses written in plain PyTorch |
| numba / networkx / scikit-image | 0.53.0 / 2.2 / 0.19.3 | 0.56.4 / 2.8.8 / 0.19.3 | Python 3.10 versions |

Nothing is compiled, so the machine needs neither gcc nor a CUDA toolkit. `install.sh` runs a real GPU operation
for torch, spconv and torch-scatter before it accepts each of them, and prints which variant was installed.

What this means for the comparison: the model code, the config, the data and the starting weights are the ones of the
Kaggle run, but the numerical libraries are five years newer. Results will be close to, not bit-identical with, a
PyTorch 1.9 run (neither stack is bit-exact from run to run anyway, see `kaggle/README.md`). For a clean
before / after comparison, run the RaCA branch on the same molab stack.

## Storage on molab

molab clears the machine when a notebook shuts down (after 12 hours, or 90 minutes idle) and keeps only source files
and files under 1 GB uploaded through its file browser. So:

- the dataset is read from where it already is (`DATA` setting, below) and rebuilt under `/tmp/kitti_full/`;
- the code is cloned to `/tmp/RaCA_Og`, the environment lives at `/tmp/mra_env`, runs are written to `/tmp/mra_runs/<run>/`;
- `BACKUP_TO` copies the newest checkpoint, the logs and the results to a folder or to a private Hugging Face
  dataset repository every 10 minutes and at the end; `train.sh` resumes from `<run>/latest.pth`, and the notebook
  restores that file from the backup when the run folder is empty. Without `BACKUP_TO`, download
  `<run>_results.zip` (written next to the notebook by step 7: logs, results and the trained model) before the
  session ends.

## Dataset (`DATA`)

`prepare_data.py` accepts a folder, a `.zip` / `.tar(.gz)` archive, `hf://datasets/<user>/<repo>[/sub]`,
`kaggle://datasets/<user>/<slug>` (a Kaggle dataset), `kaggle://kernels/<user>/<slug>` (the output of a Kaggle
notebook), or `auto` (search the machine). Private Kaggle data needs `KAGGLE_API_TOKEN` in the
notebook's `.env` (kaggle.com → Settings → API → Generate New Token; older keys: `KAGGLE_USERNAME` + `KAGGLE_KEY`). It recognises two layouts:

| Layout | Recognised by | What happens |
|---|---|---|
| compact (the Kaggle "scenes" subset, `kitti_subset/`) | `selection.json`, uint8 labels or x,y,z point clouds | rebuilt into MRA's format by `kaggle/expand_subset.py` (~3 min) |
| full (MRA's own format) | float32 labels and x,y,z,1 point clouds | linked as it is |

It then checks every file MRA will read (images and point clouds including the four earlier frames per sample,
both label resolutions, `calib.txt` / `poses.txt` of all eleven sequences) and stops with a list if something is missing.

## Steps and commands (run from the repository root)

| Step | Command | Notes |
|---|---|---|
| Install the software | `bash molab/install.sh` | 5-10 min; environment at `/tmp/mra_env` |
| Rebuild and check the data | `/tmp/mra_env/bin/python molab/prepare_data.py <source> /tmp/kitti_full/` | prints `DATA_ROOT=...` |
| Starting weights | `bash molab/get_weights.sh` | ResNet-50 + SGN-T → `ckpts/warm_start.pth` (`kaggle/make_warm_start.py`); `resnet` as argument for ResNet-50 only |
| Smoke test | `source molab/env.sh && PYTHONPATH=$PWD $PY molab/smoke_test.py molab/configs/<run>.py` | two training steps + one inference step, prints peak GPU memory (~1 min) |
| Train | `bash molab/train.sh molab/configs/<run>.py <run>` | one GPU, seed 0, `--no-validate`, resumes automatically; honours `SEED`, `DETERMINISTIC=1`, `RESUME=0`, `TIMEOUT` (11h), `OUT_ROOT`, `BACKUP_TO`, `BACKUP_EVERY` |
| Evaluate | `bash molab/eval.sh molab/configs/<run>.py /tmp/mra_runs/<run>/latest.pth out.json "label"` | `kaggle/evaluate.py`, unchanged |

## Configurations

| Config | Trains |
|---|---|
| `molab/configs/mra_before.py` | original MRA-T with the Kaggle recipe (inherits `kaggle/configs/mra_before.py`; only the output folder differs) |

The 96 GB card also fits MRA's own initialisation, which the T4 could not hold:
`bash molab/train.sh molab/configs/mra_before.py mra_paper_init --cfg-options load_from=None`.
The code processes one sample per step (`img_metas[0]` in the heads), so the batch size stays 1.

## Notes

- **Resuming.** MRA's self-distillation teacher is re-copied from the student on the first step after a restart
  (`local_step` is not stored in the checkpoint; this is MRA's code, unchanged). A resumed run therefore differs
  slightly from an uninterrupted one.
- **Checkpoint loading.** PyTorch 2.6+ loads checkpoints with `weights_only=True`; mmcv 1.x checkpoints also contain
  plain metadata, so `molab/env.sh` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for these files.
- **ipynb and marimo.** molab runs marimo notebooks. The `.ipynb` uses plain Python cells (no `%%bash`) whose steps
  depend on each other through variables, so it can be converted with `marimo convert molab_before.ipynb -o molab_before.py`
  and runs in the same order in both formats.

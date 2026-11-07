<div align="center">

# RaCA: Range- and Class-Aware Critical Voxel Alignment for Camera-Based 3D Semantic Scene Completion

</div>

> **RaCA** extends **MRA** (*Multi-Resolution Alignment for Voxel Sparsity in Camera-Based 3D Semantic Scene Completion*, Zhiwen Yang and Yuxin Peng, IEEE TIP 2026) to improve two documented weaknesses of camera-based SSC: **far-range voxels** and **rare / small-object classes** (bicycle, motorcycle, person, bicyclist, motorcyclist).
>
> Course project, *Advanced Artificial Intelligence*, MS AI – A.
> Farquleet Farhat Gondal (26I-7631) and Daniyal Imran (26I-7604).

This repository is a fork of the official MRA code (`PKU-ICST-MIPL/MRA_TIP`, commit `d6b560f`). MRA itself is left untouched and still runs from its own configs; RaCA is added next to it as a new head, a utility module and new configs.

## 1. What RaCA changes in MRA

MRA's Critical Distribution Alignment (CDA) draws 4,096 "critical" voxels per grid by prediction confidence, draws them **independently** for the fine (128×128×16) and coarse (64×64×8) grids, aligns the two sets with a **symmetric** KL, and supervises the coarse grid with a **strided** copy of the label (every second voxel). RaCA replaces these three design choices; everything else in MRA (view transformation, SGB / SDB, CSA, self-distillation, losses, schedule) is unchanged.

| Component | What it does | Where | Training / inference |
|---|---|---|---|
| **C2** Stratified shared critical voxels | One set of N = 4,096 voxel-centre locations per frame, drawn by **strata**: foreground (with a sub-quota for rare classes when the frame has any), far background (forward index ≥ 128, i.e. beyond 25.6 m), remaining background, and uniform random. The **same locations** are read on both grids, so the two refined predictions are co-located. Unused budget of an empty stratum is passed down the chain rare → foreground → far → rest → random. | `raca_utils.stratified_critical_voxels`, used by `SharedCDAHead` in `raca_head.py` | training only |
| **C3** Co-located directed alignment | Per-voxel KL over the **class axis** between the fine and coarse refined predictions at the shared locations, with a stop-gradient on the leading grid. The fine grid leads on rare and foreground voxels; elsewhere the grid with lower predictive entropy leads. Replaces MRA's symmetric `cda_kl_loss`. Requires C2. | `raca_utils.directed_alignment` | training only |
| **C4** Rare-preserving coarse supervision | Coarse label of each 2×2×2 block = a rare class if any voxel in the block has one, else the majority label (empty included), else 255. CSA map max-pooled instead of strided. Coarse seed **proposal** max-pooled instead of strided. | `raca_utils.coarse_labels_rare_preserving`, `raca_utils.pool_proposal` | labels / CSA: training only; proposal pooling also at inference |

`RaCAHead` has exactly the parameters of `MRAHead` (no new layers), so MRA and SGN checkpoints load unchanged, and `RaCAHead` with `use_c2 = use_c3 = use_c4 = False` is numerically `MRAHead`.

### Files added or edited

| Path | Status | Content |
|---|---|---|
| `projects/mmdet3d_plugin/mra/utils/raca_utils.py` | new | C2 / C3 / C4 as pure-torch functions (torch ≥ 1.9, no `torch.isin`) |
| `projects/mmdet3d_plugin/mra/dense_heads/raca_head.py` | new | `SharedCDAHead` (MRA's CDA head that can read given locations) and `RaCAHead` (MRA head + C2 / C3 / C4 + loss-weight options) |
| `projects/mmdet3d_plugin/mra/dense_heads/__init__.py` | one line appended | registers `RaCAHead` |
| `projects/configs/raca/RaCA-T.py` | new | full method on top of `MRA-T.py` |
| `projects/configs/raca/RaCA-T-C2.py`, `RaCA-T-C2C3.py`, `RaCA-T-C4.py` | new | ablations (C2 only, C2 + C3, C4 only) |
| `README.md` | rewritten | this file |

Everything else (`mra_head.py`, `cda_header.py`, detectors, datasets, losses, preprocessing, tools) is byte-identical to the original MRA repository.

### Head options (`pts_bbox_head` in the config)

| Option | Default | Meaning |
|---|---|---|
| `use_c2` | `True` | stratified, shared critical voxels |
| `use_c3` | `True` | co-located directed alignment (needs `use_c2=True`) |
| `use_c4` | `True` | rare-preserving coarse labels / CSA and max-pooled coarse proposal |
| `c2_shares` | `(0.25, 0.25, 0.40, 0.10)` | share of N for foreground, far background, rest, uniform random |
| `c2_rare_share` | `0.5` | share of the foreground budget reserved for rare classes when the frame contains any |
| `c2_far_x` | `128` | first forward voxel index counted as "far" (128 × 0.2 m = 25.6 m) |
| `lambda_align` | `1.0` | weight of the C3 alignment loss |
| `w_mining` | `1.0` | weight of the fine-grid critical-voxel CE loss (MRA: fixed 1.0) |
| `w_mining_down1` | `1.0` | weight of the coarse-grid critical-voxel CE loss (MRA: fixed 1.0) |

Rare classes are SemanticKITTI ids 2, 3, 6, 7, 8; the foreground group is MRA's own "thing" group (ids 1–8, 14, 18, 19) from `get_csa`.

## 2. Getting started

Installation and data preparation are those of MRA (below). Then:

```bash
# RaCA-T (full method), 4 GPUs
./tools/dist_train.sh ./projects/configs/raca/RaCA-T.py 4
./tools/dist_test.sh  ./projects/configs/raca/RaCA-T.py ./path/to/ckpt.pth 4

# ablations
./tools/dist_train.sh ./projects/configs/raca/RaCA-T-C2.py 4     # C2 only
./tools/dist_train.sh ./projects/configs/raca/RaCA-T-C2C3.py 4   # C2 + C3
./tools/dist_train.sh ./projects/configs/raca/RaCA-T-C4.py 4     # C4 only

# original MRA (unchanged)
./tools/dist_train.sh ./projects/configs/mra/MRA-T.py 4
```

The RaCA configs inherit every MRA-T setting (data, AdamW 2e-4, cosine schedule, gradient clipping at 35) and override only the head.

## 3. Preliminary result (controlled small-scale run)

Both models were trained under identical conditions on Kaggle: SemanticKITTI subset of **400 training / 150 validation frames**, **6 epochs**, batch 1, learning rate 5e-5, one T4 GPU, seed 0, the **same starting weights** (SGN-T weights with SGN's main branch copied into MRA's coarse branch; the CDA output layers and the fusion conv start random in both runs). "Before" is `MRAHead`; "after" is `RaCAHead` with all three components and default weights. Evaluation uses MRA's `SSCMetrics`, plus four disjoint 12.8 m forward-range bands and a pooled vulnerable-road-user (VRU) IoU over the five rare classes.

| metric (150 val frames) | MRA (before) | RaCA (after) | change |
|---|---|---|---|
| IoU | 42.20 | 41.72 | −0.48 |
| mIoU | 13.98 | 13.25 | −0.73 |
| wIoU | 29.54 | 29.07 | −0.47 |
| VRU IoU (5 rare classes pooled) | 2.23 | 2.23 | +0.00 |

| range band | mIoU before → after | VRU IoU before → after |
|---|---|---|
| 0–12.8 m | 16.01 → 15.45 | 3.91 → 3.74 |
| 12.8–25.6 m | 15.95 → 15.07 | 2.96 → 2.55 |
| 25.6–38.4 m | 12.49 → 11.62 | 0.81 → 1.14 |
| 38.4–51.2 m | 9.29 → 9.13 | 0.15 → 0.09 |

Per class, person (+0.34) and motorcycle (+0.30) improve slightly; car (−3.81), truck (−2.51) and fence (−2.16) drop and drive the mIoU change.

**Reading of this run.** The stratified set is deliberately hard, so the two critical-voxel CE losses end about four times larger than in MRA (1.09 / 2.24 vs 0.55 / 0.56), and the gradient norm exceeded MRA's clip ceiling of 35 in **62 %** of logged steps (2 % for MRA). Clipping then scales down every gradient, including the dense losses that the common classes rely on. This is the reason the head now exposes `w_mining` and `w_mining_down1`; the planned re-run lowers them so that the total gradient norm matches MRA's. With a single seed and no deterministic flag, differences of a few tenths of a point are within noise; the methodology calls for three seeds per configuration.

These numbers are **not** comparable with the paper's full-dataset results (3,834 training frames, 48 epochs in the released config, 4 GPUs).

## 4. Notes

- The Kaggle notebooks that produced the result above (`train-mra-before.ipynb`, `train-mra-after.ipynb`, kept outside this repository) wrote the same code into the cloned repo under the earlier class name `MRAHeadRC`; in this repository the config type is `RaCAHead`. Checkpoints are interchangeable because the parameter names are identical.
- Environment deviations used on Kaggle (spconv 2.3.6 instead of 2.1.25, mmdet3d 0.17.1 installed without its CUDA operators, torch 1.9.1 + cu111) are notebook-side and do not change this repository.
- Observations about the released MRA code that matter for interpreting results: the released critical-voxel selection ranks candidates by confidence only (the CSA factor of Eq. 18 is not applied there); the circulated KL in `mra_head.py` is taken over the sample axis; the CSA weight of the mining loss is cast to integer and therefore equals 1. RaCA keeps these as they are except where C2 / C3 replace the corresponding step.

---

## Original MRA README

<div align="center">

### Multi-Resolution Alignment for Voxel Sparsity in Camera-Based 3D Semantic Scene Completion

</div>

> **Multi-Resolution Alignment for Voxel Sparsity in Camera-Based 3D Semantic Scene Completion**, Zhiwen Yang and Yuxin Peng, IEEE Transactions on Image Processing, 2026.

### News
- [2026/01]: Accepted by IEEE TIP 2026!
- [2025/12]: We release the evaluation results and training code for SemanticKITTI and SSCBench-KITTI-360.

### Abstract
Camera-based 3D semantic scene completion (SSC) offers a cost-effective solution for assessing the geometric occupancy and semantic labels of each voxel in the surrounding 3D scene with image inputs, providing a voxel-level scene perception foundation for the perception-prediction-planning autonomous driving systems. Although significant progress has been made in existing methods, their optimization rely solely on the supervision from voxel labels and face the challenge of voxel sparsity as a large portion of voxels in autonomous driving scenarios are empty, which limits both optimization efficiency and model performance. To address this issue, we propose a Multi-Resolution Alignment (MRA) approach to mitigate voxel sparsity in camera-based 3D semantic scene completion, which exploits the scene and instance level alignment across multi-resolution 3D features as auxiliary supervision.
Specifically, we first propose the Multi-resolution View Transformer module, which projects 2D image features into multi-resolution 3D features and aligns them at the scene level through fusing discriminative seed features.
Furthermore, we design the Cubic Semantic Anisotropy module to identify the instance-level semantic significance of each voxel, accounting for the semantic differences of a specific voxel against its neighboring voxels within a cubic area.
Finally, we devise a Critical Distribution Alignment module, which selects critical voxels as instance-level anchors with the guidance of cubic semantic anisotropy, and applies a circulated loss for auxiliary supervision on the critical feature distribution consistency across different resolutions.
Extensive experiments on the SemanticKITTI and SSCBench-KITTI-360 datasets demonstrate that our MRA approach significantly outperforms existing state-of-the-art methods, showcasing its effectiveness in mitigating the impact of sparse voxel labels.

### Method

| ![MRA.jpg](./teaser/arch.png) |
|:--:|

### Installation
Please refer to [Voxformer](https://github.com/NVlabs/VoxFormer) to create base environment. Some extra packages are needed to be installed:
- spconv-cu111==2.1.25
- torch-scatter==2.0.8
- tochmetrics>=0.9.0

### Prepare Dataset
Please refer to the [README](preprocess/README.md) in the preprocess folder for details.

### Run and Eval

Train MRA with 4 GPUs
```
./tools/dist_train.sh ./projects/configs/mra/MRA-T.py 4
```

Eval MRA with 4 GPUs
```
./tools/dist_test.sh ./projects/configs/mra/MRA-T.py ./path/to/ckpts.pth 4
```

### Acknowledgement

Many thanks to these excellent open source projects:
- [MRA](https://github.com/PKU-ICST-MIPL/MRA_TIP) (the base of this fork)
- [SGN](https://github.com/Jieqianyu/SGN)
- [HASSC](https://github.com/songw-zju/HASSC)
- [VoxFormer](https://github.com/NVlabs/VoxFormer)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [AICNet](https://github.com/waterljwant/SSC)

### Citation

If you use this code, please cite the MRA paper:
```
@article{yang2026multi,
  title={Multi-Resolution Alignment for Voxel Sparsity in Camera-Based 3D Semantic Scene Completion},
  author={Yang, Zhiwen and Peng, Yuxin},
  journal={IEEE Transactions on Image Processing},
  year={2026},
  publisher={IEEE}
}
```

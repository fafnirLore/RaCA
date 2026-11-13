_base_ = ['./raca_full.py']
# molab (https://molab.marimo.io) run on one 96 GB GPU: RaCA, full method C1 + C2 + C3 + C4.
# Same data, schedule, seed and learning rate as kaggle/configs/raca_full.py (batch 1 on one GPU, lr 5e-5); only what the
# 16 GB T4 forced is undone:
#   * starting weights are MRA's own again (ImageNet ResNet-50 from `pretrained`, default init for the rest). The SGN-T warm
#     start exists only because that initialisation needs > 13 GB on the first steps. Compare against molab_mra_before.py,
#     which starts the same way, not against the warm-started Kaggle runs.
#   * C1 (about +1.2 GB of activations on top of C2+C3+C4) fits, so the full method can be trained.
# Run with DETERMINISTIC=1 bash kaggle/train.sh ... to add --deterministic as tools/dist_train.sh does.
# Batch size stays 1: larger batches have never been run with this code.
load_from = None
work_dir = 'runs/molab_raca_full'

_base_ = ['./mra_before.py']
# molab (https://molab.marimo.io) run on one 96 GB GPU: original MRA-T, the baseline for molab_raca_full.py.
# As kaggle/configs/mra_before.py, but starting from MRA's own initialisation (ImageNet ResNet-50 from `pretrained`, default
# init for the rest) instead of the SGN-T warm start, which was only needed to fit a 16 GB T4.
load_from = None
work_dir = 'runs/molab_mra_before'

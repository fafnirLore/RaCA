_base_ = ['../../kaggle/configs/mra_before.py']
# molab run: original MRA-T (the 'before' run) on one RTX PRO 6000 Blackwell (96 GB).
# Inherits the Kaggle recipe of this branch unchanged (subset rebuilt at /tmp/kitti_full/, batch 1, learning rate 5e-5,
# seed 0, SGN-T warm start, validation after every epoch, early stopping); only the output folder differs.
# The software stack differs too (PyTorch 2.7.1 instead of 1.9.1, see molab/README.md).
#
# The 96 GB card also fits MRA's own initialisation (ImageNet ResNet-50 only), which a 16 GB T4 does not:
#   bash molab/train.sh molab/configs/mra_before.py mra_paper_init --cfg-options load_from=None
# Other overrides work the same way, e.g. --cfg-options runner.max_epochs=3 optimizer.lr=2.5e-5
work_dir = '/tmp/mra_runs/mra_before'

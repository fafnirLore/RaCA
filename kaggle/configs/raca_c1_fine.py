_base_ = ['../../projects/configs/raca/RaCA-T-C1-fine.py']
# Kaggle run: RaCA, C1 on the fine grid only
# Subset of SemanticKITTI (400 train / 150 val frames) rebuilt at /tmp/kitti_full/ by kaggle/expand_subset.py,
# one T4 GPU, batch 1. Learning rate 5e-5 = MRA's 2e-4 / 4 (linear scaling from 4 GPUs x 1 to 1 x 1).
# Override anything with --cfg-options, e.g. --cfg-options runner.max_epochs=3 optimizer.lr=2.5e-5
_root_ = '/tmp/kitti_full/'
data = dict(
    workers_per_gpu=4,
    train=dict(data_root=_root_, preprocess_root=_root_ + 'dataset'),
    val=dict(data_root=_root_, preprocess_root=_root_ + 'dataset'),
    test=dict(data_root=_root_, preprocess_root=_root_ + 'dataset'))
optimizer = dict(lr=5e-5)
total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
load_from = 'ckpts/warm_start.pth'          # shared starting weights, see kaggle/make_warm_start.py
work_dir = '/kaggle/working/runs/raca_c1_fine'

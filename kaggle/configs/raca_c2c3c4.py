_base_ = ['../../projects/configs/raca/RaCA-T-C2C3C4.py']
# Kaggle run: RaCA, training-only components C2 + C3 + C4 (the 'after' run)
# Subset of SemanticKITTI (400 train / 150 val frames) rebuilt at /tmp/kitti_full/ by kaggle/expand_subset.py,
# one or two T4 GPUs, batch 1 per GPU, seed 0. Learning rate 5e-5 = MRA's 2e-4 / 4 (linear scaling from 4 GPUs x 1 to 1 x 1).
# Override anything with --cfg-options, e.g. --cfg-options runner.max_epochs=3 optimizer.lr=2.5e-5
_root_ = '/tmp/kitti_full/'
data = dict(
    workers_per_gpu=4,
    train=dict(data_root=_root_, preprocess_root=_root_ + 'dataset'),
    val=dict(data_root=_root_, preprocess_root=_root_ + 'dataset'),
    test=dict(data_root=_root_, preprocess_root=_root_ + 'dataset'))
optimizer = dict(lr=5e-5)
total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
# Starting weights: SGN-T warm start shared by every run (kaggle/make_warm_start.py). MRA's own initialisation (ImageNet
# ResNet-50 only) does not fit a 16 GB T4: the sparse branch seeds every voxel with occupancy logit > 0, and a random occupancy
# head proposes about half the grid (> 13 GB on the first step, measured). --cfg-options load_from=None to try it anyway.
load_from = 'ckpts/warm_start.pth'
work_dir = '/kaggle/working/runs/raca_c2c3c4'

# Early stopping on the total training loss ('loss' = sum of all loss terms, mean over the epoch's iterations; source='train').
# Validation still runs each epoch (~5 min; VALIDATE=0 in train.sh disables it) and keeps best_semkitti_SSC_mIoU_epoch_N.pth.
evaluation = dict(interval=1, save_best='semkitti_SSC_mIoU', rule='greater')
custom_hooks = [dict(type='EarlyStoppingHook', monitor='loss', source='train', rule='less',
                     patience=3,        # stop after this many epochs without improvement
                     min_delta=0.0,     # a decrease of the loss smaller than this does not count
                     min_epochs=10,      # never stop before this many epochs
                     priority=80)]

_base_ = ['../../projects/configs/raca/RaCA-T.py']
# Kaggle run: RaCA, full method C1 + C2 + C3 + C4
# Subset of SemanticKITTI (400 train / 150 val frames) rebuilt at /tmp/kitti_full/ by kaggle/expand_subset.py,
# one T4 GPU, batch 1, --deterministic as in tools/dist_train.sh. Learning rate 5e-5 = MRA's 2e-4 / 4 (linear scaling from 4 GPUs x 1 to 1 x 1).
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
# Initial weights: MRA's own, i.e. ImageNet ResNet-50 from `pretrained` in MRA-T and the default init for the rest.
# Optional SGN-T warm start instead:  --cfg-options load_from=ckpts/warm_start.pth   (see kaggle/make_warm_start.py)
work_dir = '/kaggle/working/runs/raca_full'

# Early stopping on the validation mIoU (validation on the 150 val frames each epoch, ~5 min; VALIDATE=0 in train.sh disables it).
# Needs validation on (train without --no-validate). The best checkpoint is kept as best_semkitti_SSC_mIoU_epoch_N.pth.
evaluation = dict(interval=1, save_best='semkitti_SSC_mIoU', rule='greater')
custom_hooks = [dict(type='EarlyStoppingHook', monitor='semkitti_SSC_mIoU', rule='greater',
                     patience=2,        # stop after this many evaluations without improvement
                     min_delta=0.0,     # improvement smaller than this (in mIoU points) does not count
                     min_epochs=3,      # never stop before this many epochs
                     priority=80)]      # after the evaluation hook (70), before the logger hooks (90)

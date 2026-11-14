_base_ = ['../../kaggle/configs/raca_full.py']
# molab run on the FULL SemanticKITTI (3,834 train / 815 val frames; kaggle/build_full_dataset.ipynb, compact format read
# directly by the loader). Same recipe as the subset runs (SGN-T warm start, lr 5e-5, batch 1, seed 0), except:
#   - RaCA full method C1 + C2 + C3 + C4, fixed number of epochs (no early stopping): both methods get the same schedule,
#   - validation every 2 epochs (815 frames, ~6 min each); the best checkpoint by mIoU is kept.
# Change the length with --cfg-options runner.max_epochs=N total_epochs=N (MRA's paper: 24).
total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(interval=2, save_best='semkitti_SSC_mIoU', rule='greater')
custom_hooks = []
work_dir = '/tmp/mra_runs/raca_fullkitti'

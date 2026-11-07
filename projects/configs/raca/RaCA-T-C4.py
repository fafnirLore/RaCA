_base_ = ['../mra/MRA-T.py']
work_dir = './validate_raca/kitti-T-c4'

# Ablation: C4 only (MRA's sampler and KL).
# All MRA-T settings (data, optimiser, schedule, grad clip) are inherited; only the head type and its RaCA options are set here.
model = dict(
    pts_bbox_head=dict(
        type='RaCAHead',
        # ---- C1: depth-axis scan (ScanSSC Scan block + Scan loss, depth axis only); changes inference ----
        use_c1=False,
        c1_fine=True,              # scan the fine grid (128 x 128 x 16)
        c1_coarse=True,             # scan the coarse grid (64 x 64 x 8); False = R1h (fine grid only)
        c1_heads=4,                # attention heads (64 channels -> 16 per head)
        c1_margin=0.5,             # nearest share of each line that every voxel may attend (ScanSSC: 50 %)
        c1_pos_embed=False,        # learned positional embedding along the depth axis (ScanSSC: none)
        c1_zero_init=True,         # Scan block starts as the identity (safe for warm starts)
        c1_loss_coarse=True,       # also apply the scan loss to the coarse-grid logits
        lambda_scan=1.0,           # weight of the scan loss on each grid (ScanSSC: 1)
        # ---- C2 / C3 / C4: training-only changes to the critical-voxel alignment ----
        use_c2=False,               # C2: stratified, shared critical voxels (both grids)
        use_c3=False,               # C3: co-located directed alignment (needs C2)
        use_c4=True,               # C4: rare-preserving coarse labels / CSA, max-pooled coarse proposal
        c2_shares=(0.25, 0.25, 0.40, 0.10),   # foreground, far background, rest, uniform random (of N=4096)
        c2_rare_share=0.5,         # share of the foreground budget reserved for rare classes (when present)
        c2_far_x=128,              # 'far' = forward voxel index >= 128, i.e. beyond 25.6 m
        lambda_align=1.0,          # weight of the C3 alignment loss
        w_mining=1.0,              # weight of the fine-grid mining (CE) loss on critical voxels
        w_mining_down1=1.0,        # weight of the coarse-grid mining (CE) loss on critical voxels
    ))

# Early stopping on the validation mIoU (RaCA addition; MRA-T validates every epoch but never stops early).
# Needs validation on (train without --no-validate). The best checkpoint is kept as best_semkitti_SSC_mIoU_epoch_N.pth.
evaluation = dict(interval=1, save_best='semkitti_SSC_mIoU', rule='greater')
custom_hooks = [dict(type='EarlyStoppingHook', monitor='semkitti_SSC_mIoU', rule='greater',
                     patience=5,        # stop after this many evaluations without improvement
                     min_delta=0.0,     # improvement smaller than this (in mIoU points) does not count
                     min_epochs=10,      # never stop before this many epochs
                     priority=80)]      # after the evaluation hook (70), before the logger hooks (90)

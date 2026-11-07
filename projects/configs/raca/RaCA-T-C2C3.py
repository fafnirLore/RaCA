_base_ = ['../mra/MRA-T.py']
work_dir = './validate_raca/kitti-T-c2c3'

# RaCA-T: MRA-T (data, optimiser, schedule, grad clip) with the RaCA head. Ablation: C2 + C3 (no C4).
# All MRA-T settings are inherited; only the head type and its RaCA options are set here.
model = dict(
    pts_bbox_head=dict(
        type='RaCAHead',
        use_c2=True,               # C2: stratified, shared critical voxels (both grids)
        use_c3=True,               # C3: co-located directed alignment (needs C2)
        use_c4=False,               # C4: rare-preserving coarse labels / CSA, max-pooled coarse proposal
        c2_shares=(0.25, 0.25, 0.40, 0.10),   # foreground, far background, rest, uniform random (of N=4096)
        c2_rare_share=0.5,         # share of the foreground budget reserved for rare classes (when present)
        c2_far_x=128,              # 'far' = forward voxel index >= 128, i.e. beyond 25.6 m
        lambda_align=1.0,          # weight of the C3 alignment loss
        w_mining=1.0,              # weight of the fine-grid mining (CE) loss on critical voxels
        w_mining_down1=1.0,        # weight of the coarse-grid mining (CE) loss on critical voxels
    ))

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Jianbiao Mei
# ---------------------------------------------
#  Modified by Zhiwen Yang
# ---------------------------------------------
#  Modified by Farquleet Farhat Gondal and Daniyal Imran (RaCA: C1 / C2 / C3 / C4)
# ---------------------------------------------

import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, builder
from projects.mmdet3d_plugin.mra.utils.header import Header, SparseHeader
from projects.mmdet3d_plugin.mra.modules.sgb import SGB
from projects.mmdet3d_plugin.mra.modules.sdb import SDB
from projects.mmdet3d_plugin.mra.modules.flosp import FLoSP
from projects.mmdet3d_plugin.mra.utils.lovasz_losses import lovasz_softmax
from projects.mmdet3d_plugin.mra.utils.ssc_loss import sem_scal_loss, geo_scal_loss, CE_ssc_loss, distill_ssc_loss

from projects.mmdet3d_plugin.mra.utils.cda_header import CriticalDisttributionAlignmentHead, voxel_sample
from projects.mmdet3d_plugin.mra.utils.ssc_metric_torch import SSCMetrics
from projects.mmdet3d_plugin.mra.utils.raca_utils import (stratified_critical_voxels, coarse_labels_rare_preserving,
                                                          pool_proposal, directed_alignment,
                                                          DepthScanBlock, scan_loss_depth)

class SharedCDAHead(CriticalDisttributionAlignmentHead):
    """MRA's critical-distribution-alignment head, but able to read a given set of locations (C2: shared by both grids).
    Same parameters as the original, so MRA / SGN weights load unchanged."""
    def forward(self, x, voxel_feature, coords=None):
        if coords is None:
            return super().forward(x, voxel_feature)
        output_dict = {}
        coarse_prediction = self.occ_conv(voxel_feature)
        output_dict["ssc_logit"] = F.interpolate(coarse_prediction, scale_factor=2, mode="trilinear",
                                                 align_corners=False).contiguous()
        if not self.training:
            return output_dict
        sampled_coarse_pred = voxel_sample(coarse_prediction, coords, align_corners=False)
        sampled_voxel_feature = voxel_sample(voxel_feature, coords, align_corners=False)
        output_dict["refined_pred"] = self.refined_mlp(torch.cat([sampled_coarse_pred, sampled_voxel_feature], dim=1))
        output_dict["sampled_voxel_coords"] = coords
        return output_dict


@HEADS.register_module()
class RaCAHead(nn.Module):
    """RaCA head = MRA head + C1 (depth-axis scan, after ScanSSC), C2 (stratified shared critical voxels),
    C3 (co-located directed alignment), C4 (rare-preserving coarse supervision).

    With use_c1 = use_c2 = use_c3 = use_c4 = False and w_mining = w_mining_down1 = 1.0 it is numerically MRAHead.
    C2-C4 add no parameters, so MRA / SGN checkpoints load unchanged; C1 adds one Scan block per scanned grid
    (new parameters, zero-initialised to the identity by default so a warm start still begins as MRA)."""
    def __init__(
        self,
        *args,
        bev_h,
        bev_w,
        bev_z,
        embed_dims,
        scale_2d_list,
        pts_header_dict,
        depth=3,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        save_flag = False,
        use_c1=False,
        use_c2=True,
        use_c3=True,
        use_c4=True,
        c1_fine=True,
        c1_coarse=True,
        c1_heads=4,
        c1_margin=0.5,
        c1_pos_embed=False,
        c1_zero_init=True,
        c1_loss_coarse=True,
        lambda_scan=1.0,
        c2_shares=(0.25, 0.25, 0.40, 0.10),
        c2_rare_share=0.5,
        c2_far_x=128,
        lambda_align=1.0,
        w_mining=1.0,
        w_mining_down1=1.0,
        **kwargs
    ):
        super().__init__()
        assert not (use_c3 and not use_c2), 'C3 needs the shared locations of C2'
        self.use_c1, self.use_c2, self.use_c3, self.use_c4 = use_c1, use_c2, use_c3, use_c4
        self.c1_fine, self.c1_coarse, self.c1_loss_coarse, self.lambda_scan = c1_fine, c1_coarse, c1_loss_coarse, lambda_scan
        self.c2_shares, self.c2_rare_share, self.c2_far_x = tuple(c2_shares), c2_rare_share, c2_far_x
        self.lambda_align = lambda_align
        self.w_mining, self.w_mining_down1 = w_mining, w_mining_down1      # weights of the two mining (CE) losses
        self.bev_h = bev_h
        self.bev_w = bev_w 
        self.bev_z = bev_z
        self.real_w = 51.2
        self.real_h = 51.2
        self.embed_dims = embed_dims
        
        self.alpha = 0.5
        self.beta = 1.0
        self.lamda = 48
        self.delta = 0.1
        self.count = 0

        if kwargs.get('dataset', 'semantickitti') == 'semantickitti':
            self.class_names =  [ "empty", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist", "motorcyclist", "road", 
                                "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",]
            self.class_weights = torch.from_numpy(np.array([0.446, 0.603, 0.852, 0.856, 0.747, 0.734, 0.801, 0.796, 0.818, 0.557, 0.653, 0.568, 0.683, 0.560, 0.603, 0.530, 0.688, 0.574, 0.716, 0.786]))
        elif kwargs.get('dataset', 'semantickitti') == 'kitti360':
            self.class_names =  ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'road',
         'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'terrain',
         'pole', 'traffic-sign', 'other-structure', 'other-object']
            self.class_weights = torch.from_numpy(np.array([0.464, 0.595, 0.865, 0.871, 0.717, 0.657, 0.852, 0.541, 0.602, 0.567, 0.607, 0.540, 0.636, 0.513, 0.564, 0.701, 0.774, 0.580, 0.690]))
        self.n_classes = len(self.class_names)

        self.flosp = FLoSP(scale_2d_list)
        self.bottleneck = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1)
        self.bottleneck_down1 = nn.Conv3d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1)
        self.sgb = SGB(sizes=[self.bev_h, self.bev_w, self.bev_z], channels=self.embed_dims)
        self.sgb_down1 = SGB(sizes=[self.bev_h//2, self.bev_w//2, self.bev_z//2], channels=self.embed_dims)
        self.mlp_prior = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//2),
            nn.LayerNorm(self.embed_dims//2),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dims//2, self.embed_dims)
        )
        self.mlp_prior_down1 = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims//2),
            nn.LayerNorm(self.embed_dims//2),
            nn.LeakyReLU(),
            nn.Linear(self.embed_dims//2, self.embed_dims)
        )

        occ_channel = 8 if pts_header_dict.get('guidance', False) else 0
        self.sdb = SDB(channel=self.embed_dims+occ_channel, out_channel=self.embed_dims//2, depth=depth)
        self.sdb_down1 = SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=depth)
        
        self.occ_header = nn.Sequential(
            SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=1),
            nn.Conv3d(self.embed_dims//2, 1, kernel_size=3, padding=1)
        )
        self.occ_header_down1 = nn.Sequential(
            SDB(channel=self.embed_dims, out_channel=self.embed_dims//2, depth=1),
            nn.Conv3d(self.embed_dims//2, 1, kernel_size=3, padding=1)
        )
        self.sem_header = SparseHeader(self.n_classes, feature=self.embed_dims)
        self.sem_header_down1 = SparseHeader(self.n_classes, feature=self.embed_dims)

        self.pts_header = builder.build_head(pts_header_dict)
        
        self.cda_header = SharedCDAHead(in_channel=self.embed_dims//2+self.n_classes, num_classes=self.n_classes)
        self.cda_header_down1 = SharedCDAHead(in_channel=self.embed_dims//2+self.n_classes, num_classes=self.n_classes)

        self.interpolate_conv = nn.Conv3d(self.embed_dims, self.embed_dims//2, kernel_size=3, padding=1)

        if self.use_c1:                                                                              # C1
            scan_kw = dict(heads=c1_heads, margin=c1_margin, pos_embed=c1_pos_embed, zero_init=c1_zero_init)
            self.scan = DepthScanBlock(self.embed_dims//2, self.bev_h, **scan_kw) if c1_fine else None
            self.scan_down1 = DepthScanBlock(self.embed_dims//2, self.bev_h//2, **scan_kw) if c1_coarse else None

        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.ssc_metric = SSCMetrics(self.class_names)
        self.save_flag = save_flag
        
    def forward(self, mlvl_feats, img_metas, target):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
        Returns:
            ssc_logit (Tensor): Outputs from the segmentation head.
        """

        # Multi-resolution View Trnasformation

        out = {}
        x3d = self.flosp(mlvl_feats, img_metas) # bs, c, nq
        x3d_down1 = self.flosp(mlvl_feats, img_metas, down_scale=2)
        bs, c, _ = x3d.shape
        x3d = self.bottleneck(x3d.reshape(bs, c, self.bev_h, self.bev_w, self.bev_z))
        occ = self.occ_header(x3d).squeeze(1)
        out["occ"] = occ

        x3d_down1 = self.bottleneck_down1(x3d_down1.reshape(bs, c, self.bev_h//2, self.bev_w//2, self.bev_z//2))
        occ_down1 = self.occ_header_down1(x3d_down1).squeeze(1)
        out["occ_down1"] = occ_down1

        x3d = x3d.reshape(bs, c, -1)
        x3d_down1 = x3d_down1.reshape(bs, c, -1)
        # Load proposals
        pts_out = self.pts_header(mlvl_feats, img_metas, target)
        pts_occ = pts_out['occ_logit'].squeeze(1)
        proposal =  (pts_occ > 0).float().detach().cpu().numpy()
        proposal_down1 = pool_proposal(proposal) if self.use_c4 else proposal[:, ::2, ::2, ::2]   # C4
        out['pts_occ'] = pts_occ

        if proposal.sum() < 2:
            proposal = np.ones_like(proposal)
        unmasked_idx = np.asarray(np.where(proposal.reshape(-1)>0)).astype(np.int32)
        masked_idx = np.asarray(np.where(proposal.reshape(-1)==0)).astype(np.int32)
        vox_coords = self.get_voxel_indices()

        if proposal_down1.sum() < 2:
            proposal_down1 = np.ones_like(proposal_down1)
        unmasked_idx_down1 = np.asarray(np.where(proposal_down1.reshape(-1)>0)).astype(np.int32)
        masked_idx_down1 = np.asarray(np.where(proposal_down1.reshape(-1)==0)).astype(np.int32)
        vox_coords_down1 = self.get_voxel_indices(down_scale=2)

        # Compute seed features
        seed_feats = x3d[0, :, vox_coords[unmasked_idx[0], 3]].permute(1, 0)
        seed_coords = vox_coords[unmasked_idx[0], :3]
        coords_torch = torch.from_numpy(np.concatenate(
            [np.zeros_like(seed_coords[:, :1]), seed_coords], axis=1)).to(seed_feats.device)
        seed_feats_desc = self.sgb(seed_feats, coords_torch)
        sem = self.sem_header(seed_feats_desc)
        out["sem_logit"] = sem
        out["coords"] = seed_coords

        seed_feats_down1 = x3d_down1[0, :, vox_coords_down1[unmasked_idx_down1[0], 3]].permute(1, 0)
        seed_coords_down1 = vox_coords_down1[unmasked_idx_down1[0], :3]
        coords_torch_down1 = torch.from_numpy(np.concatenate(
            [np.zeros_like(seed_coords_down1[:, :1]), seed_coords_down1], axis=1)).to(seed_feats_down1.device)
        seed_feats_desc_down1 = self.sgb_down1(seed_feats_down1, coords_torch_down1)
        sem_down1 = self.sem_header_down1(seed_feats_desc_down1)
        out["sem_logit_down1"] = sem_down1
        out["coords_down1"] = seed_coords_down1

        # Complete voxel features
        vox_feats = torch.empty((self.bev_h, self.bev_w, self.bev_z, self.embed_dims), device=x3d.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx[0], 3], :] = seed_feats_desc
        seed_feats = vox_feats_flatten.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0)
        vox_feats_flatten[vox_coords[masked_idx[0], 3], :] = self.mlp_prior(x3d[0, :, vox_coords[masked_idx[0], 3]].permute(1, 0))

        vox_feats_down1 = torch.empty((self.bev_h//2, self.bev_w//2, self.bev_z//2, self.embed_dims), device=x3d_down1.device)
        vox_feats_flatten_down1 = vox_feats_down1.reshape(-1, self.embed_dims)
        vox_feats_flatten_down1[vox_coords_down1[unmasked_idx_down1[0], 3], :] = seed_feats_desc_down1
        seed_feats_down1 = vox_feats_flatten_down1.reshape(self.bev_h//2, self.bev_w//2, self.bev_z//2, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0)
        vox_feats_flatten_down1[vox_coords_down1[masked_idx_down1[0], 3], :] = self.mlp_prior_down1(x3d_down1[0, :, vox_coords_down1[masked_idx_down1[0], 3]].permute(1, 0))

        vox_feats_diff = vox_feats_flatten.reshape(self.bev_h, self.bev_w, self.bev_z, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0)
        vox_feats_diff_down1 = vox_feats_flatten_down1.reshape(self.bev_h//2, self.bev_w//2, self.bev_z//2, self.embed_dims).permute(3, 0, 1, 2).unsqueeze(0)
        vox_feats_diff = vox_feats_diff + F.interpolate(seed_feats_down1, size=vox_feats_diff.shape[-3:], mode='trilinear', align_corners=True)
        vox_feats_diff_down1 = vox_feats_diff_down1 + F.interpolate(seed_feats, size=vox_feats_diff_down1.shape[-3:], mode='trilinear', align_corners=True)
        if self.pts_header.guidance:
            vox_feats_diff = torch.cat([vox_feats_diff, pts_out['occ_x']], dim=1)
        vox_feats_diff = self.sdb(vox_feats_diff) # 1, C,H,W,Z
        vox_feats_diff_down1 = self.sdb_down1(vox_feats_diff_down1)

        if self.use_c1:                       # C1: depth-axis scan after propagation, before the fusion of the grids
            if self.scan is not None:
                vox_feats_diff = self.scan(vox_feats_diff)
            if self.scan_down1 is not None:
                vox_feats_diff_down1 = self.scan_down1(vox_feats_diff_down1)

        # SSC Head with Critical Distribution Alignment

        target_down1 = target[:, ::2, ::2, ::2]
        shared = None
        if self.training and self.use_c2:                                                          # C2
            shared, stratum = stratified_critical_voxels(target, self.cda_header.N, self.c2_shares,
                                                         self.c2_rare_share, self.c2_far_x)
            out['sampling_stratum'] = stratum
        cda_dict_down1 = self.cda_header_down1(target_down1, vox_feats_diff_down1, shared)

        vox_feats_diff_down1 = F.interpolate(vox_feats_diff_down1, size=vox_feats_diff.shape[-3:], mode='trilinear', align_corners=True)
        vox_feats_diff = self.interpolate_conv(torch.cat([vox_feats_diff, vox_feats_diff_down1], dim=1))

        cda_dict = self.cda_header(target, vox_feats_diff, shared)

        out.update(cda_dict)
        out['ssc_logit_down1'] = cda_dict_down1['ssc_logit']
        if self.training:
            out['refined_pred_down1'] = cda_dict_down1['refined_pred']
            out['sampled_voxel_coords_down1'] = cda_dict_down1['sampled_voxel_coords']
        
        return out

    def step(self, out_dict, teacher_out_dict, target, img_metas, csa, step_type):
        """Training/validation function.
        Args:
            out_dict (dict[Tensor]): Segmentation output.
            img_metas: Meta information such as camera intrinsics.
            target: Semantic completion ground truth. 
            step_type: Train or test.
        Returns:
            loss or predictions
        """

        ssc_pred = out_dict["ssc_logit"]

        if step_type== "train":
            sem_pred_2 = out_dict["sem_logit"]

            csa = self.alpha * csa + self.beta
            ssc_pred_down1 = out_dict["ssc_logit_down1"]
            if self.use_c4:                                                                          # C4
                target_down1 = coarse_labels_rare_preserving(target)
                csa_down1 = F.max_pool3d(csa.unsqueeze(1).float(), 2).squeeze(1)
            else:
                target_down1 = target[:, ::2, ::2, ::2]
                csa_down1 = csa[:, ::2, ::2, ::2]

            target_2 = torch.from_numpy(img_metas[0]['target_1_2']).unsqueeze(0).to(target.device)
            coords = out_dict['coords']
            sp_target_2 = target_2.clone()[0, coords[:, 0], coords[:, 1], coords[:, 2]]
            loss_dict = dict()

            class_weight = self.class_weights.type_as(target)
            if self.CE_ssc_loss:
                loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight, csa)
                loss_dict['loss_ssc'] = loss_ssc

                loss_ssc_down1 = CE_ssc_loss(ssc_pred_down1, target_down1, class_weight, csa_down1)
                loss_dict['loss_ssc_down1'] = loss_ssc_down1

            if self.sem_scal_loss:
                loss_sem_scal = sem_scal_loss(ssc_pred, target)
                loss_dict['loss_sem_scal'] = loss_sem_scal

                loss_sem_scal_down1 = sem_scal_loss(ssc_pred_down1, target_down1)
                loss_dict['loss_sem_scal_down1'] = loss_sem_scal_down1

            if self.geo_scal_loss:
                loss_geo_scal = geo_scal_loss(ssc_pred, target)
                loss_dict['loss_geo_scal'] = loss_geo_scal

                loss_geo_scal_down1 = geo_scal_loss(ssc_pred_down1, target_down1)
                loss_dict['loss_geo_scal_down1'] = loss_geo_scal_down1

            if self.use_c1 and self.lambda_scan > 0:                                                 # C1 scan loss
                loss_dict['loss_scan'] = scan_loss_depth(ssc_pred, target) * self.lambda_scan
                if self.c1_coarse and self.c1_loss_coarse:
                    loss_dict['loss_scan_down1'] = scan_loss_depth(ssc_pred_down1, target_down1) * self.lambda_scan
            
            refined_pred = out_dict["refined_pred"]
            sampled_voxel_coords = out_dict["sampled_voxel_coords"]
            gt_voxels = voxel_sample(
                target.float().unsqueeze(1),
                sampled_voxel_coords,
                mode="nearest",
                align_corners=False
            ).squeeze_(1).long()

            csa_voxels = voxel_sample(
                csa.float().unsqueeze(1),
                sampled_voxel_coords,
                mode="nearest",
                align_corners=False
            ).squeeze_(1).long()

            ce_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
            cda_loss = ce_criterion(refined_pred, gt_voxels)

            local_hardness = csa_voxels
            cda_loss = cda_loss * local_hardness

            flatten_targets = gt_voxels.flatten()
            valid_mask = flatten_targets != 255
            class_weights = self.class_weights.type_as(ssc_pred)
            norm_weights = class_weights[flatten_targets[valid_mask]]

            cda_loss = cda_loss.flatten()[valid_mask].sum() / norm_weights.sum()
            loss_dict['loss_hard_voxel_mining'] = cda_loss * self.w_mining

            refined_pred_down1 = out_dict["refined_pred_down1"]
            sampled_voxel_coords_down1 = out_dict["sampled_voxel_coords_down1"]
            gt_voxels_down1 = voxel_sample(
                target_down1.float().unsqueeze(1),
                sampled_voxel_coords_down1,
                mode="nearest",
                align_corners=False
            ).squeeze_(1).long()

            csa_voxels_down1 = voxel_sample(
                csa_down1.float().unsqueeze(1),
                sampled_voxel_coords_down1,
                mode="nearest",
                align_corners=False
            ).squeeze_(1).long()

            ce_criterion_down1 = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
            cda_loss_down1 = ce_criterion_down1(refined_pred_down1, gt_voxels_down1)

            local_hardness_down1 = csa_voxels_down1
            cda_loss_down1 = cda_loss_down1 * local_hardness_down1

            flatten_targets_down1 = gt_voxels_down1.flatten()
            valid_mask_down1 = flatten_targets_down1 != 255
            class_weights_down1 = self.class_weights.type_as(ssc_pred_down1)
            norm_weights_down1 = class_weights_down1[flatten_targets_down1[valid_mask_down1]]

            cda_loss_down1 = cda_loss_down1.flatten()[valid_mask_down1].sum() / norm_weights_down1.sum()
            loss_dict['loss_hard_voxel_mining_down1'] = cda_loss_down1 * self.w_mining_down1

            cda_kl_loss = 0 if self.use_c3 else F.kl_div(F.log_softmax(refined_pred, dim=-1), F.softmax(refined_pred_down1, dim=-1), reduction='none').sum(-1).mean() \
                        + F.kl_div(F.log_softmax(refined_pred_down1, dim=-1), F.softmax(refined_pred, dim=-1), reduction='none').sum(-1).mean()
            if self.use_c3:                                                                          # C3
                loss_align, fine_lead_share = directed_alignment(refined_pred, refined_pred_down1, out_dict['sampling_stratum'])
                loss_dict['loss_align_directed'] = loss_align * self.lambda_align
                loss_dict['fine_leads_share'] = fine_lead_share.detach()
            else:
                loss_dict['cda_kl_loss'] = cda_kl_loss

            if teacher_out_dict is not None:
                teacher_ssc_pred = teacher_out_dict["ssc_logit"]
                output_voxels_tmp = ssc_pred.clone().detach()
                teacher_voxels_tmp = teacher_ssc_pred.clone().detach()
                target_voxels_tmp = target.clone().detach()
                output_voxels_tmp = torch.argmax(output_voxels_tmp, dim=1)
                teacher_voxels_tmp = torch.argmax(teacher_voxels_tmp, dim=1)
                mask = target_voxels_tmp != 255
                tp, fp, fn = self.ssc_metric.get_score_completion(output_voxels_tmp, target_voxels_tmp, mask)
                tp_sum, fp_sum, fn_sum = self.ssc_metric.get_score_semantic_and_completion(output_voxels_tmp,
                                                                                        target_voxels_tmp, mask)
                sc_iou = tp / (tp + fp + fn)
                ssc_iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-5)
                ssc_miou = ssc_iou[1:].mean()

                loss_dict['sc_iou'] = sc_iou
                loss_dict['ssc_miou'] = ssc_miou

                tp, fp, fn = self.ssc_metric.get_score_completion(teacher_voxels_tmp, target_voxels_tmp, mask)
                tp_sum, fp_sum, fn_sum = self.ssc_metric.get_score_semantic_and_completion(teacher_voxels_tmp,
                                                                                        target_voxels_tmp, mask)
                teacher_sc_iou = tp / (tp + fp + fn)
                teacher_ssc_iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-5)
                teacher_ssc_miou = teacher_ssc_iou[1:].mean()

                loss_dict['teacher_sc_iou'] = teacher_sc_iou
                loss_dict['teacher_ssc_miou'] = teacher_ssc_miou

                # compute self-distillation loss
                distill_mask = (target != 255)
                loss_logit_distill = distill_ssc_loss(ssc_pred, teacher_ssc_pred, distill_mask)
                dynamic_weight = math.exp(teacher_ssc_miou)
                loss_dict['loss_logit_distill'] = loss_logit_distill * dynamic_weight * self.lamda

                # compute hard voxel mining loss in the teacher brunch
                teacher_sampled_voxel_coords = teacher_out_dict["sampled_voxel_coords"]
                student_corr_pred = voxel_sample(
                    ssc_pred,
                    teacher_sampled_voxel_coords,
                    mode="nearest",
                    align_corners=False
                ).squeeze_(1)

                teacher_gt_voxels = voxel_sample(
                    target.float().unsqueeze(1),
                    teacher_sampled_voxel_coords,
                    mode="nearest",
                    align_corners=False
                ).squeeze_(1).long()

                teacher_csa_voxels = voxel_sample(
                    csa.float().unsqueeze(1),
                    teacher_sampled_voxel_coords,
                    mode="nearest",
                    align_corners=False
                ).squeeze_(1).long()

                teacher_ce_criterion = nn.CrossEntropyLoss(ignore_index=255, reduction="none")
                teacher_cda_loss = teacher_ce_criterion(student_corr_pred, teacher_gt_voxels)

                # teacher_local_hardness = self.alpha + self.beta * teacher_csa_voxels
                teacher_local_hardness = teacher_csa_voxels
                teacher_cda_loss = teacher_cda_loss * teacher_local_hardness

                flatten_targets = teacher_csa_voxels.flatten()
                valid_mask = flatten_targets != 255
                class_weights = self.class_weights.type_as(ssc_pred)
                norm_weights = class_weights[flatten_targets[valid_mask]]
                teacher_cda_loss = teacher_cda_loss.flatten()[valid_mask].sum() / norm_weights.sum()
                loss_dict['loss_hard_voxel_mining_teacher'] = teacher_cda_loss * self.delta
                self.count += 1

            loss_sem = lovasz_softmax(F.softmax(sem_pred_2, dim=1), sp_target_2, ignore=255)
            loss_sem += F.cross_entropy(sem_pred_2, sp_target_2.long(), ignore_index=255)
            loss_dict['loss_sem'] = loss_sem

            ones = torch.ones_like(target_2).to(target_2.device)
            target_2_binary = torch.where(torch.logical_or(target_2==255, target_2==0), target_2, ones)
            loss_occ = F.binary_cross_entropy(out_dict['occ'].sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())
            loss_dict['loss_occ'] = loss_occ

            loss_dict['loss_pts'] = F.binary_cross_entropy(out_dict['pts_occ'].sigmoid()[target_2_binary!=255], target_2_binary[target_2_binary!=255].float())

            return loss_dict

        elif step_type== "val" or "test":
            result = dict()
            result['output_voxels'] = ssc_pred
            result['target_voxels'] = target

            if self.save_flag:
                y_pred = ssc_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis=1)
                self.save_pred(img_metas, y_pred)

            return result

    def training_step(self, out_dict, target, img_metas, csa, teacher_out_dict=None):
        """Training step.
        """
        return self.step(out_dict, teacher_out_dict, target, img_metas, csa, "train")

    def validation_step(self, out_dict, target, img_metas):
        """Validation step.
        """
        return self.step(out_dict, None, target, img_metas, None, "val")

    def get_voxel_indices(self, down_scale=1):
        """Get reference points in 3D.
        Args:
            self.real_h, self.bev_h
        Returns:
            vox_coords (Array): Voxel indices
        """
        scene_size = (51.2, 51.2, 6.4)
        vox_origin = np.array([0, -25.6, -2])
        voxel_size = self.real_h / (self.bev_h // down_scale)

        vol_bnds = np.zeros((3,2))
        vol_bnds[:,0] = vox_origin
        vol_bnds[:,1] = vox_origin + np.array(scene_size)

        # Compute the voxels index in lidar cooridnates
        vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
        idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
        xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
        vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1), idx], axis=0).astype(int).T

        return vox_coords

    def save_pred(self, img_metas, y_pred):
        """Save predictions for evaluations and visualizations.

        learning_map_inv: inverse of previous map
        
        0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
        5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
        10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
        15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
        Note: only for semantickitti
        """

        y_pred[y_pred==10] = 44
        y_pred[y_pred==11] = 48
        y_pred[y_pred==12] = 49
        y_pred[y_pred==13] = 50
        y_pred[y_pred==14] = 51
        y_pred[y_pred==15] = 70
        y_pred[y_pred==16] = 71
        y_pred[y_pred==17] = 72
        y_pred[y_pred==18] = 80
        y_pred[y_pred==19] = 81
        y_pred[y_pred==1] = 10
        y_pred[y_pred==2] = 11
        y_pred[y_pred==3] = 15
        y_pred[y_pred==4] = 18
        y_pred[y_pred==5] = 20
        y_pred[y_pred==6] = 30
        y_pred[y_pred==7] = 31
        y_pred[y_pred==8] = 32
        y_pred[y_pred==9] = 40

        # save predictions
        pred_folder = os.path.join("./mra", "sequences", img_metas[0]['sequence_id'], "predictions") 
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        y_pred_bin = y_pred.astype(np.uint16)
        y_pred_bin.tofile(os.path.join(pred_folder, img_metas[0]['frame_id'] + ".label"))

import os
import torch
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda:0')

import cv2
import sys
import json
import time
import glob
import pickle
import numpy as np
from torch import optim
from argparse import ArgumentParser
import torch.nn.functional as torch_F
from torchvision.ops import roi_align
from pytorch3d import ops as py3d_ops
from pytorch3d import transforms as py3d_transform
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

L1Loss = torch.nn.L1Loss(reduction='mean')
MSELoss = torch.nn.MSELoss(reduction='mean')
SSIM_METRIC = SSIM(data_range=1, size_average=True, channel=3) # channel=1 for grayscale images
MS_SSIM_METRIC = MS_SSIM(data_range=1, size_average=True, channel=3)

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJ_ROOT)

from dataset import misc
from misc_utils import gs_utils
from misc_utils.metric_utils import *
from config import inference_cfg as CFG
from model.network import model_arch as ModelNet
from dataset.inference_datasets import datasetCallbacks
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts

# import original gaussian splatting modules
# sub_module_dir = os.path.join(PROJ_ROOT, 'gaussian-splatting')
# sub_module_dir = os.path.join('/home/dingding/Workspace/CDD/gaussian-splatting')

# sys.path.append(sub_module_dir)

# import the customized modules
from gaussian_object.utils.graphics_utils import focal2fov
from gaussian_object.gaussian_render import render as GS_Renderer
from gaussian_object.cameras import Camera as GS_Camera
from gaussian_object.gaussian_model import GaussianModel
from gaussian_object.arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_object.build_3DGaussianObject import create_3D_Gaussian_object

parser = ArgumentParser()
gaussian_ModelP = ModelParams(parser)
gaussian_PipeP = PipelineParams(parser)
gaussian_OptimP = OptimizationParams(parser)
gaussian_BG = torch.zeros((3), device=device)

model_net = ModelNet().to(device)
ckpt_file = os.path.join(PROJ_ROOT, 'checkpoints/model_weights.pth')
model_net.load_state_dict(torch.load(ckpt_file, map_location=device))
print('Pretrained weights are loaded from ', ckpt_file.split('/')[-1])
model_net.eval()

def create_reference_database_from_RGB_images(model_func, obj_dataset, device, save_pred_mask=False):    
    if CFG.USE_ALLOCENTRIC:
        obj_poses = np.stack(obj_dataset.allo_poses, axis=0)
    else:
        obj_poses = np.stack(obj_dataset.poses, axis=0)

    obj_poses = torch.as_tensor(obj_poses, dtype=torch.float32).to(device)
    obj_matRs = obj_poses[:, :3, :3]
    # obj_vecRs = obj_matRs[:, 2, :3]
    obj_vecRs = py3d_transform.matrix_to_axis_angle(obj_matRs)
    fps_inds = py3d_ops.sample_farthest_points(
        obj_vecRs[None, :, :], K=CFG.refer_view_num, random_start_point=False)[1].squeeze(0)  # obtain the FPS indices
    ref_fps_images = list()
    ref_fps_poses = list()
    ref_fps_camKs = list()
    for ref_idx in fps_inds:
        view_idx = ref_idx.item()
        datum = obj_dataset[view_idx]
        camK = datum['camK']      # 3x3
        image = datum['image']    # HxWx3
        pose = datum.get('allo_pose', datum['pose']) # 4x4
        
        ref_fps_images.append(image)
        ref_fps_poses.append(pose)
        ref_fps_camKs.append(camK)
    ref_fps_poses = torch.stack(ref_fps_poses, dim=0)
    ref_fps_camKs = torch.stack(ref_fps_camKs, dim=0)
    ref_fps_images = torch.stack(ref_fps_images, dim=0)
    zoom_fps_images = gs_utils.zoom_in_and_crop_with_offset(image=ref_fps_images, # KxHxWx3 -> KxSxSx3
                                                                K=ref_fps_camKs, 
                                                                t=ref_fps_poses[:, :3, 3], 
                                                                radius=obj_dataset.bbox3d_diameter/2,
                                                                target_size=CFG.zoom_image_scale, 
                                                                margin=CFG.zoom_image_margin)['zoom_image']
    with torch.no_grad():
        if zoom_fps_images.shape[-1] == 3:
            zoom_fps_images = zoom_fps_images.permute(0, 3, 1, 2)
        obj_fps_feats, _, obj_fps_dino_tokens = model_func.extract_DINOv2_feature(zoom_fps_images.to(device), return_last_dino_feat=True) # Kx768x16x16
        obj_fps_masks = model_func.refer_cosegmentation(obj_fps_feats).sigmoid() # Kx1xSxS

        obj_token_masks = torch_F.interpolate(obj_fps_masks,
                                             scale_factor=1.0/model_func.dino_patch_size, 
                                             mode='bilinear', align_corners=True, recompute_scale_factor=True) # Kx1xS/14xS/14
        obj_fps_dino_tokens = obj_fps_dino_tokens.flatten(0, 1)[obj_token_masks.view(-1).round().type(torch.bool), :] # KxLxC -> KLxC -> MxC

    refer_allo_Rs = list()
    refer_pred_masks = list()
    refer_Remb_vectors = list()
    refer_coseg_mask_info = list()
    num_instances = len(obj_dataset)
    for idx in range(num_instances):
        ref_data = obj_dataset[idx]
        camK = ref_data['camK']
        image = ref_data['image']
        pose = ref_data.get('allo_pose', ref_data['pose']) # 4x4
        refer_allo_Rs.append(pose[:3, :3])
        ref_tz = (1 + CFG.zoom_image_margin) * camK[:2, :2].max() * obj_dataset.bbox3d_diameter / CFG.zoom_image_scale
        zoom_outp = gs_utils.zoom_in_and_crop_with_offset(image=image, # HxWx3 -> SxSx3
                                                            K=camK, 
                                                            t=pose[:3, 3], 
                                                            radius=obj_dataset.bbox3d_diameter/2,
                                                            target_size=CFG.zoom_image_scale, 
                                                            margin=CFG.zoom_image_margin) # SxSx3
        with torch.no_grad():
            zoom_image = zoom_outp['zoom_image'].unsqueeze(0)
            if zoom_image.shape[-1] == 3:
                zoom_image = zoom_image.permute(0, 3, 1, 2)
            zoom_feat = model_func.extract_DINOv2_feature(zoom_image.to(device))
            zoom_mask = model_func.query_cosegmentation(zoom_feat, 
                                                        x_ref=obj_fps_feats, 
                                                        ref_mask=obj_fps_masks).sigmoid()
            y_Remb = model_func.generate_rotation_aware_embedding(zoom_feat, zoom_mask)
            refer_Remb_vectors.append(y_Remb.squeeze(0)) # 64
            try:
                msk_yy, msk_xx = torch.nonzero(zoom_mask.detach().cpu().squeeze().round().type(torch.uint8), as_tuple=True)
                msk_cx = (msk_xx.max() + msk_xx.min()) / 2
                msk_cy = (msk_yy.max() + msk_yy.min()) / 2
            except: # no mask is found
                msk_cx = CFG.zoom_image_scale / 2
                msk_cy = CFG.zoom_image_scale / 2

            prob_mask_area = zoom_mask.detach().cpu().sum()
            bin_mask_area = zoom_mask.round().detach().cpu().sum()            
            refer_coseg_mask_info.append(torch.tensor([msk_cx, msk_cy, ref_tz, bin_mask_area, prob_mask_area]))

        if save_pred_mask:
            orig_mask = gs_utils.zoom_out_and_uncrop_image(zoom_mask.squeeze(), # SxS
                                                            bbox_center=zoom_outp['bbox_center'],
                                                            bbox_scale=zoom_outp['bbox_scale'],
                                                            orig_hei=image.shape[0],
                                                            orig_wid=image.shape[1],
                                                            )# 1xHxWx1
            coseg_mask_path = ref_data['coseg_mask_path']
            orig_mask = (orig_mask.detach().cpu().squeeze() * 255).numpy().astype(np.uint8) # HxW
            if not os.path.exists(os.path.dirname(coseg_mask_path)):
                os.makedirs(os.path.dirname(coseg_mask_path))
            cv2.imwrite(coseg_mask_path, orig_mask)
        else:
            refer_pred_masks.append(zoom_mask.detach().cpu().squeeze()) # SxS

        if (idx + 1) % 100 == 0:
            time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())
            print('[{}/{}], {}'.format(idx+1, num_instances, time_stamp))

    refer_allo_Rs = torch.stack(refer_allo_Rs, dim=0).squeeze() # Nx3x3
    refer_Remb_vectors = torch.stack(refer_Remb_vectors, dim=0).squeeze()      # Nx64
    refer_coseg_mask_info = torch.stack(refer_coseg_mask_info, dim=0).squeeze() # Nx3
    
    ref_database = dict()
    if not save_pred_mask:
        refer_pred_masks = torch.stack(refer_pred_masks, dim=0).squeeze() # NxSxS
        ref_database['refer_pred_masks'] = refer_pred_masks

    ref_database['obj_fps_inds'] = fps_inds
    ref_database['obj_fps_images'] = zoom_fps_images

    ref_database['obj_fps_feats'] = obj_fps_feats
    ref_database['obj_fps_masks'] = obj_fps_masks
    ref_database['obj_fps_dino_tokens'] = obj_fps_dino_tokens

    ref_database['refer_allo_Rs'] = refer_allo_Rs
    ref_database['refer_Remb_vectors'] = refer_Remb_vectors
    ref_database['refer_coseg_mask_info'] = refer_coseg_mask_info
    
    return ref_database
  
def perform_segmentation_and_encoding(model_func, que_image, ref_database, device):
    with torch.no_grad():
        start_timer = time.time()
        
        if que_image.dim() == 3:
            que_image = que_image.unsqueeze(0)
        if que_image.shape[-1] == 3:
            que_image = que_image.permute(0, 3, 1, 2)
        que_feats = model_func.extract_DINOv2_feature(que_image.to(device))    
        pd_coarse_mask = model_func.query_cosegmentation(x_que=que_feats, 
                                                        x_ref=ref_database['obj_fps_feats'], 
                                                        ref_mask=ref_database['obj_fps_masks'],
                                                        ).sigmoid() # 1xCxHxW -> 1x1xHxW
        mask_threshold = CFG.coarse_threshold
        while True:
            que_binary_mask = (pd_coarse_mask.squeeze() >= mask_threshold).type(torch.uint8)
            if que_binary_mask.sum() < CFG.DINO_PATCH_SIZE**2:
                mask_threshold -= 0.01
                continue
            else:
                break
        _, pd_coarse_tight_scales, pd_coarse_centers = misc.torch_find_connected_component(
            que_binary_mask, include_supmask=CFG.CC_INCLUDE_SUPMASK, min_bbox_scale=CFG.DINO_PATCH_SIZE, return_bbox=True)

        pd_coarse_scales = pd_coarse_tight_scales * CFG.coarse_bbox_padding
        pd_coarse_bboxes = torch.stack([pd_coarse_centers[:, 0] - pd_coarse_scales / 2.0,
                                        pd_coarse_centers[:, 1] - pd_coarse_scales / 2.0,
                                        pd_coarse_centers[:, 0] + pd_coarse_scales / 2.0,
                                        pd_coarse_centers[:, 1] + pd_coarse_scales / 2.0], dim=-1)
        roi_RGB_crops = roi_align(que_image, boxes=[pd_coarse_bboxes],
                                  output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale), 
                                  sampling_ratio=4) # 1x3xHxW -> Mx3xSxS
        
        if roi_RGB_crops.shape[0] == 1:
            rgb_img_crop = roi_RGB_crops  # 1x3xSxS 
            rgb_box_scale = pd_coarse_scales.squeeze(0)
            rgb_box_center = pd_coarse_centers.squeeze(0)
            rgb_box_tight_scale = pd_coarse_tight_scales.squeeze(0)
            rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
            rgb_crop_mask = model_func.query_cosegmentation(x_que=rgb_img_feat, 
                                                            x_ref=ref_database['obj_fps_feats'], 
                                                            ref_mask=ref_database['obj_fps_masks']).sigmoid()
        else:
            roi_img_feats, _, roi_dino_tokens = model_func.extract_DINOv2_feature(roi_RGB_crops, return_last_dino_feat=True)
            roi_img_masks = model_func.query_cosegmentation(x_que=roi_img_feats, 
                                                            x_ref=ref_database['obj_fps_feats'], 
                                                            ref_mask=ref_database['obj_fps_masks']).sigmoid() # KxCxSxS -> Kx1xSxS
            roi_obj_mask = torch_F.interpolate(roi_img_masks, 
                                                scale_factor=1.0/CFG.DINO_PATCH_SIZE, 
                                                mode='bilinear', align_corners=True, 
                                                recompute_scale_factor=True).flatten(2).permute(0, 2, 1).round() # Kx1xSxS -> KxLx1
            roi_dino_tokens = roi_obj_mask * roi_dino_tokens
            token_cosim = torch.einsum('klc,nc->kln', torch_F.normalize(roi_dino_tokens, dim=-1), 
                                                        torch_F.normalize(ref_database['obj_fps_dino_tokens'], dim=-1))
            if CFG.cosim_topk > 0:
                cosim_score = token_cosim.topk(dim=1, k=CFG.cosim_topk).values.mean(dim=-1).mean(dim=1)
            else:
                cosim_score = token_cosim.mean(dim=-1).sum(dim=1) / (1 + roi_obj_mask.squeeze(-1).sum(dim=1))  # KxLxN -> KxL -> K 
            
            optim_index = cosim_score.argmax()
            rgb_box_scale = pd_coarse_scales[optim_index]
            rgb_box_center = pd_coarse_centers[optim_index]
            rgb_box_tight_scale = pd_coarse_tight_scales[optim_index]
            
            rgb_img_crop = roi_RGB_crops[optim_index].unsqueeze(0)  # 1x3xSxS
            rgb_img_feat = roi_img_feats[optim_index].unsqueeze(0)  # 1xCxPxP
            rgb_crop_mask = roi_img_masks[optim_index].unsqueeze(0) # 1x1xSxS
        
        coarse_det_cost = time.time() - start_timer
        if CFG.enable_fine_detection:
            mask_threshold = CFG.finer_threshold
            while True:
                    fine_binary_mask = (rgb_crop_mask.squeeze() >= mask_threshold).type(torch.uint8)
                    if fine_binary_mask.sum() < CFG.DINO_PATCH_SIZE**2:
                        mask_threshold -= 0.1
                        continue
                    else:
                        break

            _, pd_fine_tight_scales, pd_fine_centers = misc.torch_find_connected_component(
                fine_binary_mask, include_supmask=CFG.CC_INCLUDE_SUPMASK, min_bbox_scale=CFG.DINO_PATCH_SIZE, return_bbox=True)

            fine_offset_center = (pd_fine_centers / CFG.zoom_image_scale - 0.5) * rgb_box_scale[None]
            fine_bbox_centers = rgb_box_center[None, :] + fine_offset_center
            fine_bbox_tight_scales = rgb_box_scale[None] * pd_fine_tight_scales / CFG.zoom_image_scale
            
            fine_bbox_scales = fine_bbox_tight_scales * CFG.finer_bbox_padding 
            pd_fine_bboxes = torch.stack([fine_bbox_centers[:, 0] - fine_bbox_scales / 2.0,
                                            fine_bbox_centers[:, 1] - fine_bbox_scales / 2.0,
                                            fine_bbox_centers[:, 0] + fine_bbox_scales / 2.0,
                                            fine_bbox_centers[:, 1] + fine_bbox_scales / 2.0], dim=-1)
            roi_RGB_crops = roi_align(que_image, boxes=[pd_fine_bboxes],
                                        output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale), 
                                        sampling_ratio=4) # 1x3xHxW -> Mx3xSxS
            
            if roi_RGB_crops.shape[0] == 1: 
                rgb_img_crop = roi_RGB_crops  # 1x3xSxS
                rgb_box_scale = fine_bbox_scales.squeeze(0)
                rgb_box_center = fine_bbox_centers.squeeze(0)
                rgb_box_tight_scale = fine_bbox_tight_scales.squeeze(0)
                rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
                rgb_crop_mask = model_func.query_cosegmentation(x_que=rgb_img_feat, 
                                                                x_ref=ref_database['obj_fps_feats'], 
                                                                ref_mask=ref_database['obj_fps_masks']).sigmoid() # 1xCxSxS -> 1x1xSxS
            else:
                roi_img_feats, _, roi_dino_tokens = model_func.extract_DINOv2_feature(roi_RGB_crops, return_last_dino_feat=True)
                roi_img_masks = model_func.query_cosegmentation(x_que=roi_img_feats, 
                                                                x_ref=ref_database['obj_fps_feats'], 
                                                                ref_mask=ref_database['obj_fps_masks']).sigmoid() # KxCxSxS -> Kx1xSxS
                roi_obj_mask = torch_F.interpolate(roi_img_masks, 
                                                    scale_factor=1.0/CFG.DINO_PATCH_SIZE, 
                                                    mode='bilinear', align_corners=True, 
                                                    recompute_scale_factor=True).flatten(2).permute(0, 2, 1).round() # KxLx1
                roi_dino_tokens = roi_obj_mask * roi_dino_tokens
                token_cosim = torch.einsum('klc,nc->kln', torch_F.normalize(roi_dino_tokens, dim=-1), 
                                                            torch_F.normalize(ref_database['obj_fps_dino_tokens'], dim=-1))
                if CFG.cosim_topk > 0:
                    cosim_score = token_cosim.topk(dim=1, k=CFG.cosim_topk).values.mean(dim=-1).mean(dim=1) # KxLxN -> KxTxN -> KxN -> K
                else:
                    cosim_score = token_cosim.mean(dim=-1).sum(dim=1) / (1 + roi_obj_mask.squeeze(-1).sum(dim=1))  # KxLxN -> KxL -> K
                
                optim_index = cosim_score.argmax()
                rgb_box_scale = fine_bbox_scales[optim_index]
                rgb_box_center = fine_bbox_centers[optim_index]
                rgb_box_tight_scale = fine_bbox_tight_scales[optim_index]
                rgb_img_crop = roi_RGB_crops[optim_index].unsqueeze(0)  # 1x3xSxS
                rgb_img_feat = roi_img_feats[optim_index].unsqueeze(0)  # 1xCxPxP
                rgb_crop_mask = roi_img_masks[optim_index].unsqueeze(0) # 1x1xSxS                                
        
        fine_det_cost = time.time() - start_timer

        RAEncoder_timer = time.time()
        rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
        rgb_crop_mask = model_func.query_cosegmentation(x_que=rgb_img_feat, 
                                                        x_ref=ref_database['obj_fps_feats'], 
                                                        ref_mask=ref_database['obj_fps_masks']).sigmoid()
        obj_Remb_vec = model_func.generate_rotation_aware_embedding(rgb_img_feat, rgb_crop_mask)
        RAEncoder_cost = time.time() - RAEncoder_timer

    return {
        'bbox_scale': rgb_box_scale,
        'bbox_center': rgb_box_center,
        'bbox_tight_scale': rgb_box_tight_scale,
        'obj_Remb': obj_Remb_vec.squeeze(0),
        'rgb_image': rgb_img_crop.squeeze(0), # 3xSxS
        'rgb_mask': rgb_crop_mask.squeeze(0), # 1xSxS
        'coarse_det_cost': coarse_det_cost,
        'fine_det_cost': fine_det_cost,
        'RAEncoder_cost': RAEncoder_cost,
    }

def naive_perform_segmentation_and_encoding(model_func, que_image, ref_database, device):
    with torch.no_grad():
        init_start_timer = time.time()
        
        if que_image.dim() == 3:
            que_image = que_image.unsqueeze(0)
        if que_image.shape[-1] == 3:
            que_image = que_image.permute(0, 3, 1, 2)
        que_feats = model_func.extract_DINOv2_feature(que_image.to(device))    
        pd_coarse_mask = model_func.query_cosegmentation(x_que=que_feats, 
                                                        x_ref=ref_database['obj_fps_feats'], 
                                                        ref_mask=ref_database['obj_fps_masks'],
                                                        ).sigmoid() # 1xCxHxW -> 1x1xHxW
        que_binary_mask = pd_coarse_mask.round().type(torch.uint8).squeeze()
        msk_yy, msk_xx = torch.nonzero(que_binary_mask, as_tuple=True)
        msk_yy, msk_xx = msk_yy.float(), msk_xx.float()
        m_x1, m_y1, m_x2, m_y2 = msk_xx.min(), msk_yy.min(), msk_xx.max(), msk_yy.max()
        msk_tight_scale = max(m_x2 - m_x1, m_y2 - m_y1)
        msk_center = torch.tensor([(m_x1 + m_x2) / 2, (m_y1 + m_y2) / 2])
        msk_scale = msk_tight_scale * CFG.coarse_bbox_padding

        pd_msk_bbox = torch.stack([msk_center[0] - msk_scale / 2.0,
                                        msk_center[1] - msk_scale / 2.0,
                                        msk_center[0] + msk_scale / 2.0,
                                        msk_center[1] + msk_scale / 2.0], dim=-1)
        rgb_img_crop = roi_align(que_image, boxes=[pd_msk_bbox[None]],
                                  output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale), 
                                  sampling_ratio=4) # 1x3xHxW -> Mx3xSxS
        
        rgb_box_scale = msk_scale
        rgb_box_center = msk_center
        rgb_box_tight_scale = msk_tight_scale

        RAEncoder_timer = time.time()
        rgb_img_feat = model_func.extract_DINOv2_feature(rgb_img_crop)
        rgb_crop_mask = model_func.query_cosegmentation(x_que=rgb_img_feat, 
                                                        x_ref=ref_database['obj_fps_feats'], 
                                                        ref_mask=ref_database['obj_fps_masks']).sigmoid()
        obj_Remb_vec = model_func.generate_rotation_aware_embedding(rgb_img_feat, rgb_crop_mask)
        RAEncoder_cost = time.time() - RAEncoder_timer
        
    return {
        'bbox_scale': rgb_box_scale,
        'bbox_center': rgb_box_center,
        'bbox_tight_scale': rgb_box_tight_scale,
        'obj_Remb': obj_Remb_vec.squeeze(0),
        'rgb_image': rgb_img_crop.squeeze(0), # 3xSxS
        'rgb_mask': rgb_crop_mask.squeeze(0), # 1xSxS
        'RAEncoder_cost': RAEncoder_cost,
    }

def perform_segmentation_and_encoding_from_bbox(model_func, que_image, ref_database, device):
    with torch.no_grad():
        if que_image.dim() == 3:
            que_image = que_image.unsqueeze(0)
        if que_image.shape[-1] == 3:
            que_image = que_image.permute(0, 3, 1, 2)
        
        RAEncoder_timer = time.time()
        rgb_img_feat = model_func.extract_DINOv2_feature(que_image.to(device))
        rgb_crop_mask = model_func.query_cosegmentation(x_que=rgb_img_feat, 
                                                        x_ref=ref_database['obj_fps_feats'], 
                                                        ref_mask=ref_database['obj_fps_masks']).sigmoid() # 1xCxSxS -> 1x1xSxS
        obj_Remb_vec = model_func.generate_rotation_aware_embedding(rgb_img_feat, rgb_crop_mask)
        RAEncoder_cost = time.time() - RAEncoder_timer

    return {
        'rgb_image': que_image.squeeze(0),    # 3x224x224
        'rgb_mask': rgb_crop_mask.squeeze(0), # 1x224x224
        'obj_Remb': obj_Remb_vec.squeeze(0),  # 64
        'RAEncoder_cost': RAEncoder_cost,
    }

def multiple_initial_pose_inference(obj_data, ref_database, device):
    camK = obj_data['camK'].to(device).squeeze()
    obj_Remb = obj_data['obj_Remb'].to(device).squeeze()
    obj_mask = obj_data['rgb_mask'].to(device).squeeze()
    bbox_scale = obj_data['bbox_scale'].to(device).squeeze()
    bbox_center = obj_data['bbox_center'].to(device).squeeze()
    
    que_msk_yy, que_msk_xx = torch.nonzero(obj_mask.round().squeeze(), as_tuple=True)
    que_msk_cx = (que_msk_xx.max() + que_msk_xx.min()) / 2
    que_msk_cy = (que_msk_yy.max() + que_msk_yy.min()) / 2
    que_bin_msk_area = obj_mask.round().sum()
    que_prob_msk_area = obj_mask.sum()

    Remb_cosim = torch.einsum('c, mc->m', obj_Remb, ref_database['refer_Remb_vectors'])
    max_inds = Remb_cosim.flatten().topk(dim=0, k=CFG.ROT_TOPK).indices
    init_Rs = ref_database['refer_allo_Rs'][max_inds]           # Kx3x3
    selected_nnb_info = ref_database['refer_coseg_mask_info'][max_inds] # Kx4

    nnb_ref_Cx = selected_nnb_info[:, 0]   # K
    nnb_ref_Cy = selected_nnb_info[:, 1]   # K
    nnb_ref_Tz = selected_nnb_info[:, 2]   # K
    nnb_ref_bin_area = selected_nnb_info[:, 3] # K
    nnb_ref_prob_area = selected_nnb_info[:, 4] # K
    if CFG.BINARIZE_MASK:
        delta_S = (que_bin_msk_area / nnb_ref_bin_area)**0.5
    else:
        delta_S = (que_prob_msk_area / nnb_ref_prob_area)**0.5

    delta_Px = (que_msk_cx - nnb_ref_Cx) / CFG.zoom_image_scale # K
    delta_Py = (que_msk_cy - nnb_ref_Cy) / CFG.zoom_image_scale # K
    delta_Pxy = torch.stack([delta_Px, delta_Py], dim=-1)   # Kx2
    que_Tz = nnb_ref_Tz / delta_S * CFG.zoom_image_scale / bbox_scale # K

    obj_Pxy = delta_Pxy * bbox_scale + bbox_center    # Kx2
    homo_pxpy = torch_F.pad(obj_Pxy, (0, 1), value=1) # Kx3
    init_Ts = torch.einsum('ij,kj->ki', torch.inverse(camK), homo_pxpy) * que_Tz.unsqueeze(1)
    
    init_RTs = torch.eye(4)[None, :, :].repeat(init_Rs.shape[0], 1, 1) # Kx4x4
    init_RTs[:, :3, :3] = init_Rs.detach().cpu()
    init_RTs[:, :3, 3] = init_Ts.detach().cpu()
    init_RTs = init_RTs.numpy()

    if CFG.USE_ALLOCENTRIC:
        for idx in range(init_RTs.shape[0]):
            init_RTs[idx, :3, :4] = gs_utils.allocentric_to_egocentric(init_RTs[idx, :3, :4])[:3, :4]

    return init_RTs

def multiple_refine_pose_with_GS_refiner(obj_data, init_pose, gaussians, device):
    def GS_Refiner(image, mask, init_camera, gaussians, return_loss=False):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[2] == 3:
            image = image.permute(2, 0, 1) # 3xSxS
        if mask.dim() == 2:
            mask = mask[None, :, :]
        if mask.dim() == 4:
            mask = mask.squeeze(0)
        if mask.shape[2] == 1:
            mask = mask.permute(2, 0, 1) # 1xSxS
        
        assert(image.dim() == 3 and image.shape[0] == 3), image.shape

        trunc_mask = (image.sum(dim=0, keepdim=True) > 0).type(torch.float32) # 1xSxS        
        target_img = (image * mask).to(device)

        gaussians.initialize_pose()
        optimizer = optim.AdamW([gaussians._delta_R, gaussians._delta_T], lr=CFG.START_LR)
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, CFG.MAX_STEPS,
                                                    warmup_steps=CFG.WARMUP, 
                                                    max_lr=CFG.START_LR, 
                                                    min_lr=CFG.END_LR)
        iter_losses = list()
        for iter_step in range(CFG.MAX_STEPS):
            render_img = GS_Renderer(init_camera, gaussians, gaussian_PipeP, gaussian_BG)['render'] * trunc_mask
            loss = 0.0

            if CFG.USE_MSE:
                loss += MSELoss(render_img, target_img).mean()
            if CFG.USE_SSIM:
                loss  += (1 - SSIM_METRIC(render_img[None, ...], target_img[None, ...]))
            if CFG.USE_MS_SSIM:
                loss += (1 - MS_SSIM_METRIC(render_img[None, ...], target_img[None, ...]))
                
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            iter_losses.append(loss.item())
            if iter_step >= CFG.EARLY_STOP_MIN_STEPS:
                loss_grads = (torch.as_tensor(iter_losses)[1:] - torch.as_tensor(iter_losses)[:-1]).abs()
                if loss_grads[-CFG.EARLY_STOP_MIN_STEPS:].mean() < CFG.EARLY_STOP_LOSS_GRAD_NORM: # early stop the refinement
                    break
        
        gs3d_delta_RT = gaussians.get_delta_pose.squeeze(0).detach().cpu().numpy()

        outp = {
            'gs3d_delta_RT': gs3d_delta_RT,
            'iter_step': iter_step,
            'render_img': render_img,
        }
        
        if return_loss:
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 3, 1, 1)
            sobel_x = sobel_x.to(image.device)
            sobel_y = sobel_x.transpose(-2, -1)
            # Apply Sobel filter to the images
            query_sobel_h = torch_F.conv2d(image[None], sobel_x, padding=0)
            query_sobel_v = torch_F.conv2d(image[None], sobel_y, padding=0)
            rend_sobel_h = torch_F.conv2d(render_img[None], sobel_x, padding=0)
            rend_sobel_v = torch_F.conv2d(render_img[None], sobel_y, padding=0)
            edge_err = (query_sobel_h - rend_sobel_h).abs().mean() + (query_sobel_v - rend_sobel_v).abs().mean()
            outp['edge_err'] = edge_err

        return outp

    target_size = CFG.GS_RENDER_SIZE
    camK = obj_data['camK'].clone().squeeze()
    bbox_center = obj_data['bbox_center']
    bbox_scale = obj_data['bbox_scale'] 
    image = obj_data['rgb_image']   # 3xSxS
    mask = obj_data['rgb_mask']
    if CFG.BINARIZE_MASK:
        mask = mask.round()
    
    if CFG.APPLY_ZOOM_AND_CROP:
        
        que_zoom_rescaling_factor = target_size / bbox_scale
        zoom_cam_fx = camK[0, 0] * que_zoom_rescaling_factor
        zoom_cam_fy = camK[1, 1] * que_zoom_rescaling_factor
        que_zoom_FovX = focal2fov(zoom_cam_fx, target_size)
        que_zoom_FovY = focal2fov(zoom_cam_fy, target_size)
        que_zoom_offsetX = -2 * (bbox_center[0] - camK[0, 2]) / bbox_scale
        que_zoom_offsetY = -2 * (bbox_center[1] - camK[1, 2]) / bbox_scale
    else:
        img_scale = obj_data['img_scale']
        que_zoom_rescaling_factor = target_size / img_scale
        zoom_cam_fx = camK[0, 0] * que_zoom_rescaling_factor
        zoom_cam_fy = camK[1, 1] * que_zoom_rescaling_factor
        zoom_cam_cx = camK[0, 2] * que_zoom_rescaling_factor
        zoom_cam_cy = camK[1, 2] * que_zoom_rescaling_factor
        que_zoom_FovX = focal2fov(zoom_cam_fx, target_size)
        que_zoom_FovY = focal2fov(zoom_cam_fy, target_size)
        que_zoom_offsetX = 2 * zoom_cam_cx / target_size - 1
        que_zoom_offsetY = 2 * zoom_cam_cy / target_size - 1
    
    if image.shape[-1] != target_size:
        image = torch_F.interpolate(image[None, ...], 
                                            size=target_size, 
                                            mode='bilinear', 
                                            align_corners=True).squeeze(0)
    if mask.shape[-1] != target_size:
        mask = torch_F.interpolate(mask[None, ...], 
                                            size=target_size, 
                                            mode='bilinear', 
                                            align_corners=True).squeeze(0)

    # trunc_mask = (image.sum(dim=0, keepdim=True) > 0).type(torch.float32) # 1xSxS        
    # target_image = (image * mask).to(device)
    
    gs3d_refined_errors = list()
    gs3d_refined_RTs = init_pose.copy() # Kx4x4
    for idx, init_RT in enumerate(init_pose):
        init_camera = GS_Camera(T=init_RT[:3, 3], R=init_RT[:3, :3].T,
                                FoVx=que_zoom_FovX, FoVy=que_zoom_FovY,
                                cx_offset=que_zoom_offsetX, cy_offset=que_zoom_offsetY,
                                image=image, colmap_id=0, uid=0, image_name='', gt_alpha_mask=None, data_device=device)
        
        ret_outp = GS_Refiner(image=image, mask=mask, init_camera=init_camera, gaussians=gaussians, return_loss=True)
        gs3d_delta_RT = ret_outp['gs3d_delta_RT']
        refined_err = ret_outp['edge_err']
        gs3d_refined_RTs[idx] = init_RT @ gs3d_delta_RT
        gs3d_refined_errors.append(refined_err)
    gs3d_refined_errors = torch.as_tensor(gs3d_refined_errors)
    best_idx = gs3d_refined_errors.argmin().item()
    gs3d_refined_RT = gs3d_refined_RTs[best_idx]

    ret_outp['bbox_center'] = bbox_center
    ret_outp['bbox_scale'] = bbox_scale
    ret_outp['gs3d_refined_RT'] = gs3d_refined_RT

    return ret_outp

def eval_GSPose(model_func, obj_dataset, ref_database_dir, output_pose_dir=None, save_pred_mask=False):  
    ref_database_path = os.path.join(ref_database_dir, 'reference_database.pkl')
    with open(ref_database_path, 'rb') as df:
        reference_database = pickle.load(df)
    for _key, _val in reference_database.items():
        if isinstance(_val, np.ndarray):
            reference_database[_key] = torch.as_tensor(_val, dtype=torch.float32).to(device)
    print('load database from ', ref_database_path)

    if args.enable_GS_Refiner:
        assert(ref_database_dir is not None)
        obj_gaussians = GaussianModel()
        gs_ply_path = os.path.join(ref_database_dir, '3DGO_model.ply')

        # all_ply_dirs = glob.glob(os.path.join(f'{ref_database_dir}/point_cloud/iteration_*'))
        # gs_ply_dir = sorted(all_ply_dirs, key=lambda x: int(x.split('_')[-1]), reverse=True)[0]
        # gs_ply_path = f'{gs_ply_dir}/point_cloud.ply'

        obj_gaussians.load_ply(gs_ply_path)
        print('load 3D-OGS model from ', gs_ply_path)
    
    if not os.path.exists(output_pose_dir):
        os.makedirs(output_pose_dir)
    log_file_f = open(os.path.join(output_pose_dir, 'log.txt'), 'w')

    num_que_views = len(obj_dataset)
    obj_name = obj_dataset.obj_name
    obj_diameter = obj_dataset.diameter
    is_symmetric = obj_dataset.is_symmetric
    obj_pointcloud = obj_dataset.obj_pointcloud
    
    name_info_log = 'name: {}, is_symmetric: {}, diameter: {:.4f} m, obj_pcd:{}'.format(
                    obj_name, is_symmetric, obj_diameter, obj_pointcloud.shape)
    
    log_file_f.write(name_info_log + '\n')
    print(name_info_log)

    runtime_metrics = {'detector_cost': list(), 'initilizer_cost': list(), 'refiner_cost': list()}

    init_metrics = {'image_IDs': list(), 'R_errs': list(), 't_errs': list()}
    refine_metrics = {'image_IDs': list(), 'R_errs': list(), 't_errs': list()}
    for view_idx in range(num_que_views):
        start_timer = time.time()
        que_data = obj_dataset[view_idx]
        camK = que_data['camK']
        gt_pose = que_data['pose'].numpy()
        que_image = que_data['image']      # HxWx3
        que_image_ID = que_data['image_ID']
        que_hei, que_wid = que_image.shape[:2]
        try:
            if CFG.USE_YOLO_BBOX:
                pd_bbox_center = que_data['bbox_center'].to(device)
                pd_bbox_scale = que_data['bbox_scale'].to(device) * CFG.coarse_bbox_padding
                pd_bbox = torch.stack([pd_bbox_center[0] - pd_bbox_scale / 2.0,
                                        pd_bbox_center[1] - pd_bbox_scale / 2.0,
                                        pd_bbox_center[0] + pd_bbox_scale / 2.0,
                                        pd_bbox_center[1] + pd_bbox_scale / 2.0], dim=-1) # 4
                que_roi_image = roi_align(que_image[None, ...].permute(0, 3, 1, 2).to(device), 
                                        boxes=[pd_bbox[None, :]], output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale), 
                                        sampling_ratio=4) # 1x3xHxW -> Mx3xSxS
                obj_data = perform_segmentation_and_encoding_from_bbox(
                    model_func, que_image=que_roi_image, ref_database=reference_database, device=device)
                obj_data['bbox_scale'] = pd_bbox_scale
                obj_data['bbox_center'] = pd_bbox_center
                if save_pred_mask:
                    coseg_mask_path = que_data['coseg_mask_path']
                    orig_mask = gs_utils.zoom_out_and_uncrop_image(obj_data['rgb_mask'].squeeze(),
                                                                    bbox_center=pd_bbox_center,
                                                                    bbox_scale=pd_bbox_scale,
                                                                    orig_hei=que_hei, orig_wid=que_wid) # 1xHxWx1
                    orig_mask = (orig_mask.detach().cpu().squeeze() * 255).numpy().astype(np.uint8) # HxW
                    if not os.path.exists(os.path.dirname(coseg_mask_path)):
                        os.makedirs(os.path.dirname(coseg_mask_path))
                    cv2.imwrite(coseg_mask_path, orig_mask)

                    rgb_crop = (obj_data['rgb_image'].detach().cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(coseg_mask_path.replace('.png', '_rgb_yolo.png'), rgb_crop)
            else:
                raw_hei, raw_wid = que_image.shape[:2]
                raw_long_size = max(raw_hei, raw_wid)
                raw_short_size = min(raw_hei, raw_wid)
                raw_aspect_ratio = raw_short_size / raw_long_size
                if raw_hei < raw_wid:
                    new_wid = CFG.query_longside_scale
                    new_hei = int(new_wid * raw_aspect_ratio)
                else:
                    new_hei = CFG.query_longside_scale
                    new_wid = int(new_hei * raw_aspect_ratio)
                query_rescaling_factor = CFG.query_longside_scale / raw_long_size
                que_image = que_image[None, ...].permute(0, 3, 1, 2).to(device)
                que_image = torch_F.interpolate(que_image, size=(new_hei, new_wid), mode='bilinear', align_corners=True)
                
                # obj_data = naive_perform_segmentation_and_encoding(model_func, 
                                                                    # device=device,
                                                                    # que_image=que_image,
                                                                    # ref_database=reference_database)
                
                obj_data = perform_segmentation_and_encoding(model_func, 
                                                            device=device,
                                                            que_image=que_image,
                                                            ref_database=reference_database)
                
                obj_data['bbox_scale'] /= query_rescaling_factor  # back to the original image scale
                obj_data['bbox_center'] /= query_rescaling_factor # back to the original image scale
                
                if save_pred_mask:
                    coseg_mask_path = que_data['coseg_mask_path']
                    orig_mask = gs_utils.zoom_out_and_uncrop_image(obj_data['rgb_mask'].squeeze(),
                                                                            bbox_center=obj_data['bbox_center'],
                                                                            bbox_scale=obj_data['bbox_scale'],
                                                                            orig_hei=que_hei,
                                                                            orig_wid=que_wid) # 1xHxWx1
                    orig_mask = (orig_mask.detach().cpu().squeeze() * 255).numpy().astype(np.uint8) # HxW
                    if not os.path.exists(os.path.dirname(coseg_mask_path)):
                        os.makedirs(os.path.dirname(coseg_mask_path))
                    cv2.imwrite(coseg_mask_path, orig_mask)

                    rgb_crop = (obj_data['rgb_image'].detach().cpu().squeeze().permute(1, 2, 0) * 255).numpy().astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(coseg_mask_path.replace('.png', '_rgb.png'), rgb_crop)
 
            obj_data['camK'] = camK  # object sequence-wise camera intrinsics, e.g., a fixed camera intrinsics for all frames within a sequence
            obj_data['img_scale'] = max(que_hei, que_wid)            
            
            initilizer_timer = time.time()
            init_RTs = multiple_initial_pose_inference(obj_data=obj_data, ref_database=reference_database, device=device)
            init_RT = init_RTs[0]

            if obj_data.get('RAEncoder_cost', None) is not None:
                initilizer_cost = time.time() - initilizer_timer + obj_data['RAEncoder_cost']
                runtime_metrics['initilizer_cost'].append(initilizer_cost)

            if obj_data.get('fine_det_cost', None) is not None:
                runtime_metrics['detector_cost'].append(obj_data['fine_det_cost'])
            
            if args.enable_GS_Refiner:
                refiner_timer = time.time()
                refiner_oupt = multiple_refine_pose_with_GS_refiner(obj_data, init_pose=init_RTs, gaussians=obj_gaussians, device=device)
                
                refine_RT = refiner_oupt['gs3d_refined_RT']
                refiner_cost = time.time() - refiner_timer
                runtime_metrics['refiner_cost'].append(refiner_cost)

                refine_Rerr, refine_Terr = calc_pose_error(refine_RT, gt_pose)
                refine_metrics['R_errs'].append(refine_Rerr)
                refine_metrics['t_errs'].append(refine_Terr)
                refine_metrics['image_IDs'].append(que_image_ID)
                try:
                    refine_add = calc_add_metric(obj_pointcloud, obj_diameter, refine_RT, gt_pose, syn=is_symmetric)

                    if 'ADD_metric' not in refine_metrics.keys():
                        refine_metrics['ADD_metric'] = list()
                    refine_metrics['ADD_metric'].append(refine_add)

                    refine_proj_err = calc_projection_2d_error(obj_pointcloud, refine_RT, gt_pose, camK)
                    if 'Proj2D' not in refine_metrics.keys():
                        refine_metrics['Proj2D'] = list()
                    refine_metrics['Proj2D'].append(refine_proj_err)

                except:
                    pass
        
        except Exception as e:
            print('Error in processing image {}: {}'.format(que_image_ID, e))
            init_RT = np.eye(4)
            refine_RT = np.eye(4)
            
        
        init_Rerr, init_Terr = calc_pose_error(init_RT, gt_pose)
        init_metrics['R_errs'].append(init_Rerr)
        init_metrics['t_errs'].append(init_Terr)
        init_metrics['image_IDs'].append(que_image_ID)

        try:
            init_add = calc_add_metric(obj_pointcloud, obj_diameter, init_RT, gt_pose, syn=is_symmetric)
            if 'ADD_metric' not in init_metrics.keys():
                init_metrics['ADD_metric'] = list()
            init_metrics['ADD_metric'].append(init_add)

            init_proj_err = calc_projection_2d_error(obj_pointcloud, init_RT, gt_pose, camK)
            if 'Proj2D' not in init_metrics.keys():
                init_metrics['Proj2D'] = list()
            init_metrics['Proj2D'].append(init_proj_err)
        except:
            pass

        if output_pose_dir is not None:
            coseg_pose_txt = os.path.join(output_pose_dir, 'coseg_init_pose', f'{que_image_ID}.txt')
            if not os.path.exists(os.path.dirname(coseg_pose_txt)):
                os.makedirs(os.path.dirname(coseg_pose_txt))
            np.savetxt(coseg_pose_txt, init_RT.tolist())

            if args.enable_GS_Refiner:
                refined_pose_txt = os.path.join(output_pose_dir, 'gs3d_refine_pose', f'{que_image_ID}.txt')
                if not os.path.exists(os.path.dirname(refined_pose_txt)):
                    os.makedirs(os.path.dirname(refined_pose_txt))
                np.savetxt(refined_pose_txt, refine_RT.tolist())

        if (view_idx + 1) % 100 == 0 or (view_idx + 1) == num_que_views:
            time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())

            init_log_str = ''
            init_results = aggregate_metrics(init_metrics)
            for _key, _val in init_results.items():
                init_log_str += ' {}:{:.2f},'.format(_key, _val*100)
            init_log_str = init_log_str[1:-1]

            refine_log_str = ''
            refine_results = aggregate_metrics(refine_metrics)
            for _key, _val in refine_results.items():
                refine_log_str += ' {}:{:.2f},'.format(_key, _val*100)
            refine_log_str = refine_log_str[1:-1]
            print('[{}/{}], init=[{}], refine=[{}], {}'.format(view_idx+1, num_que_views, init_log_str, refine_log_str, time_stamp))

            # print the runtime metrics
            det_cost = 0
            init_cost = 0
            refine_cost = 0
            if len(runtime_metrics['detector_cost']) > 0:
                det_cost = np.array(runtime_metrics['detector_cost']).mean()
            if len(runtime_metrics['initilizer_cost']) > 0:
                init_cost = np.array(runtime_metrics['initilizer_cost']).mean()
            if len(runtime_metrics['refiner_cost']) > 0:
                refine_cost = np.array(runtime_metrics['refiner_cost']).mean()
            total_cost = det_cost + init_cost + refine_cost
            log_runtime = 'detector_cost: {:.4f}, initilizer_cost: {:.4f}, refiner_cost: {:.4f}, total_cost: {:.4f}'.format(det_cost, init_cost, refine_cost, total_cost)
            print('[{}/{}], {}'.format(view_idx+1, num_que_views, log_runtime))

            log_file_f.write('[{}/{}], init=[{}], refine=[{}], {}, {}\n'.format(view_idx+1, num_que_views, init_log_str, refine_log_str, time_stamp, log_runtime))
            
    time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())

    init_log_str = ''
    init_results = aggregate_metrics(init_metrics)
    for _key, _val in init_results.items():
        init_log_str += ' {}:{:.2f},'.format(_key, _val*100)
    init_log_str = init_log_str[1:-1]

    refine_log_str = ''
    refine_results = aggregate_metrics(refine_metrics)
    for _key, _val in refine_results.items():
        refine_log_str += ' {}:{:.2f},'.format(_key, _val*100)
    refine_log_str = refine_log_str[1:-1]
    print('{}, init=[{}], refine=[{}], {}'.format(obj_name, init_log_str, refine_log_str, time_stamp))
    
    init_pose_summary_path = os.path.join(output_pose_dir, 'coseg_init_pose_summary.txt')
    with open(init_pose_summary_path, 'w') as f:
        f.write('### image_id, R_err(˚), t_err(cm) {} \n'.format(init_results))
        for idx, img_ID in enumerate(init_metrics['image_IDs']):
            f.write('{}, {:.4f}, {:.4f} \n'.format(
                img_ID, init_metrics['R_errs'][idx], init_metrics['t_errs'][idx])
            )
    
    if args.enable_GS_Refiner:
        refine_pose_summary_path = os.path.join(output_pose_dir, 'gs3d_refine_pose_summary.txt')
        with open(refine_pose_summary_path, 'w') as f:
            f.write('### image_id, R_err(˚), t_err(cm) {} \n'.format(refine_results))
            for idx, img_ID in enumerate(refine_metrics['image_IDs']):
                f.write('{}, {:.4f}, {:.4f} \n'.format(
                    img_ID, refine_metrics['R_errs'][idx], refine_metrics['t_errs'][idx])
                )
    
    log_file_f.close()
    return {'init': init_results, 'refine': refine_results}


def render_Gaussian_object_model(obj_gaussians, camK, pose, img_hei, img_wid, device): 
    if isinstance(pose, torch.Tensor):
        pose = pose.numpy()
    obj_gaussians.initialize_pose()
    FovX = focal2fov(camK[0, 0], img_wid)
    FovY = focal2fov(camK[1, 1], img_hei)
    target_image = torch.zeros((3, img_hei, img_wid)).to(device)
    gs_camera = GS_Camera(T=pose[:3, 3],
                          R=pose[:3, :3].T, 
                          FoVx=FovX, FoVy=FovY,
                          cx_offset=0, cy_offset=0,
                          image=target_image, colmap_id=0, uid=0, image_name='', 
                          gt_alpha_mask=None, data_device=device)    
    render_img = GS_Renderer(gs_camera, obj_gaussians, gaussian_PipeP, gaussian_BG)['render']
    render_img_np = render_img.permute(1, 2, 0).detach().cpu().numpy()
    render_img_np = (render_img_np * 255).astype(np.uint8)
    
    return render_img_np


def GS_Tracker(model_func, ref_database, frame, camK, prev_pose):
    zoom_outp = gs_utils.zoom_in_and_crop_with_offset(image=frame, K=camK, 
                                                        t=prev_pose[:3, 3],
                                                        radius=ref_database['bbox3d_diameter']/2,
                                                        target_size=CFG.zoom_image_scale, 
                                                        margin=CFG.zoom_image_margin)                
    zoom_camK = zoom_outp['zoom_camK']      
    zoom_image = zoom_outp['zoom_image']
    bbox_scale = zoom_outp['bbox_scale']   
    bbox_center = zoom_outp['bbox_center']
    zoom_offset = zoom_outp['zoom_offset']
    zoom_offsetX = zoom_offset[0]
    zoom_offsetY = zoom_offset[1]
    
    zoom_FovX = focal2fov(zoom_camK[0, 0], CFG.zoom_image_scale)
    zoom_FovY = focal2fov(zoom_camK[1, 1], CFG.zoom_image_scale)
    target_image = zoom_image.permute(2, 0, 1).to(device) # 3xSxS

    fg_trunc_mask = (target_image.sum(dim=0, keepdim=True) > 0).type(torch.float32) # 1xSxS
    
    with torch.no_grad():
        target_mask = model_func.query_cosegmentation(
            model_func.extract_DINOv2_feature(target_image[None]), 
            x_ref=ref_database['obj_fps_feats'], ref_mask=ref_database['obj_fps_masks'],
        ).sigmoid().squeeze(0)

    zoom_image_np = (target_image.detach().cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8)
    
    obj_gaussians = ref_database['obj_gaussians']
    track_camera = GS_Camera(T=prev_pose[:3, 3],
                            R=prev_pose[:3, :3].T, 
                            FoVx=zoom_FovX, FoVy=zoom_FovY,
                            cx_offset=zoom_offsetX, cy_offset=zoom_offsetY,
                            image=target_image, colmap_id=0, uid=0, image_name='', 
                            gt_alpha_mask=None, data_device=device)

    obj_gaussians.initialize_pose()
    
    optimizer = optim.AdamW([obj_gaussians._delta_R, obj_gaussians._delta_T])

    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                                 CFG.MAX_STEPS, 
                                                 warmup_steps=CFG.WARMUP, 
                                                 max_lr=CFG.START_LR, min_lr=CFG.END_LR)
    losses = list()
    target_image *= target_mask
    for iter_step in range(CFG.MAX_STEPS):
        optimizer.zero_grad()
        render_img = GS_Renderer(track_camera, obj_gaussians, gaussian_PipeP, gaussian_BG)['render'] * fg_trunc_mask
        loss = 0
        
        rgb_loss = L1Loss(render_img, target_image)
        loss += rgb_loss
    
        ssim_loss = 1 - SSIM_METRIC(render_img[None], target_image[None])
        loss += ssim_loss
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        losses.append(loss.item())
        if iter_step >= CFG.EARLY_STOP_MIN_STEPS:
            loss_grads = (torch.as_tensor(losses)[1:] - torch.as_tensor(losses)[:-1]).abs()
            loss_grad = loss_grads[-CFG.EARLY_STOP_MIN_STEPS:].mean() 
            if loss_grad < CFG.EARLY_STOP_LOSS_GRAD_NORM:
                break
    
    gs3d_delta_RT = obj_gaussians.get_delta_pose.squeeze(0).detach().cpu().numpy()
    curr_pose = prev_pose @ gs3d_delta_RT
        
    return{
        'track_pose': curr_pose,
        'render_img': render_img,
        'bbox_scale': bbox_scale,
        'bbox_center': bbox_center,
        'iter_step': iter_step,
    }

def eval_GSPose_with_database(model_func, obj_dataset, reference_database, output_pose_dir=None, save_pred_mask=False):  
    gs_ply_path = reference_database['obj_gaussians_path']
    obj_gaussians = GaussianModel()
    obj_gaussians.load_ply(gs_ply_path)

    for _key, _val in reference_database.items():
        if isinstance(_val, np.ndarray):
            reference_database[_key] = torch.as_tensor(_val, dtype=torch.float32).to(device)
    
    if not os.path.exists(output_pose_dir):
        os.makedirs(output_pose_dir)
    log_file_f = open(os.path.join(output_pose_dir, 'log.txt'), 'w')

    num_que_views = len(obj_dataset)
    obj_name = obj_dataset.obj_name
    obj_diameter = obj_dataset.diameter
    is_symmetric = obj_dataset.is_symmetric
    obj_pointcloud = obj_dataset.obj_pointcloud
    
    name_info_log = 'name: {}, is_symmetric: {}, diameter: {:.4f} m, obj_pcd:{}'.format(
                    obj_name, is_symmetric, obj_diameter, obj_pointcloud.shape)
    
    log_file_f.write(name_info_log + '\n')
    print(name_info_log)

    runtime_metrics = {'detector_cost': list(), 'initilizer_cost': list(), 'refiner_cost': list()}

    init_metrics = {'image_IDs': list(), 'R_errs': list(), 't_errs': list()}
    refine_metrics = {'image_IDs': list(), 'R_errs': list(), 't_errs': list()}
    for view_idx in range(num_que_views):
        start_timer = time.time()
        que_data = obj_dataset[view_idx]
        camK = que_data['camK']
        gt_pose = que_data['pose'].numpy()
        que_image = que_data['image']      # HxWx3
        que_image_ID = que_data['image_ID']
        que_hei, que_wid = que_image.shape[:2]
        try:
            if CFG.USE_YOLO_BBOX:
                pd_bbox_center = que_data['bbox_center'].to(device)
                pd_bbox_scale = que_data['bbox_scale'].to(device) * CFG.coarse_bbox_padding
                pd_bbox = torch.stack([pd_bbox_center[0] - pd_bbox_scale / 2.0,
                                        pd_bbox_center[1] - pd_bbox_scale / 2.0,
                                        pd_bbox_center[0] + pd_bbox_scale / 2.0,
                                        pd_bbox_center[1] + pd_bbox_scale / 2.0], dim=-1) # 4
                que_roi_image = roi_align(que_image[None, ...].permute(0, 3, 1, 2).to(device), 
                                        boxes=[pd_bbox[None, :]], output_size=(CFG.zoom_image_scale, CFG.zoom_image_scale), 
                                        sampling_ratio=4) # 1x3xHxW -> Mx3xSxS
                obj_data = perform_segmentation_and_encoding_from_bbox(
                    model_func, que_image=que_roi_image, ref_database=reference_database, device=device)
                obj_data['bbox_scale'] = pd_bbox_scale
                obj_data['bbox_center'] = pd_bbox_center
                if save_pred_mask:
                    coseg_mask_path = que_data['coseg_mask_path']
                    orig_mask = gs_utils.zoom_out_and_uncrop_image(obj_data['rgb_mask'].squeeze(),
                                                                    bbox_center=pd_bbox_center,
                                                                    bbox_scale=pd_bbox_scale,
                                                                    orig_hei=que_hei, orig_wid=que_wid) # 1xHxWx1
                    orig_mask = (orig_mask.detach().cpu().squeeze() * 255).numpy().astype(np.uint8) # HxW
                    if not os.path.exists(os.path.dirname(coseg_mask_path)):
                        os.makedirs(os.path.dirname(coseg_mask_path))
                    cv2.imwrite(coseg_mask_path, orig_mask)

                    rgb_crop = (obj_data['rgb_image'].detach().cpu().permute(1, 2, 0) * 255).numpy().astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(coseg_mask_path.replace('.png', '_rgb_yolo.png'), rgb_crop)
            else:
                raw_hei, raw_wid = que_image.shape[:2]
                raw_long_size = max(raw_hei, raw_wid)
                raw_short_size = min(raw_hei, raw_wid)
                raw_aspect_ratio = raw_short_size / raw_long_size
                if raw_hei < raw_wid:
                    new_wid = CFG.query_longside_scale
                    new_hei = int(new_wid * raw_aspect_ratio)
                else:
                    new_hei = CFG.query_longside_scale
                    new_wid = int(new_hei * raw_aspect_ratio)
                query_rescaling_factor = CFG.query_longside_scale / raw_long_size
                que_image = que_image[None, ...].permute(0, 3, 1, 2).to(device)
                que_image = torch_F.interpolate(que_image, size=(new_hei, new_wid), mode='bilinear', align_corners=True)
                
                # obj_data = naive_perform_segmentation_and_encoding(model_func, 
                                                                    # device=device,
                                                                    # que_image=que_image,
                                                                    # ref_database=reference_database)
                
                obj_data = perform_segmentation_and_encoding(model_func, 
                                                            device=device,
                                                            que_image=que_image,
                                                            ref_database=reference_database)
                
                obj_data['bbox_scale'] /= query_rescaling_factor  # back to the original image scale
                obj_data['bbox_center'] /= query_rescaling_factor # back to the original image scale
                
                if save_pred_mask:
                    coseg_mask_path = que_data['coseg_mask_path']
                    orig_mask = gs_utils.zoom_out_and_uncrop_image(obj_data['rgb_mask'].squeeze(),
                                                                            bbox_center=obj_data['bbox_center'],
                                                                            bbox_scale=obj_data['bbox_scale'],
                                                                            orig_hei=que_hei,
                                                                            orig_wid=que_wid) # 1xHxWx1
                    orig_mask = (orig_mask.detach().cpu().squeeze() * 255).numpy().astype(np.uint8) # HxW
                    if not os.path.exists(os.path.dirname(coseg_mask_path)):
                        os.makedirs(os.path.dirname(coseg_mask_path))
                    cv2.imwrite(coseg_mask_path, orig_mask)

                    rgb_crop = (obj_data['rgb_image'].detach().cpu().squeeze().permute(1, 2, 0) * 255).numpy().astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(coseg_mask_path.replace('.png', '_rgb.png'), rgb_crop)
 
            obj_data['camK'] = camK  # object sequence-wise camera intrinsics, e.g., a fixed camera intrinsics for all frames within a sequence
            obj_data['img_scale'] = max(que_hei, que_wid)            
            
            initilizer_timer = time.time()
            init_RTs = multiple_initial_pose_inference(obj_data=obj_data, ref_database=reference_database, device=device)
            init_RT = init_RTs[0]

            if obj_data.get('RAEncoder_cost', None) is not None:
                initilizer_cost = time.time() - initilizer_timer + obj_data['RAEncoder_cost']
                runtime_metrics['initilizer_cost'].append(initilizer_cost)

            if obj_data.get('fine_det_cost', None) is not None:
                runtime_metrics['detector_cost'].append(obj_data['fine_det_cost'])
            
            if args.enable_GS_Refiner:
                refiner_timer = time.time()
                refiner_oupt = multiple_refine_pose_with_GS_refiner(obj_data, init_pose=init_RTs, gaussians=obj_gaussians, device=device)
                
                refine_RT = refiner_oupt['gs3d_refined_RT']
                refiner_cost = time.time() - refiner_timer
                runtime_metrics['refiner_cost'].append(refiner_cost)

                refine_Rerr, refine_Terr = calc_pose_error(refine_RT, gt_pose)
                refine_metrics['R_errs'].append(refine_Rerr)
                refine_metrics['t_errs'].append(refine_Terr)
                refine_metrics['image_IDs'].append(que_image_ID)
                try:
                    refine_add = calc_add_metric(obj_pointcloud, obj_diameter, refine_RT, gt_pose, syn=is_symmetric)

                    if 'ADD_metric' not in refine_metrics.keys():
                        refine_metrics['ADD_metric'] = list()
                    refine_metrics['ADD_metric'].append(refine_add)

                    refine_proj_err = calc_projection_2d_error(obj_pointcloud, refine_RT, gt_pose, camK)
                    if 'Proj2D' not in refine_metrics.keys():
                        refine_metrics['Proj2D'] = list()
                    refine_metrics['Proj2D'].append(refine_proj_err)

                except:
                    pass
        
        except Exception as e:
            print('Error in processing image {}: {}'.format(que_image_ID, e))
            init_RT = np.eye(4)
            refine_RT = np.eye(4)
            
        
        init_Rerr, init_Terr = calc_pose_error(init_RT, gt_pose)
        init_metrics['R_errs'].append(init_Rerr)
        init_metrics['t_errs'].append(init_Terr)
        init_metrics['image_IDs'].append(que_image_ID)

        try:
            init_add = calc_add_metric(obj_pointcloud, obj_diameter, init_RT, gt_pose, syn=is_symmetric)
            if 'ADD_metric' not in init_metrics.keys():
                init_metrics['ADD_metric'] = list()
            init_metrics['ADD_metric'].append(init_add)

            init_proj_err = calc_projection_2d_error(obj_pointcloud, init_RT, gt_pose, camK)
            if 'Proj2D' not in init_metrics.keys():
                init_metrics['Proj2D'] = list()
            init_metrics['Proj2D'].append(init_proj_err)
        except:
            pass

        if output_pose_dir is not None:
            coseg_pose_txt = os.path.join(output_pose_dir, 'coseg_init_pose', f'{que_image_ID}.txt')
            if not os.path.exists(os.path.dirname(coseg_pose_txt)):
                os.makedirs(os.path.dirname(coseg_pose_txt))
            np.savetxt(coseg_pose_txt, init_RT.tolist())

            if args.enable_GS_Refiner:
                refined_pose_txt = os.path.join(output_pose_dir, 'gs3d_refine_pose', f'{que_image_ID}.txt')
                if not os.path.exists(os.path.dirname(refined_pose_txt)):
                    os.makedirs(os.path.dirname(refined_pose_txt))
                np.savetxt(refined_pose_txt, refine_RT.tolist())

        if (view_idx + 1) % 100 == 0 or (view_idx + 1) == num_que_views:
            time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())

            init_log_str = ''
            init_results = aggregate_metrics(init_metrics)
            for _key, _val in init_results.items():
                init_log_str += ' {}:{:.2f},'.format(_key, _val*100)
            init_log_str = init_log_str[1:-1]

            refine_log_str = ''
            refine_results = aggregate_metrics(refine_metrics)
            for _key, _val in refine_results.items():
                refine_log_str += ' {}:{:.2f},'.format(_key, _val*100)
            refine_log_str = refine_log_str[1:-1]
            print('[{}/{}], init=[{}], refine=[{}], {}'.format(view_idx+1, num_que_views, init_log_str, refine_log_str, time_stamp))

            # print the runtime metrics
            det_cost = 0
            init_cost = 0
            refine_cost = 0
            if len(runtime_metrics['detector_cost']) > 0:
                det_cost = np.array(runtime_metrics['detector_cost']).mean()
            if len(runtime_metrics['initilizer_cost']) > 0:
                init_cost = np.array(runtime_metrics['initilizer_cost']).mean()
            if len(runtime_metrics['refiner_cost']) > 0:
                refine_cost = np.array(runtime_metrics['refiner_cost']).mean()
            total_cost = det_cost + init_cost + refine_cost
            log_runtime = 'detector_cost: {:.4f}, initilizer_cost: {:.4f}, refiner_cost: {:.4f}, total_cost: {:.4f}'.format(det_cost, init_cost, refine_cost, total_cost)
            print('[{}/{}], {}'.format(view_idx+1, num_que_views, log_runtime))

            log_file_f.write('[{}/{}], init=[{}], refine=[{}], {}, {}\n'.format(view_idx+1, num_que_views, init_log_str, refine_log_str, time_stamp, log_runtime))
            
    time_stamp = time.strftime('%d-%H:%M:%S', time.localtime())

    init_log_str = ''
    init_results = aggregate_metrics(init_metrics)
    for _key, _val in init_results.items():
        init_log_str += ' {}:{:.2f},'.format(_key, _val*100)
    init_log_str = init_log_str[1:-1]

    refine_log_str = ''
    refine_results = aggregate_metrics(refine_metrics)
    for _key, _val in refine_results.items():
        refine_log_str += ' {}:{:.2f},'.format(_key, _val*100)
    refine_log_str = refine_log_str[1:-1]
    print('{}, init=[{}], refine=[{}], {}'.format(obj_name, init_log_str, refine_log_str, time_stamp))
    
    init_pose_summary_path = os.path.join(output_pose_dir, 'coseg_init_pose_summary.txt')
    with open(init_pose_summary_path, 'w') as f:
        f.write('### image_id, R_err(˚), t_err(cm) {} \n'.format(init_results))
        for idx, img_ID in enumerate(init_metrics['image_IDs']):
            f.write('{}, {:.4f}, {:.4f} \n'.format(
                img_ID, init_metrics['R_errs'][idx], init_metrics['t_errs'][idx])
            )
    
    if args.enable_GS_Refiner:
        refine_pose_summary_path = os.path.join(output_pose_dir, 'gs3d_refine_pose_summary.txt')
        with open(refine_pose_summary_path, 'w') as f:
            f.write('### image_id, R_err(˚), t_err(cm) {} \n'.format(refine_results))
            for idx, img_ID in enumerate(refine_metrics['image_IDs']):
                f.write('{}, {:.4f}, {:.4f} \n'.format(
                    img_ID, refine_metrics['R_errs'][idx], refine_metrics['t_errs'][idx])
                )
    
    log_file_f.close()
    return {'init': init_results, 'refine': refine_results}



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Training script parameters")
    ###### arguments for 3D-Gaussian Splatting Refiner ########
    gaussian_ModelP = ModelParams(parser)
    gaussian_PipeP  = PipelineParams(parser)
    gaussian_OptimP = OptimizationParams(parser)
    gaussian_BG = torch.zeros((3), device=device)

    # Set up command line argument parser
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    ###### arguments for CoSegPose ########
    parser.add_argument('--use_model_based', action='store_true', help='enable 3D GaussianObject model-based database reconstruction')
    parser.add_argument('--num_refer_views', type=int, default=-1)
    parser.add_argument('--dataset_name', default='LINEMOD', type=str, help='dataset name')
    parser.add_argument('--outpose_dir', default='output_pose', type=str, help='output pose directory')
    parser.add_argument('--database_dir', default='reference_database', type=str, help='reference database directory')

    parser.add_argument('--build_GS_model', action='store_true', help='enable fine detection')
    parser.add_argument('--enable_GS_Refiner', action='store_true', help='enable 3D Gaussian Splatting Refiner')
    args = parser.parse_args()


    CFG.USE_YOLO_BBOX = False
    if 'yolo' in args.outpose_dir:
        CFG.USE_YOLO_BBOX = True
    
    if CFG.BINARIZE_MASK:
        postfix = 'binamask'
    else:
        postfix = 'probmask'
    
    args.build_GS_model = True
    args.enable_GS_Refiner = True

    dataset_name = args.dataset_name
    num_refer_views = args.num_refer_views
    enable_3DGO_model_based_database = args.use_model_based

    output_dir = os.path.join(PROJ_ROOT, f'obj_database', dataset_name)
    
    outpose_dir = os.path.join(output_dir, args.outpose_dir + f'_{postfix}')
    database_dir = os.path.join(output_dir, args.database_dir + f'_{postfix}')

    if num_refer_views > 0:
        outpose_dir = os.path.join(output_dir, args.outpose_dir + f'_{postfix}_views{num_refer_views}')
        database_dir = os.path.join(output_dir, args.database_dir + f'_{postfix}_views{num_refer_views}')
    
    if not os.path.exists(outpose_dir):
        os.makedirs(outpose_dir)

    with open(os.path.join(outpose_dir, 'config.yaml'), 'w') as cfg_f:
        log_str = '----------- CONFIG Parameters ----------- \n'
        cfg_f.write(log_str)
        print(log_str[:-2])
        for cfg_key in vars(CFG):
            if cfg_key.startswith('__'):
                continue
            log_str = '{}={} \n'.format(cfg_key, eval('CFG.{}'.format(cfg_key)))
            cfg_f.write(log_str)
            print(log_str[:-2])

    assert(dataset_name in datasetCallbacks.keys()), '{} is not in {}'.format(dataset_name, datasetCallbacks.keys())
    data_root = datasetCallbacks[dataset_name]['DATAROOT']
    datasetObjects = datasetCallbacks[dataset_name]['OBJECTS']
    datasetLoader = datasetCallbacks[dataset_name]['DATASETLOADER']
    
    summarized_results = dict()
    for obj_name, obj_dir_name in datasetObjects.items():
        obj_refer_database_dir = os.path.join(database_dir, obj_name)
        obj_ref_database_path = os.path.join(obj_refer_database_dir, 'reference_database.pkl') 

        print(f'loading test dataset for {obj_name}')
        obj_output_pose_dir = os.path.join(outpose_dir, obj_name)
        obj_test_dataset = datasetLoader(data_root, obj_name, subset_mode='test', 
                                         obj_database_dir=obj_refer_database_dir, 
                                         load_yolo_det=CFG.USE_YOLO_BBOX)

        if not os.path.exists(obj_ref_database_path):
            print(f'preprocess reference data for {obj_name}')
            obj_refer_dataset = datasetLoader(data_root, obj_name, 
                                            subset_mode='train', 
                                            num_refer_views=num_refer_views,
                                            use_binarized_mask=CFG.BINARIZE_MASK,
                                            obj_database_dir=obj_refer_database_dir)  
            ref_database = create_reference_database_from_RGB_images(model_net, obj_refer_dataset, device=device, save_pred_mask=True)
            ref_database['obj_bbox3D'] = torch.as_tensor(obj_refer_dataset.obj_bbox3d, dtype=torch.float32)
            ref_database['bbox3d_diameter'] = torch.as_tensor(obj_refer_dataset.bbox3d_diameter, dtype=torch.float32)
            print(f'building the 3D Gaussian model for {obj_name}')
            gs_pipeData  = gaussian_PipeP.extract(args)
            gs_modelData = gaussian_ModelP.extract(args)
            gs_optimData = gaussian_OptimP.extract(args)
            
            gs_modelData.source_path = os.path.join(data_root, obj_dir_name)
            gs_modelData.model_path = obj_refer_database_dir
            gs_modelData.referloader = obj_refer_dataset
            gs_modelData.queryloader = obj_test_dataset

            create_3D_Gaussian_object(gs_modelData, gs_optimData, gs_pipeData,
                                    args.test_iterations, args.save_iterations,
                                    args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

            ref_database['obj_gaussians_path'] = f'{obj_refer_database_dir}/3DGO_model.ply'
            # Save dictionary to a file
            for _key, _val in ref_database.items():
                if isinstance(_val, torch.Tensor):
                    ref_database[_key] = _val.detach().cpu().numpy()
            
            with open(obj_ref_database_path, 'wb') as df:
                pickle.dump(ref_database, df)
            print('save database to ', obj_ref_database_path)
        
        else:
            print('load database from ', obj_ref_database_path)
            with open(obj_ref_database_path, 'rb') as df:
                ref_database = pickle.load(df)
    
        print(f'performing the pose estimation for {obj_name}')
        obj_result = eval_GSPose_with_database(model_net,
                                    obj_test_dataset, 
                                    reference_database=ref_database,
                                    output_pose_dir=obj_output_pose_dir, save_pred_mask=True)
        
        for _key, _val in obj_result.items():
            if _key not in summarized_results.keys():
                summarized_results[_key] = dict()
            summarized_results[_key][obj_name] = _val

    print('summarized_results: ', summarized_results)
    out_summary_path = os.path.join(outpose_dir, 'summarized_results.txt')
    with open(out_summary_path, 'w') as sum_f:
        str_len = 6
        for _mode, mode_datum in summarized_results.items():
            print(f'--------------------------------------------------------{_mode}--------------------------------------------------------')        
            title_str = 'metric: '
            metric_values = dict()
            for obj_name, obj_datum in mode_datum.items():
                title_str += f'{obj_name[:str_len]:>{str_len}}, '
                for m_type, m_val in obj_datum.items():
                    if m_type not in metric_values:
                        metric_values[m_type] = list()
                    metric_values[m_type].append(m_val)
            title_str += 'Mean'
            sum_f.write(title_str + '\n')
            print(title_str) 
            
            for m_type, m_vals in metric_values.items():
                value_str = f'{m_type[:str_len]:>{str_len}}: '
                for _val in m_vals:
                    val_str = '{:.4f}'.format(_val)
                    value_str += f'{val_str[:str_len]:>{str_len}}, '
                avg_str = '{:.4f}'.format(np.mean(m_vals))
                value_str += f'{avg_str[:str_len]:>{str_len}}'
                sum_f.write(value_str + '\n')
                print(value_str) 

"""
old_scale =
{  'ape':          0.0974, # 0.1021
    'benchvise':   0.2869, # 0.2475
    'cam':         0.1716, # 0.1725
    'can':         0.1934, # 0.2014
    'cat':         0.1526, # 0.1545
    'driller':     0.2594, # 
    'duck':        0.1071,
    'eggbox':      0.1764,
    'glue':        0.1649,
    'holepuncher': 0.1482,
    'iron':        0.3032,
    'lamp':        0.2852,
    'phone':       0.2084,
}

python inference.py --dataset_name LINEMOD  --database_dir LM_database --outpose_dir LM_yolo_pose
python inference.py --dataset_name LINEMOD_SUBSET  --database_dir LMSubSet_database --outpose_dir LMSubSet_pose
python inference.py --dataset_name LOWTEXTUREVideo  --database_dir LTVideo_database --outpose_dir LTVideo_pose


"""
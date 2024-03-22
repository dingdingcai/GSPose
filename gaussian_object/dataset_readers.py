#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import cv2
import sys
import json
import torch

import numpy as np
from PIL import Image
from pathlib import Path
from typing import NamedTuple
# from plyfile import PlyData, PlyElement
from pytorch3d import transforms as py3d_transform


# import the customized modules
from misc_utils import gs_utils
from gaussian_object.utils.sh_utils import SH2RGB
from gaussian_object.gaussian_model import BasicPointCloud
from gaussian_object.utils.graphics_utils import getWorld2View2, focal2fov

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    cx_offset: np.array = 0
    cy_offset: np.array = 0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

# def fetchPly(path):
#     plydata = PlyData.read(path)
#     vertices = plydata['vertex']
#     positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
#     colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
#     normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
#     return BasicPointCloud(points=positions, colors=colors, normals=normals)

# def storePly(path, xyz, rgb):
#     # Define the dtype for the structured array
#     dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
#     normals = np.zeros_like(xyz)

#     elements = np.empty(xyz.shape[0], dtype=dtype)
#     attributes = np.concatenate((xyz, normals, rgb), axis=1)
#     elements[:] = list(map(tuple, attributes))

#     # Create the PlyData object and write to file
#     vertex_element = PlyElement.describe(elements, 'vertex')
#     ply_data = PlyData([vertex_element])
#     ply_data.write(path)

def readCameras(dataloader, zoom_scale=512, margin=0.0, frame_sample_interval=1):
    cam_infos = []
    use_binarized_mask = dataloader.use_binarized_mask
    bbox3d_diameter = dataloader.bbox3d_diameter
    for frame_idx in range(len(dataloader)):
        if frame_idx % frame_sample_interval != 0:
            continue
        obj_data = dataloader[frame_idx]
        camK = np.array(obj_data['camK'])        
        pose = np.array(obj_data['pose'])
        R = np.transpose(pose[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = pose[:3, 3]

        if 'image_path' not in obj_data:
            image_path = None
            image = (obj_data['image'] * 255).numpy()
            image = Image.fromarray(image.astype(np.uint8))
            image_name = f'{frame_idx}.png'
        else:
            image_path = obj_data['image_path']
            image = Image.open(image_path)
            image_name = os.path.basename(image_path)
            
        image = torch.from_numpy(np.array(image))
        raw_height, raw_width = image.shape[:2]

        out = gs_utils.zoom_in_and_crop_with_offset(image, t=T, K=camK, 
                                            radius=bbox3d_diameter/2, 
                                            margin=margin, target_size=zoom_scale)
        image = out['zoom_image'].squeeze()
        height, width = image.shape[:2]

        # if 'coseg_mask_path' not in obj_data:
            # mask = np.ones((height, width, 1), dtype=np.float32)
        try:
            mask = Image.open(obj_data['coseg_mask_path'])
            mask = torch.from_numpy(np.array(mask, dtype=np.float32)) / 255.0
            mask = gs_utils.zoom_in_and_crop_with_offset(
                mask, t=T, K=camK, radius=bbox3d_diameter/2, 
                margin=margin, target_size=zoom_scale
            )['zoom_image'].squeeze()
            if mask.dim() == 2:
                mask = mask[:, :, None]
            if use_binarized_mask:
                mask = mask.round()
        except Exception as e:
            print(e)
            mask = np.ones((height, width, 1), dtype=np.float32)
        
        image = (image * mask).type(torch.uint8).numpy()
        image = Image.fromarray(image.astype(np.uint8))

        zoom_camk = out['zoom_camK'].squeeze().numpy()
        zoom_offset = out['zoom_offset'].squeeze().numpy()
        cx_offset = zoom_offset[0]
        cy_offset = zoom_offset[1]
        cam_fx = zoom_camk[0, 0]
        cam_fy = zoom_camk[1, 1]
        FovX = focal2fov(cam_fx, width)
        FovY = focal2fov(cam_fy, height)
        
        cam_info = CameraInfo(R=R, T=T, FovY=FovY, FovX=FovX, 
                                cx_offset=cx_offset, cy_offset=cy_offset,
                                uid=frame_idx, image=image,
                                image_path=image_path, image_name=image_name, 
                                width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos


def readObjectInfo(train_dataset, test_dataset, model_path, num_points=4096, zoom_scale=512, margin=0.0, random_points3D=False):

    print(f"Reading {len(train_dataset)}  training image ...")
    train_cam_infos = readCameras(train_dataset, zoom_scale=zoom_scale, margin=margin, frame_sample_interval=1)
    num_training_samples = len(train_cam_infos)
    print(f"{num_training_samples} training samples")
    print(f"-----------------------------------------")

    test_interval = len(test_dataset) // 3
    test_cam_infos = readCameras(test_dataset, zoom_scale=zoom_scale, margin=margin, frame_sample_interval=test_interval)
    num_test_samples = len(test_cam_infos)
    print(f"{num_test_samples} testing samples")
    print(f"----------------------------------------")

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(model_path, "3DGS_points3d.ply")
    if not random_points3D:
        obj_bbox3D = train_dataset.obj_bbox3d
        # obj_bbox3D = np.loadtxt(os.path.join(path, 'corners.txt'))
        min_3D_corner = obj_bbox3D.min(axis=0)
        max_3D_corner = obj_bbox3D.max(axis=0)
        obj_bbox3D_dims = max_3D_corner - min_3D_corner
        grid_cube_size = (np.prod(obj_bbox3D_dims, axis=0) / num_points)**(1/3)
        
        xnum, ynum, znum = np.ceil(obj_bbox3D_dims / grid_cube_size).astype(np.int64)
        xmin, ymin, zmin = min_3D_corner
        xmax, ymax, zmax = max_3D_corner
        zgrid, ygrid, xgrid = np.meshgrid(np.linspace(zmin, zmax, znum),
                                            np.linspace(ymin, ymax, ynum), 
                                            np.linspace(xmin, xmax, xnum), 
                                            indexing='ij')
        xyz = np.stack([xgrid, ygrid, zgrid], axis=-1).reshape(-1, 3)
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    if random_points3D:
        # Since this data set has no colmap data, we start with random points
        # num_pts = 100_000
        print(f"Generating random point cloud ({num_points})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_points, 3)) #* 2.6 - 1.3
        shs = np.random.random((num_points, 3)) #/ 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_points, 3)))

        # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None

    object_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return object_info




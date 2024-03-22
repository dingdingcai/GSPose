import cv2
import math
import torch
import numpy as np
from transforms3d.axangles import axangle2mat


@torch.jit.script
def bbox_to_grid(bbox, in_size, out_size):
    h = in_size[0]
    w = in_size[1]
    xmin = bbox[0].item()
    ymin = bbox[1].item()
    xmax = bbox[2].item()
    ymax = bbox[3].item()
    grid_y, grid_x = torch.meshgrid([
        torch.linspace(ymin / h, ymax / h, out_size[0], device=bbox.device) * 2 - 1,
        torch.linspace(xmin / w, xmax / w, out_size[1], device=bbox.device) * 2 - 1,
    ], indexing='ij')

    return torch.stack((grid_x, grid_y), dim=-1)


@torch.jit.script
def bboxes_to_grid(boxes, in_size, out_size):
    grids = torch.zeros(boxes.size(0), out_size[1], out_size[0], 2, device=boxes.device)
    for i in range(boxes.size(0)):
        box = boxes[i]
        grids[i, :, :, :] = bbox_to_grid(box, in_size, out_size)

    return grids

def GS_zoom_and_crop(image, K, t, radius, target_size=224, margin=0, mode='bilinear'):
    unsqueeze = False
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float()
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    if K.dim() == 2:
        K = K[None, :, :]
    if t.dim() == 1:
        t = t[None, :]
    if image.dim() == 3:
        unsqueeze = True
        image = image[None, ...]
    if image.shape[3] == 3:
        image = image.permute(0, 3, 1, 2)
    
    cam_focal = K[:, :2, :2].max()
    bbox_scale = 2 * (1 + margin) * cam_focal * radius / t[:, 2] 
    rescaling_factor = target_size / bbox_scale
    
    bbox_uvs = torch.einsum('nij,nj->ni', K, t)
    bbox_center = bbox_uvs[:, :2] / bbox_uvs[:, 2:3]    
    
    Smat = torch.eye(3, device=K.device)[None, ...].repeat(K.size(0), 1, 1)
    Smat[:, :2, :2] *= rescaling_factor
    zoom_K = torch.einsum('nij,njk->nik', Smat, K)    
    zoom_K[:, :2, 2] = target_size / 2
    
    zoom_uvs = torch.einsum('nij,nj->ni', zoom_K, t)
    zoom_uvs = zoom_uvs[..., :2] / zoom_uvs[..., 2:3]
    uvs_shift = target_size / 2 - zoom_uvs
    bbox_center += uvs_shift / rescaling_factor
    
    bboxes = torch.zeros(len(bbox_center), 4)
    bboxes[:, 0] = (bbox_center[:, 0] - bbox_scale / 2)
    bboxes[:, 1] = (bbox_center[:, 1] - bbox_scale / 2)
    bboxes[:, 2] = (bbox_center[:, 0] + bbox_scale / 2)
    bboxes[:, 3] = (bbox_center[:, 1] + bbox_scale / 2)
    
    img_hei, img_wid = image.shape[-2:]
    in_size = torch.tensor((img_hei, img_wid))
    out_size = torch.tensor((target_size, target_size))
    grids = bboxes_to_grid(bboxes, in_size, out_size)

    image_new = torch.nn.functional.grid_sample(image.type(torch.float32), 
                                                grids.type(torch.float32), 
                                                mode=mode, align_corners=True)
    image_new = image_new.permute(0, 2, 3, 1)

    if unsqueeze:
        image_new = image_new.squeeze()
        zoom_K = zoom_K.squeeze()
    return image_new, zoom_K

def center_zoom_and_crop_cxcy(image, K, t, radius, target_size=224, margin=0, mode='bilinear'):
    unsqueeze = False
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float()
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
        
    if K.dim() == 2:
        K = K[None, :, :]
    if t.dim() == 1:
        t = t[None, :]
    if image.dim() == 3:
        unsqueeze = True
        image = image[None, ...]
    if image.shape[3] == 3 or image.shape[3] == 1:
        image = image.permute(0, 3, 1, 2)
    
    obj_tz = t[:, 2]
    cam_focal = K[:, :2, :2].max()
    bbox_uvs = torch.einsum('nij,nj->ni', K, t)
    bbox_center = bbox_uvs[:, :2] / bbox_uvs[:, 2:3] 
    bbox_scale = 2 * (1 + margin) * cam_focal * radius / obj_tz 
    rescaling_factor = target_size / bbox_scale
    
    
    Smat = torch.eye(3, device=K.device)[None, ...].repeat(K.size(0), 1, 1)
    Smat[:, :2, :2] *= rescaling_factor
    zoom_K = torch.einsum('nij,njk->nik', Smat, K)    
    zoom_K[:, :2, 2] = target_size / 2
    
    zoom_uvs = torch.einsum('nij,nj->ni', zoom_K, t)
    zoom_uvs = zoom_uvs[..., :2] / zoom_uvs[..., 2:3]
    cxcy_offset = target_size / 2 - zoom_uvs
    cxcy_offset = 2 * cxcy_offset / target_size
    
    bboxes = torch.zeros(len(bbox_center), 4)
    bboxes[:, 0] = (bbox_center[:, 0] - bbox_scale / 2)
    bboxes[:, 1] = (bbox_center[:, 1] - bbox_scale / 2)
    bboxes[:, 2] = (bbox_center[:, 0] + bbox_scale / 2)
    bboxes[:, 3] = (bbox_center[:, 1] + bbox_scale / 2)
    
    img_hei, img_wid = image.shape[-2:]
    in_size = torch.tensor((img_hei, img_wid))
    out_size = torch.tensor((target_size, target_size))
    grids = bboxes_to_grid(bboxes, in_size, out_size)

    image_new = torch.nn.functional.grid_sample(image.type(torch.float32), 
                                                grids.type(torch.float32), 
                                                mode=mode, align_corners=True)
    image_new = image_new.permute(0, 2, 3, 1)
    
    if unsqueeze:
        cxcy_offset = cxcy_offset.squeeze(0)
        image_new = image_new.squeeze(0)
        zoom_K = zoom_K.squeeze(0)
        
    return image_new, zoom_K, cxcy_offset


def zoom_in_and_crop_with_offset(image, K, t, radius, target_size=224, margin=0, mode='bilinear'):
    """
    distortation is preserved
    """
    unsqueeze = False
    if isinstance(radius, float):
        radius = torch.tensor(radius)
    if isinstance(radius, np.ndarray):
        radius = torch.from_numpy(radius).float()
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t).float()
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    if K.dim() == 2:
        K = K[None, :, :]
    if t.dim() == 1:
        t = t[None, :]
    if image.dim() == 2:
        image = image[..., None] # HxWx1
    if image.dim() == 3:
        unsqueeze = True
        image = image[None, ...]
    if image.shape[3] == 1 or image.shape[3] == 3:
        image = image.permute(0, 3, 1, 2)
    
    cam_focal = K[:, :2, :2].max()
    bbox_scale = 2 * (1 + margin) * cam_focal * radius.to(K.device) / t[:, 2].to(K.device) 
    rescaling_factor = target_size / bbox_scale
    
    Smat = torch.eye(3, device=K.device)[None, ...].repeat(K.size(0), 1, 1)
    Smat[:, 0, 0] *= rescaling_factor
    Smat[:, 1, 1] *= rescaling_factor
    zoom_camK = torch.einsum('nij,njk->nik', Smat, K)    
    zoom_camK[:, :2, 2] = target_size / 2

    bbox_uvs = torch.einsum('nij,nj->ni', K, t)
    bbox_center = bbox_uvs[:, :2] / bbox_uvs[:, 2:3]  

    zoom_uvs = torch.einsum('nij,nj->ni', zoom_camK, t)
    zoom_uvs = zoom_uvs[..., :2] / zoom_uvs[..., 2:3]
    uvs_shift = target_size / 2 - zoom_uvs
    zoom_offset = 2 * uvs_shift / target_size

    bboxes = torch.zeros(len(bbox_center), 4)
    bboxes[:, 0] = (bbox_center[:, 0] - bbox_scale / 2)
    bboxes[:, 1] = (bbox_center[:, 1] - bbox_scale / 2)
    bboxes[:, 2] = (bbox_center[:, 0] + bbox_scale / 2)
    bboxes[:, 3] = (bbox_center[:, 1] + bbox_scale / 2)
    
    img_hei, img_wid = image.shape[-2:]
    in_size = torch.tensor((img_hei, img_wid))
    out_size = torch.tensor((target_size, target_size))
    grids = bboxes_to_grid(bboxes, in_size, out_size)

    image_new = torch.nn.functional.grid_sample(
        image.type(torch.float32), grids.type(torch.float32), mode=mode, align_corners=True)
    image_new = image_new.permute(0, 2, 3, 1)
    
    if unsqueeze:
        bbox_center = bbox_center.squeeze()
        bbox_scale = bbox_scale.squeeze()
        
        zoom_offset = zoom_offset.squeeze()
        image_new = image_new.squeeze()
        zoom_camK = zoom_camK.squeeze()
    
    # zoom_tz = 2 * (1 + margin) * cam_focal * radius / target_size
    # zoom_tz = t[:, 2] / rescaling_factor 
    return {
        'zoom_image': image_new,
        'zoom_offset': zoom_offset,
        'zoom_camK': zoom_camK,
        'bbox_scale': bbox_scale,
        'bbox_center': bbox_center,
        # 'canonical_tz': zoom_tz,
    }


def zoom_out_and_uncrop_image(zoom_image, bbox_center, bbox_scale, orig_hei, orig_wid):
    if isinstance(bbox_center, torch.Tensor):
        box_scale = bbox_scale.squeeze().item()
        box_cx = bbox_center.squeeze()[0].item()
        box_cy = bbox_center.squeeze()[1].item()
    else:
        box_scale = bbox_scale.squeeze()
        box_cx = bbox_center.squeeze()[0]
        box_cy = bbox_center.squeeze()[1]
        
    yy, xx = torch.meshgrid([
        torch.linspace(-box_cy, orig_hei - box_cy, orig_hei) / box_scale * 2, 
        torch.linspace(-box_cx, orig_wid - box_cx, orig_wid) / box_scale * 2, 
    ], indexing='ij')
    if zoom_image.dim() == 2:
        zoom_image = zoom_image[None, None, :, :] # SxS -> 1x1xSxS
    elif zoom_image.dim() == 3:
        zoom_image = zoom_image[None, :, :, :]  # 3xSxS -> 1x3xSxS
    
    if zoom_image.shape[-1] == 1 or zoom_image.shape[-1] == 3:
        zoom_image = zoom_image.permute(0, 3, 1, 2) # 1xSxSx3 -> 1x3xSxS
    
    grid = torch.stack([xx, yy], dim=-1)[None, ...]
    orig_image = torch.nn.functional.grid_sample(
        zoom_image,  # BxCxHxW
        grid.type(zoom_image.dtype).to(zoom_image.device),
        mode='bilinear', align_corners=True).permute(0, 2, 3, 1) # BxCxHxW -> BxHxWxC
    return orig_image


def read_numpy_data_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        txtdata = f.readlines()
        data = list()
        for _line in txtdata:
            data.append(list(map(float, _line.strip().split(' '))))
    return np.stack(data, axis=0)

def read_list_data_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        txtdata = f.readlines()
        data = list()
        for _line in txtdata:
            try:
                data.append(float(_line.strip()))
            except:
                data.append(_line.strip())
    return data


def torch_compute_diameter_from_pointcloud(point_cloud, num_points=8192):
    point_cloud = torch.as_tensor(point_cloud, dtype=torch.float32).squeeze()
    assert (point_cloud.dim() == 2), 'Point cloud must be of shape (num_points, 3)'
    num_pts = len(point_cloud)
    fps_inds = torch.randperm(num_pts)[:num_points]
    point_cloud_tensor = point_cloud[fps_inds]

    # Compute pairwise distances using PyTorch
    pairwise_distances = torch.norm(point_cloud_tensor[:, None, :] - point_cloud_tensor[None, :, :], dim=-1)

    # Find the index of the maximum distance in the flattened distance matrix
    max_distance_index = torch.argmax(pairwise_distances)

    # Convert the flattened index to 2D coordinates
    num_points = len(point_cloud)
    max_distance_row = max_distance_index // num_points
    max_distance_col = max_distance_index % num_points

    # Extract the coordinates of the points with the maximum distance
    point1 = point_cloud_tensor[max_distance_row]
    point2 = point_cloud_tensor[max_distance_col]

    # Compute the Euclidean distance between the two points
    diameter = torch.norm(point1 - point2).item()

    return diameter


def onepose_lowtexture_bbox3d_for_drawing(obj_3D_bbox):
    cannon_3D_bbox = obj_3D_bbox.clone()
    cannon_3D_bbox[0] = obj_3D_bbox[0]
    cannon_3D_bbox[1] = obj_3D_bbox[4]
    cannon_3D_bbox[2] = obj_3D_bbox[6]
    cannon_3D_bbox[3] = obj_3D_bbox[2]

    cannon_3D_bbox[4] = obj_3D_bbox[1]
    cannon_3D_bbox[5] = obj_3D_bbox[5]
    cannon_3D_bbox[6] = obj_3D_bbox[7]
    cannon_3D_bbox[7] = obj_3D_bbox[3]
    return cannon_3D_bbox

def linemod_bbox3d_for_drawing(obj_3D_bbox):
    cannon_3D_bbox = obj_3D_bbox.clone()
    cannon_3D_bbox[0] = obj_3D_bbox[0]
    cannon_3D_bbox[1] = obj_3D_bbox[4]
    cannon_3D_bbox[2] = obj_3D_bbox[7]
    cannon_3D_bbox[3] = obj_3D_bbox[3]

    cannon_3D_bbox[4] = obj_3D_bbox[1]
    cannon_3D_bbox[5] = obj_3D_bbox[5]
    cannon_3D_bbox[6] = obj_3D_bbox[6]
    cannon_3D_bbox[7] = obj_3D_bbox[2]
    return cannon_3D_bbox


def draw_3d_bounding_box(rgb_image, projected_bbox, color, linewidth=3):
    """
    Draw a projected 3D bounding box onto an RGB image.

    Parameters:
    - rgb_image: The RGB image as a NumPy array.
    - projected_bbox: The projected 2D coordinates of the 3D bounding box vertices.
                     It should be a list of eight (x, y) coordinates.

    [(x1, y1),  # Front-top-left corner
     (x2, y2),  # Front-top-right corner
     (x3, y3),  # Front-bottom-right corner
     (x4, y4),  # Front-bottom-left corner
     (x5, y5),  # Back-top-left corner
     (x6, y6),  # Back-top-right corner
     (x7, y7),  # Back-bottom-right corner
     (x8, y8)]  # Back-bottom-left corner

    Returns:
    - An RGB image with the 3D bounding box drawn.
    """

    # Convert the RGB image to BGR (OpenCV uses BGR format)
    image_with_bbox = rgb_image.copy()
#     image_with_bbox = cv2.cvtColor(rgb_image.copy(), cv2.COLOR_RGB2BGR)

    # Define the edges of the bounding box by connecting the vertices
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]

    # Draw the edges of the bounding box on the image
    for edge in edges:
        start_point = projected_bbox[edge[0]]
        end_point = projected_bbox[edge[1]]
        start_point = tuple(map(int, start_point))
        end_point = tuple(map(int, end_point))
        cv2.line(image_with_bbox, start_point, end_point, color, linewidth)  # Green color

#     return cv2.cvtColor(image_with_bbox, cv2.COLOR_BGR2RGB)
    return image_with_bbox


def egocentric_to_allocentric(ego_pose, cam_ray=(0, 0, 1.0)):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    assert(ego_pose.shape[-1] == 4), "ego_pose should be a 3x4 or 4x4 matrix"
    trans = ego_pose[:3, 3]
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))
    # Rotate back by that amount
    if angle > 0:
        allo_pose = np.zeros((3, 4), dtype=ego_pose.dtype)
        allo_pose[:3, 3] = trans
        rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=-angle)
        allo_pose[:3, :3] = np.dot(rot_mat, ego_pose[:3, :3])
    else:  # ego tp allo
        allo_pose = ego_pose.copy()
    return allo_pose

def allocentric_to_egocentric(allo_pose, cam_ray=(0, 0, 1.0)):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    assert(allo_pose.shape[-1] == 4), "allo_pose should be a 3x4 or 4x4 matrix"

    trans = allo_pose[:3, 3]
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))
    if angle > 0:
        ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
        ego_pose[:3, 3] = trans
        rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
        ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
    else:  # allo to ego
        ego_pose = allo_pose.copy()
    return ego_pose
    

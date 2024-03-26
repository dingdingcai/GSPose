import cv2
import math
import torch
import functools
import numpy as np
from itertools import groupby
import torch.nn.functional as F
import pycocotools.mask as cocomask
from pytorch3d import io as pyt3d_io
from pytorch3d import structures as pyt3d_struct
from pytorch3d import renderer as pyt3d_renderer
from pytorch3d import transforms as py3d_transform
from pytorch3d.transforms import euler_angles_to_matrix
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc_utils import gs_utils

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_size, (int, float)):
        output_size = (output_size, output_size)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def read_obj_ply_as_mesh(filename, return_bbox=False, return_diameter=False):
    assert(filename.endswith('.ply')), 'can only read .ply file'
    verts, faces = pyt3d_io.load_ply(filename) # Nx3
    features = torch.ones_like(verts) # Nx3
    tex = pyt3d_renderer.TexturesVertex(verts_features=features[None, ...])
    obj_mesh_model = pyt3d_struct.Meshes(verts=verts[None, ...], 
                                        faces=faces[None, ...], 
                                        textures=tex)

    if return_diameter or return_bbox:
        # dx, dy, dz = verts.max(dim=0).values - verts.min(dim=0).values
        # obj_diameter = (dx**2 + dy**2 + dz**2)**0.5
        mesh_extent = verts.max(dim=0).values - verts.min(dim=0).values # 3
        return obj_mesh_model, mesh_extent                        

    return obj_mesh_model

def resize_short_edge(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR, return_scale=False):
    """Scale the shorter edge to the given size, with a limit of `max_size` on
    the longer edge. If `max_size` is reached, then downscale so that the
    longer edge does not exceed max_size. only resize input image to target
    size and return scale.

    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        if return_scale:
            return im, im_scale
        else:
            return im
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        if return_scale:
            return padded_im, im_scale
        else:
            return padded_im

def flat_dataset_dicts(dataset_dicts):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    new_dicts = []
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        if "annotations" in dataset_dict:
            for inst_id, anno in enumerate(dataset_dict["annotations"]):
                rec = {"inst_id": inst_id, "inst_infos": anno}
                rec.update(img_infos)
                new_dicts.append(rec)
        else:
            rec = img_infos
            new_dicts.append(rec)
    return new_dicts

def lazy_property(function):
    # https://danijar.com/structuring-your-tensorflow-models/
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def binary_mask_to_rle(mask, compressed=True):
    """
    encode mask image to save storage space
    """
    mask = mask.astype(np.uint8)
    if compressed:
        rle = cocomask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
    else:
        rle = {"counts": [], "size": list(mask.shape)}
        counts = rle.get("counts")
        for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):  # noqa: E501
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
    return rle

def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask

def segmToRLE(segm, h, w):
    """Convert segmentation which can be polygons, uncompressed RLE to RLE.

    :return: binary mask (numpy 2D array)
    """
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = cocomask.frPyObjects(segm, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = cocomask.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle

def cocosegm2mask(segm, h, w):
    rle = segmToRLE(segm, h, w)
    mask = rle2mask(rle, h, w)
    return mask

def aug_bbox_DZI(bbox_xyxy, im_H, im_W, scale_ratio=0.0, shift_ratio=0.0, pad_ratio=0.0):
    """Used for DZI, the augmented box is a square (maybe enlarged)
    Args:
        bbox_xyxy (np.ndarray):
    Returns:
            center, scale
    """
    x1, y1, x2, y2 = bbox_xyxy.copy()
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1
    
    scale_ratio = 1 + scale_ratio * (2 * np.random.random_sample() - 1) # [1-0.25, 1+0.25]
    shift_ratio = shift_ratio * (2 * np.random.random_sample(2) - 1)    # [-0.25, 0.25]
    bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]], dtype=np.float32)  # (h/2, w/2)
    pad_ratio = 1 + np.random.random_sample() * pad_ratio # [1, 1.5]
    scale = max(y2 - y1, x2 - x1) * scale_ratio * pad_ratio
    # if enable_dynamic_pad:
    #     dyn_pad_scale = 1 + np.random.random_sample() * 0.5 # [1, 1.5]
    #     scale = max(y2 - y1, x2 - x1) * scale_ratio * dyn_pad_scale
    # else:
    #     scale = max(y2 - y1, x2 - x1) * scale_ratio * pad_scale
    scale = min(scale, max(im_H, im_W)) * 1.0
    return bbox_center, scale

def evenly_distributed_rotation(n, random_seed=None):
    """
    uniformly sample N examples on a sphere
    """
    def normalize(vector, dim: int = -1):
        return vector / torch.norm(vector, p=2.0, dim=dim, keepdim=True)
    
    indices = torch.arange(0, n, dtype=torch.float32) + 0.5
    phi = torch.acos(1 - 2 * indices / n)
    theta = math.pi * (1 + 5 ** 0.5) * indices
    points = torch.stack([
        torch.cos(theta) * torch.sin(phi), 
        torch.sin(theta) * torch.sin(phi), 
        torch.cos(phi),], dim=1)
    forward = -points
    
    if random_seed is not None:
        torch.manual_seed(random_seed) # fix the sampling of viewpoints for reproducing evaluation
    
    down = normalize(torch.randn(n, 3), dim=1)
    right = normalize(torch.cross(down, forward))
    down = normalize(torch.cross(forward, right))
    R_mat = torch.stack([right, down, forward], dim=1)
    return R_mat

def uniform_z_rotation(n, eps_degree=0, rang_degree=180):
    """
    uniformly sample N examples range from 0 to 360
    """
    assert n > 0, "sample number must be nonzero"
    eps_rad = eps_degree / 180.0 * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    z_radians = (torch.arange(n) + 1)/(n + 1) * rang_degree/180.0 * math.pi * 2 # -pi, pi
    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_rotation_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_rotation_matrix

def random_z_rotation(n, eps_degree=0, rang_degree=180):
    """
    randomly sample N examples range from 0 to 360
    """
    eps_rad = eps_degree / 180. * math.pi
    rang_rad = rang_degree / 180 * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad # -eps, eps
    z_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * rang_rad # -pi, pi
    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_euler_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_euler_matrix 

def random_xy_rotation(n, eps_degree=0, rang_degree=180):
    """
    randomly sample N examples range from 0 to 360
    """
    eps_rad = eps_degree / 180. * math.pi
    rang_rad = rang_degree / 180 * math.pi
    x_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * rang_rad # -pi, pi
    y_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * rang_rad # -pi, pi
    
    z_radians = (torch.rand(n, dtype=torch.float32) * 2.0 - 1.0) * eps_rad  # -eps, eps

    target_euler_radians = torch.stack([x_radians, y_radians, z_radians], dim=-1)
    target_euler_matrix = euler_angles_to_matrix(target_euler_radians, "XYZ")
    return target_euler_matrix 

def rotate_batch(batch: torch.Tensor):  # (..., H, H) -> (4, ..., H, H)
    assert batch.shape[-1] == batch.shape[-2]
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(batch)
    return torch.stack([
        batch,  # 0 deg
        torch.flip(batch, [-2]).transpose(-1, -2),  # 90 deg
        torch.flip(batch, [-1, -2]),                # 180 deg
        torch.flip(batch, [-1]).transpose(-1, -2),  # 270 deg
    ])  # (4, ..., H, H)

def rotate_batch_back(batch: torch.Tensor):  # (4, ..., H, H) -> (4, ..., H, H)
    assert batch.shape[0] == 4
    assert batch.shape[-1] == batch.shape[-2]
    if not isinstance(batch, torch.Tensor):
        batch = torch.as_tensor(batch)
    return torch.stack([
        batch[0],  # 0 deg
        torch.flip(batch[1], [-1]).transpose(-1, -2),  # -90 deg
        torch.flip(batch[2], [-1, -2]),                # -180 deg
        torch.flip(batch[3], [-2]).transpose(-1, -2),  # -270 deg
    ])  # (4, ..., H, H

def convert_TxTyTz_to_delta_PxPyTz(T3, camK, bbox_center, bbox_scale, zoom_scale):
    """
    convert absolute 3D location to SITE (scale-invariant-translation-estimation)
    """
    if not isinstance(T3, torch.Tensor):
        T3 = torch.as_tensor(T3, dtype=torch.float32)
    
    if not isinstance(camK, torch.Tensor):
        camK = torch.as_tensor(camK, dtype=torch.float32)
    
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)

    unsqueeze = False
    if T3.dim() == 1:
        unsqueeze = True
        T3 = T3[None, ...]
    if camK.dim() == 2:
        camK = camK[None, ...]
    if bbox_center.dim() == 1:
        bbox_center = bbox_center[None, ...]

    if bbox_scale.dim() == 0:
        bbox_scale = bbox_scale[None, ...]

    assert (bbox_center.dim() == 2 and len(T3) == len(bbox_center))
    assert (T3.dim() == 2 and camK.dim() == 3 and len(T3) == len(camK))
    assert (bbox_scale.dim() == 1)

    Kt = (camK @ T3.unsqueeze(2)).squeeze(2) # 1x3x3 @ 1x3x1 => 1x3
    obj_pxpy = Kt[:, :2] / Kt[:, 2]

    delta_pxpy = (obj_pxpy - bbox_center) / bbox_scale
    delta_tz = T3[:, -1] / zoom_scale * bbox_scale

    if unsqueeze:
        delta_tz = delta_tz.squeeze(0)
        delta_pxpy = delta_pxpy.squeeze(0)
        
    return delta_pxpy, delta_tz

def recover_TxTyTz_from_delta_PxPyTz(delta_pxpy, delta_tz, camK, bbox_center, bbox_scale, zoom_scale):
    if not isinstance(delta_tz, torch.Tensor):
        delta_tz = torch.as_tensor(delta_tz, dtype=torch.float32)
    
    if not isinstance(delta_pxpy, torch.Tensor):
        delta_pxpy = torch.as_tensor(delta_pxpy, dtype=torch.float32)
    
    if not isinstance(camK, torch.Tensor):
        camK = torch.as_tensor(camK, dtype=torch.float32)
    
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)

    unsqueeze = False
    if delta_tz.dim() == 0:
        unsqueeze = True
        delta_tz = delta_tz[None, ...]

    if delta_pxpy.dim() == 1:
        delta_pxpy = delta_pxpy[None, ...]
    
    if camK.dim() == 2:
        camK = camK[None, ...]
    if bbox_center.dim() == 1:
        bbox_center = bbox_center[None, ...]

    if bbox_scale.dim() == 0:
        bbox_scale = bbox_scale[None, ...]

    Tz = delta_tz * zoom_scale / bbox_scale # V

    obj_pxpy = delta_pxpy * bbox_scale[..., None] + bbox_center # Vx2, Vx1, Vx2 => Vx2

    homo_pxpy = torch.cat([obj_pxpy, torch.ones_like(Tz)[..., None]], dim=1) # Vx2, Vx1 => Vx3
    T3 = Tz[..., None].float() * (torch.inverse(camK).float() @ homo_pxpy[..., None].float()).squeeze(dim=2) # Vx1 [Vx3x3, Vx3x1 => Vx3]
    
    if unsqueeze:
        T3 = T3.squeeze(0)
    
    return T3

def batch_recover_TxTyTz_from_delta_PxPyTz(delta_pxpy, delta_tz, camK, bbox_center, bbox_scale, zoom_scale):
    zoom_scale = 448
    if not isinstance(camK, torch.Tensor):
        camK = torch.Tensor(camK)
        
    assert(delta_pxpy.size(0) == delta_tz.size(0))
    assert(delta_pxpy.dim() == 2 and delta_pxpy.shape[1] == 2), delta_pxpy.shape
        
    Tz = delta_tz * zoom_scale / bbox_scale # V
    obj_pxpy = delta_pxpy * bbox_scale[..., None] + bbox_center # Vx2, Vx1, Vx2 => Vx2
    homo_pxpy = torch.cat([obj_pxpy, torch.ones_like(Tz)[..., None]], dim=1) # Vx2, Vx1 => Vx3
    T3 = Tz[..., None].float() * (torch.inverse(camK).float() @ homo_pxpy[..., None].float()).squeeze(dim=2) # Vx1 [Vx3x3, Vx3x1 => Vx3]
    return T3

def generate_PEmap(im_hei, im_wid, cam_K):
    if not isinstance(cam_K, torch.Tensor):
        cam_K = torch.as_tensor(cam_K)
    K_inv = torch.inverse(cam_K)
    try:
        yy, xx = torch.meshgrid(torch.arange(im_hei),torch.arange(im_wid), indexing='ij')
    except:
        yy, xx = torch.meshgrid(torch.arange(im_hei),torch.arange(im_wid))

    homo_uv = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).type(torch.float32)
    homo_uvk = (K_inv @ homo_uv.view(3, -1)).view(3, im_hei, im_wid) # 3xHxW
    homo_uv = homo_uvk[:2] # 2xHxW

    return homo_uv

def transform_to_local_ROIcrop(bbox_center, bbox_scale, zoom_scale=256, centerize_principle_point=True):
    """
    transformation from original image to the object-centric crop region
    """
    if not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.as_tensor(bbox_center, dtype=torch.float32)
    
    if not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.as_tensor(bbox_scale, dtype=torch.float32)

    unsqueeze = False
    if bbox_center.dim() == 1:
        bbox_center = bbox_center[None, ...] # Nx2
        bbox_scale = bbox_scale[None, ...]   # N
        unsqueeze = True
    assert(len(bbox_center) == len(bbox_scale))

    Ts_B2X = list()
    cx, cy = 0.0, 0.0
    if centerize_principle_point:
        cx, cy = zoom_scale / 2, zoom_scale / 2  
    for bxby, bs in zip(bbox_center, bbox_scale):
        r = zoom_scale / bs
        T_b2x = torch.tensor([
            [r, 0, cx-r * bxby[0]], 
            [0, r, cy-r * bxby[1]], 
            [0, 0, 1]]) 
        Ts_B2X.append(T_b2x)

    Ts_B2X = torch.stack(Ts_B2X, dim=0)
    if unsqueeze:
        Ts_B2X = Ts_B2X.squeeze(0)
    return Ts_B2X

def rotation_from_Allo2Ego(obj_ray, cam_ray=torch.tensor([0.0, 0.0, 1.0])):
    if not isinstance(obj_ray, torch.Tensor):
        obj_ray = torch.as_tensor(obj_ray, dtype=torch.float32)
    
    unsqueeze = False
    if obj_ray.dim() == 1:
        unsqueeze = True
        obj_ray = obj_ray[None, ...]
    
    dim_B = obj_ray.shape[0]

    if obj_ray.dim() == 2 and cam_ray.dim() == 1:
        cam_ray = cam_ray[None, ...].repeat(dim_B, 1)

    assert(obj_ray.shape == cam_ray.shape), ' {} vs {} are mismatched.'.format(obj_ray.shape, cam_ray.shape)

    
    device = obj_ray.device
    cam_ray = cam_ray.to(device)

    obj_ray = F.normalize(obj_ray, dim=1, p=2) 
    cam_ray = F.normalize(cam_ray, dim=1, p=2) 
    r_vec = torch.cross(cam_ray, obj_ray, dim=1) 
    scalar = torch.sum(cam_ray * obj_ray, dim=1) 
    r_mat = torch.zeros((dim_B, 3, 3)).to(device)

    r_mat[:, 0, 1] = -r_vec[:, 2]
    r_mat[:, 0, 2] =  r_vec[:, 1]
    r_mat[:, 1, 0] =  r_vec[:, 2]
    r_mat[:, 1, 2] = -r_vec[:, 0]
    r_mat[:, 2, 0] = -r_vec[:, 1]
    r_mat[:, 2, 1] =  r_vec[:, 0]

    norm_r_mat2 = r_mat @ r_mat / (1 + scalar[..., None, None].repeat(1, 3, 3).to(device))  # Bx3x3
    Rc = torch.eye(3)[None, ...].to(device).repeat(dim_B, 1, 1) + r_mat + norm_r_mat2
    if unsqueeze:
        Rc = Rc.squeeze(0)
    return Rc

def inplane_augmentation(img, mask=None):
    Rz_index = torch.randperm(4)[0] # 0:0˚, 1:90˚, 2:180˚, 3:270˚
    Rz_rad = torch.tensor([0.0, 0.0, math.pi * Rz_index * 0.5]) # 0˚, 90˚, 180˚, 270˚
    Rz_mat = euler_angles_to_matrix(Rz_rad, 'XYZ').type(torch.float32)
    NOT_TORCH_TENSOR = False
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
        NOT_TORCH_TENSOR = True
        if mask is not None:
            mask = torch.as_tensor(mask)
    ##### rotate the corresponding RGB, Mask, rotation, object projection
    
    assert(img.dim() == 3)
    if img.shape[-1] == 3:
        img = img.permute(2, 0, 1)
    
    if Rz_index == 0:
        Rz_img = img
        if mask is not None:
            Rz_mask = mask
    elif Rz_index == 1:
        Rz_img = torch.flip(img, [-2]).transpose(-1, -2)   # 90 deg
        if mask is not None:
            Rz_mask = torch.flip(mask, [-2]).transpose(-1, -2)   # 90 deg
    elif Rz_index == 2:
        Rz_img = torch.flip(img, [-1, -2])                 # 180 deg
        if mask is not None:
            Rz_mask = torch.flip(mask, [-1, -2])        
    elif Rz_index == 3:
        Rz_img = torch.flip(img, [-1]).transpose(-1, -2)   # 270 deg
        if mask is not None:
            Rz_mask = torch.flip(mask, [-1]).transpose(-1, -2)   # 270 deg
    Rz_img = Rz_img.permute(1, 2, 0)

    if NOT_TORCH_TENSOR:
        Rz_img = Rz_img.numpy()
        Rz_mat = Rz_mat.numpy()
        if mask is not None:
            Rz_mask = Rz_mask.numpy()
    
    if mask is not None:
        return Rz_img, Rz_mask, Rz_mat
    else:
        return Rz_img, Rz_mat

def flat_dataset_dicts_objectwise(dataset_dicts, visib_fract_threshold=None, return_stats=False):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    # new_dicts = []
    object_dicts = {}
    instance_counter = 0
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        for anno in dataset_dict["annotations"]:
            inst_objID = anno['objID'] 
            visib_fract = anno['visib_fract']
            if visib_fract_threshold is not None and visib_fract < visib_fract_threshold:
                continue
            rec = {"inst_annos": anno}
            rec.update(img_infos)
            if inst_objID not in object_dicts:
                object_dicts[inst_objID] = dict()
                object_dicts[inst_objID]['data'] = list()
                object_dicts[inst_objID]['viewpoint'] = list()
                object_dicts[inst_objID]['amount'] = 0
            object_dicts[inst_objID]['amount'] += 1
            object_dicts[inst_objID]['data'].append(rec)
            object_dicts[inst_objID]['viewpoint'].append(anno['viewpoint'])
            instance_counter += 1
    if return_stats:
        return object_dicts, instance_counter
    return object_dicts

def flat_dataset_dicts_with_NN_viewpoint(dataset_dicts, visib_fract_threshold=None, min_covisib_viewpoint=15, max_covisib_viewpoint=60):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    # new_dicts = []
    object_lists = list()
    object_dicts = dict()
    # instance_counter = 0
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        for anno in dataset_dict["annotations"]:
            inst_objID = anno['objID'] 
            visib_fract = anno['visib_fract']
            if visib_fract_threshold is not None and visib_fract < visib_fract_threshold:
                continue
            rec = {"inst_annos": anno}
            rec.update(img_infos)
            object_lists.append(rec)
            if inst_objID not in object_dicts:
                object_dicts[inst_objID] = dict()
                object_dicts[inst_objID]['data'] = list()
                object_dicts[inst_objID]['viewpoint'] = list()
                object_dicts[inst_objID]['amount'] = 0
            object_dicts[inst_objID]['amount'] += 1
            object_dicts[inst_objID]['data'].append(rec)
            object_dicts[inst_objID]['viewpoint'].append(anno['viewpoint'])
            # instance_counter += 1

    for inst_objID in object_dicts:
        object_dicts[inst_objID]['viewpoint'] = np.stack(object_dicts[inst_objID]['viewpoint'], axis=0)
    eps = 1e-8
    for data_entry in object_lists:
        inst_objID = data_entry['inst_annos']['objID']
        inst_viewpoint = np.stack(data_entry['inst_annos']['viewpoint'], axis=0)
        viewpoint_deg_dists = np.arccos(
            np.clip(   (inst_viewpoint[None, :] * object_dicts[inst_objID]['viewpoint']).sum(-1), 
                        a_min=-1.0+eps, a_max=1.0-eps)
            ) * 180 / np.pi
        covisib_viewpoint_inds = (viewpoint_deg_dists >= min_covisib_viewpoint) & (viewpoint_deg_dists <= max_covisib_viewpoint)
        data_entry['inst_annos']['NN_viewpoints'] = covisib_viewpoint_inds.astype(np.float32).nonzero()[0]
    return object_lists, object_dicts

def flat_dataset_dicts_with_viewpoint(dataset_dicts, min_amount_of_instances=1000, visib_fract_threshold=None):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    # new_dicts = []
    object_lists = list()
    object_dicts = dict()
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        for anno in dataset_dict["annotations"]:
            inst_objID = anno['objID'] 
            visib_fract = anno['visib_fract']
            if visib_fract_threshold is not None and visib_fract < visib_fract_threshold:
                continue
            rec = {"inst_annos": anno}
            rec.update(img_infos)
            object_lists.append(rec)
            if inst_objID not in object_dicts:
                object_dicts[inst_objID] = dict()
                object_dicts[inst_objID]['data'] = list()
                object_dicts[inst_objID]['viewpoint'] = list()
                object_dicts[inst_objID]['amount'] = 0
            object_dicts[inst_objID]['amount'] += 1
            object_dicts[inst_objID]['data'].append(rec)
            object_dicts[inst_objID]['viewpoint'].append(anno['viewpoint'])
    
    excluding_objIDs = list()
    new_object_dicts = dict()
    for inst_objID in object_dicts:
        object_dicts[inst_objID]['viewpoint'] = np.stack(object_dicts[inst_objID]['viewpoint'], axis=0)
        object_dicts[inst_objID]['amount'] = len(object_dicts[inst_objID]['viewpoint'])
        if object_dicts[inst_objID]['amount'] < min_amount_of_instances:
            excluding_objIDs.append(inst_objID)
        else:
            new_object_dicts[inst_objID] = object_dicts[inst_objID]
    new_object_lists = list() 
    for entry in object_lists:
        if entry['inst_annos']['objID'] not in excluding_objIDs:
            new_object_lists.append(entry)

    return new_object_lists, new_object_dicts

def flat_dataset_dicts_with_pose(dataset_dicts, min_amount_of_instances=1000, visib_fract_threshold=None):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    # new_dicts = []
    object_lists = list()
    object_dicts = dict()
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        for anno in dataset_dict["annotations"]:
            inst_objID = anno['objID'] 
            visib_fract = anno['visib_fract']
            if visib_fract_threshold is not None and visib_fract < visib_fract_threshold:
                continue
            rec = {"inst_annos": anno}
            rec.update(img_infos)
            object_lists.append(rec)
            if inst_objID not in object_dicts:
                object_dicts[inst_objID] = dict()
                object_dicts[inst_objID]['data'] = list()
                object_dicts[inst_objID]['pose'] = list()
                object_dicts[inst_objID]['amount'] = 0
            object_dicts[inst_objID]['amount'] += 1
            object_dicts[inst_objID]['data'].append(rec)
            object_dicts[inst_objID]['pose'].append(anno['pose'])
    
    excluding_objIDs = list()
    new_object_dicts = dict()
    for inst_objID in object_dicts:
        object_dicts[inst_objID]['pose'] = np.stack(object_dicts[inst_objID]['pose'], axis=0)
        object_dicts[inst_objID]['amount'] = len(object_dicts[inst_objID]['pose'])
        if object_dicts[inst_objID]['amount'] < min_amount_of_instances:
            excluding_objIDs.append(inst_objID)
        else:
            new_object_dicts[inst_objID] = object_dicts[inst_objID]
    new_object_lists = list() 
    for entry in object_lists:
        if entry['inst_annos']['objID'] not in excluding_objIDs:
            new_object_lists.append(entry)

    return new_object_lists, new_object_dicts

def flat_dataset_dicts_with_allo_pose(dataset_dicts, min_amount_of_instances=1000, visib_fract_threshold=None):
    """
    flatten the dataset dicts of detectron2 format
    original: list of dicts, each dict contains some image-level infos
              and an "annotations" field for instance-level infos of multiple instances
    => flat the instance level annotations
    flat format:
        list of dicts,
            each dict includes the image/instance-level infos
            an `inst_id` of a single instance,
            `inst_infos` includes only one instance
    """
    # new_dicts = []
    object_lists = list()
    object_dicts = dict()
    for dataset_dict in dataset_dicts:
        img_infos = {_k: _v for _k, _v in dataset_dict.items() if _k not in ["annotations"]}
        for anno in dataset_dict["annotations"]:
            inst_objID = anno['objID'] 
            visib_fract = anno['visib_fract']
            pose = anno['pose']
            allo_pose = pose.copy()
            allo_pose[:3, :3] = gs_utils.egocentric_to_allocentric(pose)[:3, :3]

            if visib_fract_threshold is not None and visib_fract < visib_fract_threshold:
                continue
            rec = {"inst_annos": anno}
            rec.update(img_infos)
            object_lists.append(rec)
            if inst_objID not in object_dicts:
                object_dicts[inst_objID] = dict()
                object_dicts[inst_objID]['data'] = list()
                object_dicts[inst_objID]['pose'] = list()
                object_dicts[inst_objID]['allo_pose'] = list()
                object_dicts[inst_objID]['amount'] = 0
            object_dicts[inst_objID]['amount'] += 1
            object_dicts[inst_objID]['data'].append(rec)
            object_dicts[inst_objID]['pose'].append(pose)
            object_dicts[inst_objID]['allo_pose'].append(allo_pose)

    excluding_objIDs = list()
    new_object_dicts = dict()
    for inst_objID in object_dicts:
        object_dicts[inst_objID]['amount'] = len(object_dicts[inst_objID]['pose'])
        object_dicts[inst_objID]['pose'] = np.stack(object_dicts[inst_objID]['pose'], axis=0)
        object_dicts[inst_objID]['allo_pose'] = np.stack(object_dicts[inst_objID]['allo_pose'], axis=0)
        if object_dicts[inst_objID]['amount'] < min_amount_of_instances:
            excluding_objIDs.append(inst_objID)
        else:
            new_object_dicts[inst_objID] = object_dicts[inst_objID]
    new_object_lists = list() 
    for entry in object_lists:
        if entry['inst_annos']['objID'] not in excluding_objIDs:
            new_object_lists.append(entry)

    return new_object_lists, new_object_dicts


def inplane_augment(R, return_degree=False):
    unsqueeze = False
    if R.dim() == 2:
        unsqueeze = True
        R = R.unsqueeze(0)
    assert(R.dim() == 3)
    Rz_rad = (2 * torch.rand((len(R), 3)) - 1) * torch.pi
    Rz_rad[:, :2] = 0
    Rz_mat = py3d_transform.euler_angles_to_matrix(Rz_rad, 'XYZ')
    R = torch.einsum('bij,bjk->bik', Rz_mat.transpose(-1, -2), R)
    if unsqueeze:
        R = R.squeeze(0)
        Rz_mat = Rz_mat.squeeze(0)
        Rz_rad = Rz_rad.squeeze(0)
    if return_degree:
        return R, Rz_mat, Rz_rad[-1] / torch.pi * 180
    return R, Rz_mat

def read_text_data(file, delimiter=' '):
    assert(file.endswith('.txt')), f'{file} is not a .txt file'
    assert(delimiter == ' ' or 
           delimiter == ',' or 
           delimiter == ':' or 
           delimiter == ', ' or 
           delimiter == ': '
          )
    data = list()
    with open(file, 'r') as f:
        for line in f.readlines():
            dat = np.array(list(map(float, line.strip().split(delimiter))))
            data.append(dat)
    data = np.stack(data, axis=0)
    return data

def torch_find_connected_component(mask, return_bbox=True, min_bbox_scale=14, include_supmask=True): 
    try:
        from cc_torch import connected_components_labeling 
        assert(isinstance(mask, torch.Tensor))
        assert(mask.ndim == 2), 'mask: {}'.format(mask.shape)
        hei, wid = mask.shape
        pad_hei = hei
        pad_wid = wid
        if pad_hei % 2 == 1:
            pad_hei += 1
        if pad_wid % 2 == 1:
            pad_wid += 1
        pad_mask = torch.zeros(pad_hei, pad_wid, dtype=mask.dtype, device=mask.device)
        pad_mask[:hei, :wid] = mask
        area_labels = connected_components_labeling(pad_mask)
        area_labs = area_labels.unique()
        num_areas = area_labs.size(0)   # bg_mask + fg_masks 
        if num_areas <= 2:
            binary_masks = mask[None, :, :]
        else:
            binary_masks = area_labels[None, :, :].repeat(num_areas, 1, 1) == area_labs[:, None, None]
            area_pixs = binary_masks.view(num_areas, -1).sum(dim=1)
            valid_area_ind = area_pixs > min_bbox_scale**2
            while sum(valid_area_ind) == 0:
                min_bbox_scale = min_bbox_scale - 1
                valid_area_ind = area_pixs > min_bbox_scale**2
            area_labs = area_labs[valid_area_ind]
            binary_masks = binary_masks[valid_area_ind][:, :hei, :wid]
            binary_masks[0] = mask # replace the background with the orginal mask
            if not include_supmask:
                binary_masks = binary_masks[1:]
            
    except Exception as e:
        type_cast = False
        if isinstance(mask, torch.Tensor):
            device = mask.device
            mask = mask.detach().cpu().numpy()
            type_cast = True
        print(e)
        print('using the cv2.connectedComponents from opencv')
        _, area_labels = cv2.connectedComponents(mask)
        area_labels = torch.as_tensor(area_labels)
        pix_counts = torch.bincount(area_labels.flatten())
        area_pixs, area_inds = torch.topk(pix_counts, k=len(pix_counts), dim=0)
        if len(area_inds) <= 2: # bg + fg
            binary_masks = mask[None, :, :]
        else:
            valid_area_ind = area_pixs > min_bbox_scale**2
            while sum(valid_area_ind) == 0:
                min_bbox_scale = min_bbox_scale - 1
                valid_area_ind = area_pixs > min_bbox_scale**2
            area_labs = area_inds[valid_area_ind]
            label_masks = torch.zeros(len(area_labs), *mask.shape) + area_labs[:, None, None]
            binary_masks = label_masks == area_labels[None, ...]
            if type_cast:
                binary_masks = torch.as_tensor(binary_masks, device=device)
            binary_masks[0] = torch.as_tensor(mask)  
            if not include_supmask:
                binary_masks = binary_masks[1:]          
        
    if return_bbox:
        det_scales = list()
        det_centers = list()
        det_masks = list()
        for bin_mask in binary_masks:
            y1, x1 = torch.nonzero(bin_mask).min(0).values
            y2, x2 = torch.nonzero(bin_mask).max(0).values
            box_center = torch.as_tensor([(x2 + x1)/2.0, (y2 + y1)/2.0])
            box_scale = max(x2 - x1, y2 - y1)
            if box_scale > min_bbox_scale:
                det_masks.append(bin_mask)
                det_scales.append(box_scale)
                det_centers.append(box_center.to(box_scale.device))
        det_masks = torch.stack(det_masks, dim=0).type(torch.float32)
        det_scales = torch.stack(det_scales, dim=0).type(torch.float32)
        det_centers = torch.stack(det_centers, dim=0).type(torch.float32)
        
        return det_masks, det_scales, det_centers
    
    return binary_masks


def calc_roi_intrinsic(K, t, camera_dist, zoom_scale=224):
    if t.dim() == 1:
        t = t[None, :]
        
    proj_Kt = torch.einsum('ij,kj->ki', K, t)
    
    box_ratio = proj_Kt[:, -1] / camera_dist
    box_center = proj_Kt[:, :2] / proj_Kt[:, 2:3]
    
    T_cam2roi = torch.eye(3)[None, :, :].repeat(t.size(0), 1, 1)
        
    T_cam2roi[:, 0, 0] = box_ratio
    T_cam2roi[:, 1, 1] = box_ratio
    
    T_cam2roi[:, 0, 2] = zoom_scale/2 - box_ratio * box_center[:, 0]
    T_cam2roi[:, 1, 2] = zoom_scale/2 - box_ratio * box_center[:, 1]
    roi_camK = torch.einsum('nij,jk->nik', T_cam2roi, K)
    
    return roi_camK

def calc_roi_intrinsic_batchview(K, t, camera_dist, zoom_scale=224):
    """
    K: Bx3x3
    t: Bx3
    """
    unsqueeze = False
    if t.dim() == 1:
        t = t[None, :]
    if K.dim() == 2:
        K = K[None, :, :]
        unsqueeze = True
    assert(K.size(0) == t.size(0))
    proj_Kt = torch.einsum('bij,bj->bi', K, t)
    box_center = proj_Kt[..., :2] / proj_Kt[..., 2:3]
    box_rescaling_ratio = proj_Kt[..., -1] / camera_dist
    T_cam2roi = torch.eye(3)[None,  :, :].repeat(t.size(0), 1, 1) # Bx3x3
        
    T_cam2roi[..., 0, 0] = box_rescaling_ratio
    T_cam2roi[..., 1, 1] = box_rescaling_ratio
    
    T_cam2roi[..., 0, 2] = zoom_scale/2 - box_rescaling_ratio * box_center[..., 0]
    T_cam2roi[..., 1, 2] = zoom_scale/2 - box_rescaling_ratio * box_center[..., 1]
    roi_camK = torch.einsum('bij,bjm->bim', T_cam2roi, K)
    if unsqueeze:
        roi_camK = roi_camK.squeeze(0)
    return roi_camK

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

def zoom_and_crop(image, K, t, radius, target_size=224, margin=0, return_canonical_tz=False, mode='bilinear'):
    
    if K.dim() == 2:
        K = K[None, :, :]
    if t.dim() == 1:
        t = t[None, :]
    if image.dim() == 3:
        image = image[None, ...]
    if image.shape[3] == 3:
        image = image.permute(0, 3, 1, 2)
    
    cam_focal = K[:, :2, :2].max()
    obj_tz = t[:, 2]
    
    
    bbox_scale = 2 * (1 + margin) * cam_focal * radius / obj_tz 
    
    uvs = torch.einsum('nij,nj->ni', K, t)
    uvs = uvs[:, :2] / uvs[:, 2:3]
    
    bboxes = torch.zeros(len(uvs), 4)
    bboxes[:, 0] = (uvs[:, 0] - bbox_scale / 2)
    bboxes[:, 1] = (uvs[:, 1] - bbox_scale / 2)
    bboxes[:, 2] = (uvs[:, 0] + bbox_scale / 2)
    bboxes[:, 3] = (uvs[:, 1] + bbox_scale / 2)

    img_hei, img_wid = image.shape[-2:]
    in_size = torch.tensor((img_hei, img_wid))
    out_size = torch.tensor((target_size, target_size))
    grids = bboxes_to_grid(bboxes, in_size, out_size)

    image_new = torch.nn.functional.grid_sample(image.type(torch.float32), 
                                                grids.type(torch.float32), 
                                                mode=mode, align_corners=True)
    if return_canonical_tz:
        ref_delta_tz = 2 * (1 + margin) * cam_focal * radius / target_size
        return image_new, ref_delta_tz
    return image_new



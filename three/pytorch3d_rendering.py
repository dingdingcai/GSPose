import torch

def batchfy_mv_meshes(meshes, num_views, batch_size=1):
    from pytorch3d import structures as py3d_struct
    mv_meshes = list()
    for mesh in meshes.split([1 for _ in range(batch_size)]):
        mv_meshes.append(mesh.extend(num_views))
    mv_meshes = py3d_struct.join_meshes_as_batch(mv_meshes)
    return mv_meshes

def generate_mesh_model_from_3Dbbox(bbox3D):
    from pytorch3d import structures as py3d_struct
    bbox3D = bbox3D.squeeze()
    assert(bbox3D.dim() == 2 and bbox3D.shape[1] == 3), bbox3D.shape
    if len(bbox3D) == 2: # 2x3
        xmin, ymin, zmin = bbox3D[0]
        xmax, ymax, zmax = bbox3D[1]
        bbox3D_corners = torch.tensor([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax]
        ])
    elif len(bbox3D) == 8: # 8x3
        bbox3D_corners = bbox3D
    else:
        print(bbox3D)
        raise NotImplementedError
    
    assert(bbox3D_corners.dim() == 2 and len(bbox3D_corners) == 8), bbox3D_corners.shape
    bbox3D_faces = torch.tensor([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],])
    # Convert quadrilateral faces to triangles
    bbox3D_faces = torch.cat([
        bbox3D_faces[:, [0, 1, 2]],
        bbox3D_faces[:, [2, 3, 0]]
    ], dim=0)

    # Create the mesh
    bbox3D_mesh = py3d_struct.Meshes(verts=[bbox3D_corners], faces=[bbox3D_faces])
    return bbox3D_mesh

def compute_ROI_camera_intrinsic(camK, img_size, camera_dist=None, scaled_T=None, 
                                  bbox_center=None, bbox_scale=None):
    """
    camK: Bx3x3
    scaled_T: Bx3
    bbox_center: Bx2
    bbox_scale: B
    camera_dist: B
    """
    squeeze = False
    if not isinstance(camK, torch.Tensor):
        camK = torch.tensor(camK)
    
    if scaled_T is not None and not isinstance(scaled_T, torch.Tensor):
        scaled_T = torch.tensor(scaled_T)
    if bbox_center is not None and not isinstance(bbox_center, torch.Tensor):
        bbox_center = torch.tensor(bbox_center)
    if bbox_scale is not None and not isinstance(bbox_scale, torch.Tensor):
        bbox_scale = torch.tensor(bbox_scale)
    
    device = camK.device
    assert(scaled_T is not None or bbox_center is not None)
    if scaled_T is not None:
        assert(camera_dist is not None)
        if scaled_T.dim() == 1:
            squeeze = True
            scaled_T = scaled_T[None, :]
        assert(scaled_T.dim() == 2), scaled_T.shape
        obj_2D_center = torch.einsum('bij,bj->bi', camK, scaled_T.to(device))
        bbox_center = obj_2D_center[:, :2] / obj_2D_center[:, 2:3]  #
        bbox_scale = camera_dist * img_size / scaled_T[:, 2].to(device)
        # print(bbox_center, bbox_scale)
    elif bbox_center is not None:
        assert(bbox_scale is not None)

    if bbox_center.dim() == 1:
        bbox_center = bbox_center[None, :]
        squeeze = True
    if bbox_scale.dim() == 0:
        bbox_scale = bbox_scale[None]
    if camK.dim() == 2:
        camK = camK[None, :, :]
        squeeze = True

    assert(bbox_center.dim() == 2 and bbox_scale.dim() == 1), (bbox_center.shape, bbox_scale.shape)

    bbox_x1y1 = bbox_center - bbox_scale[:, None] / 2
    bbox_rescaling_factor = img_size / bbox_scale
    T_cam2roi = torch.eye(3)[None, :, :].repeat(len(bbox_rescaling_factor), 1, 1).to(device)
    T_cam2roi[:, :2, 2] = -bbox_x1y1.to(device)
    T_cam2roi[:, :2, :] *= bbox_rescaling_factor[:, None, None].to(device)
    new_camK = torch.einsum('bij,bjk->bik', T_cam2roi, camK)
    if squeeze:
        new_camK = new_camK.squeeze(0)
    return new_camK

def generate_3D_coordinate_map_from_depth(depth, camK, obj_RT):
    if depth.squeeze().dim() == 2:
        depth = depth.squeeze()[None, None, :, :]
    elif depth.squeeze().dim() == 3:
        depth = depth.squeeze()[:, None, :, :]
    
    if obj_RT.dim() == 2:
        obj_RT = obj_RT[None, :, :]
    if camK.dim() == 2:
        camK = camK[None, :, :]
    
    if len(camK) != len(obj_RT):
        camK = camK.repeat(len(obj_RT), 1, 1)
        
    assert(depth.size(0) == obj_RT.size(0))
    assert(len(depth) == len(obj_RT)), (depth.shape, obj_RT.shape)
    
    device = depth.device
    im_hei, im_wid = depth.shape[-2:]
    YY, XX = torch.meshgrid(torch.arange(im_hei), torch.arange(im_wid), indexing='ij')
    XYZ_map = torch.stack([XX, YY, torch.ones_like(XX)], dim=0).to(device) # 3xHxW
    XYZ_map = XYZ_map[None, :, :, :] * depth # 1x3xHxW, Bx1xHxW
    
    XYZ_map = torch.einsum('bij,bjhw->bihw', torch.inverse(camK).to(device), XYZ_map)    
    Rs = obj_RT[:, :3, :3].to(device)
    Ts = obj_RT[:, :3, 3].to(device)
    
    XYZ_map = torch.einsum('bij,bjhw->bihw', torch.inverse(Rs), XYZ_map - Ts[:, :, None, None])
    
    return XYZ_map

def render_depth_from_mesh_model(mesh, obj_RT, camK, img_hei, img_wid, return_coordinate_map=False):
    """
    Pytorch3D: K_4x4 = [
                        [fx,   0,   px,   0],
                        [0,   fy,   py,   0],
                        [0,    0,    0,   1],
                        [0,    0,    1,   0],
                ]
    """
    from pytorch3d import renderer as py3d_renderer
    from pytorch3d import transforms as py3d_transform
    from pytorch3d.transforms import euler_angles_to_matrix
    
    device = obj_RT.device
    if obj_RT.dim() == 2:
        obj_RT = obj_RT[None, :, :]
        
    if camK.dim() == 2:
        camK = camK[None, :, :]
    
    if len(mesh) != len(obj_RT):
        mesh = mesh.extend(len(obj_RT))
    
    assert(len(mesh) == len(obj_RT)), (len(mesh), obj_RT.shape)
    
    Rz_mat = torch.eye(4).to(device)
    Rz_mat[:3, :3] = py3d_transform.euler_angles_to_matrix(torch.as_tensor([0, 0, torch.pi]), 'XYZ')
    py3d_RT = torch.einsum('ij,bjk->bik', Rz_mat, obj_RT)
    cam_R = py3d_RT[:, :3, :3].transpose(-2, -1)
    cam_T = py3d_RT[:, :3, 3]
    
    fxfy = torch.stack([camK[:, 0, 0], camK[:, 1, 1]], dim=1)
    pxpy = torch.stack([camK[:, 0, 2], camK[:, 1, 2]], dim=1)
    cameras = py3d_renderer.PerspectiveCameras(R=cam_R, 
                                               T=cam_T,
                                               image_size=((img_hei, img_wid),),
                                               focal_length=fxfy,
                                               principal_point=pxpy,
                                               in_ndc=False,
                                               device=device)
    # Define rasterizer settings
    raster_settings = py3d_renderer.RasterizationSettings(
        image_size=(img_hei, img_wid), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size=0,
    )

    rasterizer = py3d_renderer.MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings,
        )

    fragments = rasterizer(mesh.to(device))
    depth_map = fragments.zbuf[..., 0].unsqueeze(1) # Bx1xHxW
    depth_mask = torch.zeros_like(depth_map)
    depth_mask[depth_map>0] = 1
    depth_map *= depth_mask

    if return_coordinate_map:
        XYZ_map = generate_3D_coordinate_map_from_depth(depth_map, camK, obj_RT)
        return depth_map, depth_mask, XYZ_map * depth_mask
    
    return depth_map, depth_mask


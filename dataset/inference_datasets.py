import os
import sys
import mmcv
import json
import torch
import numpy as np
from PIL import Image
import mediapy as media
from pytorch3d import io as py3d_io
from pytorch3d import ops as py3d_ops
from pytorch3d import transforms as py3d_transform

from misc_utils import gdr_utils, gs_utils

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_ROOT)
DATASPACE_DIR = os.path.join(PROJ_ROOT, 'dataspace')

SUBSET_LINEMOID_OBJECTS = {
    "benchvise": "benchvise",  
    'cam': "cam", 
    "cat": "cat", 
    "driller": "driller", 
    "duck": "duck",
}

LINEMOID_OBJECTS = {
    'ape': "ape", 
    "benchvise": "benchvise",  
    'cam': "cam", 
    "can": "can",
    "cat": "cat", 
    "driller": "driller", 
    "duck": "duck", 
    "eggbox": "eggbox", 
    "glue": "glue", 
    "holepuncher": "holepuncher", 
    "iron": "iron", 
    "lamp": "lamp", 
    "phone": "phone",
    }

YCBV_OBJECTS = {
    '002_master_chef_can': 1,
    '003_cracker_box': 2,
    '004_sugar_box': 3,
    '005_tomato_soup_can': 4,
    '006_mustard_bottle': 5,
    '007_tuna_fish_can': 6,
    '008_pudding_box': 7,
    '009_gelatin_box': 8,
    '010_potted_meat_can': 9,
    '011_banana': 10,
    '019_pitcher_base': 11,
    '021_bleach_cleanser': 12,
    '024_bowl': 13,
    '025_mug': 14,
    '035_power_drill': 15,
    '036_wood_block': 16,
    '037_scissors': 17,
    '040_large_marker': 18,
    '051_large_clamp': 19,
    '052_extra_large_clamp': 20,
    '061_foam_brick': 21,
}


LOWTEXTURE_SCNNED_OBJEJCT_PATH = os.path.join(DATASPACE_DIR, 'onepose_dataset/scanned_model')


LOWTEXTUREVideo_OBJECTS = {
    'toyrobot': '0700-toyrobot-others',
    'yellowduck': '0701-yellowduck-others',
    # 'sheep': '0702-sheep-others',
    'fakebanana': '0703-fakebanana-others',
    'teabox': '0706-teabox-box',
    'orange': '0707-orange-others',
    'greenteapot': '0708-greenteapot-others',
    'lecreusetcup': '0710-lecreusetcup-others',
    'insta': '0712-insta-others',
    'batterycharger': '0713-batterycharger-others',
    'catmodel': '0714-catmodel-others',
    'logimouse': '0715-logimouse-others',
    'goldtea': '0718-goldtea-others',
    'yellowbluebox': '0719-yellowbluebox-box',
    'narcissustea': '0720-narcissustea-others',
    'camera': '0721-camera-others',
    'ugreenbox': '0722-ugreenbox-others',
    'headphonecontainer': '0723-headphonecontainer-others',
    # 'vitamin': '0724-vitamin-others',
    'airpods': '0725-airpods-others',
    'cup': '0726-cup-others',
    'shiningscan': '0727-shiningscan-box',
    'sensenut': '0728-sensenut-box',
    'flowertea': '0729-flowertea-others',
    'blackcolumcontainer': '0730-blackcolumcontainer-others',
    'whitesonycontainer': '0731-whitesonycontainer-others',
    'moliere': '0732-moliere-others',
    'mouse': '0733-mouse-others',
    # 'arglasscontainer': '0734-arglasscontainer-others',
    'facecream': '0735-facecream-others',
    'david': '0736-david-others',
    'pelikancontainer': '0737-pelikancontainer-box',
    'marseille': '0740-marseille-others',
    # 'toothbrushcontainer': '0741-toothbrushcontainer-others',
    'hikrobotbox': '0742-hikrobotbox-box',
    'blackcharger': '0743-blackcharger-others',
    'fan': '0744-fan-others',
    'ape': '0745-ape-others',
    'fakecam': '0746-fakecam-others',
    'penboxvert': '0748-penboxvert-others',
}

LOWTEXTURECrop_OBJECTS = {
    'toyrobot': "0700-toyrobot-others",    
    'teabox': "0706-teabox-box", 
    'catmodel': "0714-catmodel-others",    
    'camera': "0721-camera-others",
    'shiningscan': "0727-shiningscan-box", 
    'moliere': "0732-moliere-others", 
    'david': "0736-david-others",          
    'marseille': "0740-marseille-others",
}

LOWTEXTUREVideo_OBJECTS = {
    'toyrobot': "0700-toyrobot-others",    
    'teabox': "0706-teabox-box", 
    'catmodel': "0714-catmodel-others",    
    'camera': "0721-camera-others",
    'shiningscan': "0727-shiningscan-box", 
    'moliere': "0732-moliere-others", 
    'david': "0736-david-others",      
    'marseille': "0740-marseille-others",    
}

TRACKING_OBJECTS = {
    'waterbottle': "2024-waterbottle-others",     
}


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
class LINEMOD_Dataset_GEN6D(torch.utils.data.Dataset):
    def __init__(self, data_root, obj_name, 
                    subset_mode='train', 
                    obj_database_dir=None, 
                    load_gt_bbox=False, 
                    load_yolo_det=False, 
                    use_gt_mask=False,
                    use_binarized_mask=False,
                    num_refer_views=-1, # default -1 means all views
                    num_grid_points=4096):
        self.name2classID = {'ape': 1, "benchvise": 2,  
                            'cam': 4, "can": 5,
                            "cat": 6, "driller": 8, 
                            "duck": 9, "eggbox": 10, 
                            "glue": 11, "holepuncher": 12, 
                            "iron": 13, "lamp": 14, "phone": 15,
              }
        assert(subset_mode in ['train', 'test']), f'{subset_mode} is not a valid subset mode [train, test]'
        assert(obj_name in LINEMOID_OBJECTS.keys()), f'{obj_name} is not in the LINEMOD {LINEMOID_OBJECTS}'
        self.subset_mode = subset_mode
        self.data_root = data_root
        self.obj_name = obj_name
        self.num_refer_views = num_refer_views
        self.obj_classID = self.name2classID[self.obj_name]
        self.num_grid_points = num_grid_points
        self.obj_dir = os.path.join(self.data_root, self.obj_name)

        self.use_gt_mask = use_gt_mask
        self.load_gt_bbox = load_gt_bbox
        self.load_yolo_det = load_yolo_det
        self.obj_database_dir = obj_database_dir
        self.use_binarized_mask = use_binarized_mask

        self.to_meter_scale = 1e-3

        self.is_symmetric = False
        if self.obj_name in ['eggbox', 'glue']:
            self.is_symmetric = True

        self.camK = np.array([
            [572.4114 ,   0.     , 325.2611 ],
            [  0.     , 573.57043, 242.049  ],
            [  0.     ,   0.     ,   1.     ]], dtype=np.float32)
        self.diameter = gs_utils.read_numpy_data_from_txt(os.path.join(self.obj_dir, 'distance.txt')).squeeze() / 100.0 # in meter
        self.obj_ply_path = os.path.join(self.obj_dir, f'{self.obj_name}.ply') 
        self.obj_pointcloud = py3d_io.load_ply(self.obj_ply_path)[0].numpy()

        self.poses = list()
        self.image_IDs = list()
        self.allo_poses = list()
        self.image_paths = list()
        self.gt_bboxes = dict()
        self.yolo_bboxes = dict()
        image_subset_lists = gs_utils.read_list_data_from_txt(os.path.join(self.obj_dir, f'{self.subset_mode}.txt'))
        for idx, img_inst in enumerate(image_subset_lists):
            image_name = img_inst.strip().split('/')[-1]
            # check if the image is valid
            if not any(image_name.endswith(img_ext) for img_ext in IMG_EXTENSIONS):
                continue
            image_path = os.path.join(self.obj_dir, 'JPEGImages', image_name)
            image_ID = int(image_name.split('.')[0])
            pose_path = os.path.join(self.obj_dir, 'pose', f'pose{int(image_ID)}.npy')
            obj_pose = np.eye(4)
            obj_pose[:3, :4] = np.load(pose_path)[:3, :4]
            self.poses.append(obj_pose)
            self.image_paths.append(image_path)

            allo_pose = obj_pose.copy() 
            allo_pose[:3, :3] = gdr_utils.egocentric_to_allocentric(allo_pose)[:3, :3]
            self.allo_poses.append(allo_pose)

            self.image_IDs.append(image_ID)
            
            if self.load_gt_bbox:
                gt_x1, gt_y1, gt_x2, gt_y2 = mmcv.load(os.path.join(self.obj_dir, 'bboxes', f'bbox{int(image_ID)}.json'))
                self.gt_bboxes[image_ID] = np.array([gt_x1, gt_y1, gt_x2, gt_y2])

        if self.num_refer_views > 0:
            all_refer_matRs = torch.as_tensor(np.array(self.allo_poses), dtype=torch.float32)[:, :3, :3]
            all_refer_vecRs = py3d_transform.matrix_to_axis_angle(all_refer_matRs)
            if self.num_refer_views < all_refer_vecRs.shape[0]:
                self.refer_fps_inds = py3d_ops.sample_farthest_points(
                    all_refer_vecRs[None, :, :], K=self.num_refer_views, random_start_point=False)[1].squeeze(0) 
                self.fps_poses = list()
                self.fps_allo_poses = list()
                self.fps_image_paths = list()
                self.fps_image_IDs = list()
                self.fps_gt_bboxes = dict()
                self.fps_yolo_bboxes = dict()
                for idx in self.refer_fps_inds:
                    self.fps_poses.append(self.poses[idx])
                    self.fps_allo_poses.append(self.allo_poses[idx])
                    self.fps_image_paths.append(self.image_paths[idx])
                    self.fps_image_IDs.append(self.image_IDs[idx])
                    if self.load_gt_bbox:
                        self.fps_gt_bboxes[self.image_IDs[idx]] = self.gt_bboxes[self.image_IDs[idx]]
                    if self.load_yolo_det:
                        self.fps_yolo_bboxes[self.image_IDs[idx]] = self.yolo_bboxes[self.image_IDs[idx]]

                self.poses = self.fps_poses
                self.allo_poses = self.fps_allo_poses
                self.image_paths = self.fps_image_paths
                self.image_IDs = self.fps_image_IDs
                if self.load_gt_bbox:
                    self.gt_bboxes = self.fps_gt_bboxes
                if self.load_yolo_det:
                    self.yolo_bboxes = self.fps_yolo_bboxes            

        self.coseg_mask_dir = os.path.join(self.obj_dir, 'pred_coseg_mask')
        self.obj_bbox3d = gs_utils.read_numpy_data_from_txt(os.path.join(self.obj_dir, 'corners.txt')) # in meter
        min_3D_corner = self.obj_bbox3d.min(axis=0)
        max_3D_corner = self.obj_bbox3d.max(axis=0)
        obj_bbox3D_dims = max_3D_corner - min_3D_corner
        grid_cube_size = (np.prod(obj_bbox3D_dims, axis=0) / self.num_grid_points)**(1/3)
        xnum, ynum, znum = np.ceil(obj_bbox3D_dims / grid_cube_size).astype(np.int64)
        xmin, ymin, zmin = min_3D_corner
        xmax, ymax, zmax = max_3D_corner
        zgrid, ygrid, xgrid = np.meshgrid(np.linspace(zmin, zmax, znum),
                                            np.linspace(ymin, ymax, ynum), 
                                            np.linspace(xmin, xmax, xnum), 
                                            indexing='ij')
        self.bbox3d_grid_points = np.stack([xgrid, ygrid, zgrid], axis=-1).reshape(-1, 3)
        self.bbox3d_diameter = np.linalg.norm(obj_bbox3D_dims)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        camK = self.camK
        pose = self.poses[idx]
        image_ID = self.image_IDs[idx]
        allo_pose = self.allo_poses[idx]
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path), dtype=np.uint8) / 255.0

        data_dict = dict()
        data_dict['camK'] = torch.as_tensor(camK, dtype=torch.float32) 
        data_dict['pose'] = torch.as_tensor(pose, dtype=torch.float32)
        data_dict['image'] = torch.as_tensor(image, dtype=torch.float32)    
        data_dict['allo_pose'] = torch.as_tensor(allo_pose, dtype=torch.float32)

        data_dict['image_ID'] = image_ID
        data_dict['image_path'] = image_path

        if self.use_gt_mask:
            mask_path = os.path.join(self.obj_dir, 'mask', f'{image_ID:04d}.png')
            data_dict['gt_mask_path'] = mask_path

        if self.obj_database_dir is not None:
            data_dict['coseg_mask_path'] = os.path.join(self.obj_database_dir, 'pred_coseg_mask', '{:06d}.png'.format(image_ID))

        if self.load_yolo_det and self.yolo_bboxes.get(image_ID, None) is not None:
            img_hei, img_wid = image.shape[:2]
            x0_n, y0_n, x1_n, y1_n = self.yolo_bboxes[image_ID]
            x0_n, x1_n = x0_n * img_wid, x1_n * img_wid
            y0_n, y1_n = y0_n * img_hei, y1_n * img_hei
            
            bbox_xyxy = np.array([x0_n, y0_n, x1_n, y1_n])
            data_dict['yolo_bbox'] = torch.as_tensor(bbox_xyxy, dtype=torch.float32)

            bbox_scale = max(x1_n - x0_n, y1_n - y0_n)
            bbox_center = np.array([(x0_n + x1_n) / 2, (y0_n + y1_n) / 2])
            data_dict['bbox_scale'] = torch.as_tensor(bbox_scale, dtype=torch.float32)
            data_dict['bbox_center'] = torch.as_tensor(bbox_center, dtype=torch.float32)
            
        if self.load_gt_bbox:
            x1, y1, x2, y2 = self.gt_bboxes[image_ID]
            bbox_scale = max(x2 - x1, y2 - y1)
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            data_dict['gt_bbox_scale'] = torch.as_tensor(bbox_scale, dtype=torch.float32)
            data_dict['gt_bbox_center'] = torch.as_tensor(bbox_center, dtype=torch.float32)
        
        return data_dict

    def collate_fn(self, batch):
        """
        batchify the data
        """
        new_batch = dict()
        for each_dat in batch:
            for key, val in each_dat.items():
                if key not in new_batch:
                    new_batch[key] = list()
                new_batch[key].append(val)

        for key, val in new_batch.items():
            new_batch[key] = torch.stack(val, dim=0)

        return new_batch


class LINEMOD_Dataset_BOP(torch.utils.data.Dataset):
    def __init__(self, data_root, obj_name, 
                 subset_mode='train', 
                 obj_database_dir=None, 
                 load_gt_bbox=False, 
                 use_gt_mask=False,
                 use_binarized_mask=False,
                 num_refer_views=-1, # default -1 means all views
                 load_yolo_det=False, 
                 num_grid_points=4096):

        self.name2classID = {'ape': 1, "benchvise": 2,  
                            'cam': 4, "can": 5,
                            "cat": 6, "driller": 8, 
                            "duck": 9, "eggbox": 10, 
                            "glue": 11, "holepuncher": 12, 
                            "iron": 13, "lamp": 14, "phone": 15,
              }

        assert(subset_mode in ['train', 'test']), f'{subset_mode} is not a valid subset mode [train, test]'
        assert(obj_name in LINEMOID_OBJECTS.keys()), f'{obj_name} is not in the LINEMOD {LINEMOID_OBJECTS}'
        self.obj_name = obj_name
        self.data_root = data_root
        self.subset_mode = subset_mode
        self.num_grid_points = num_grid_points
        self.obj_classID = self.name2classID[self.obj_name]
        self.obj_dir = os.path.join(self.data_root, 'test', '{:06}'.format(self.obj_classID))

        self.use_gt_mask = use_gt_mask
        self.load_gt_bbox = load_gt_bbox
        self.load_yolo_det = load_yolo_det
        self.obj_database_dir = obj_database_dir
        self.use_binarized_mask = use_binarized_mask
        self.to_meter_scale = 1e-3
        
        self.camK = np.array([
            [572.4114 ,   0.     , 325.2611 ],
            [  0.     , 573.57043, 242.049  ],
            [  0.     ,   0.     ,   1.     ]], dtype=np.float32)
        
        model_dir = os.path.join(self.data_root, 'models')
        if not os.path.exists(model_dir):
            model_dir = os.path.join(self.data_root, 'models_eval')
        model_file = os.path.join(model_dir, f'models_info.json')
        with open(model_file, 'r') as f:
            self.model_info = json.load(f)

        self.obj_model_info = self.model_info[str(self.obj_classID)]
        self.diameter = self.obj_model_info['diameter'] * self.to_meter_scale # convert to m

        self.is_symmetric = False
        for _key, _val in self.obj_model_info.items():
            if 'symmetries' in _key:
                self.is_symmetric = True

        bbox3d_xyz = np.array([self.obj_model_info["size_x"],
                                self.obj_model_info["size_y"],
                                self.obj_model_info["size_z"],
                            ]) * self.to_meter_scale # convert to m
        self.obj_bbox3d = np.array([
            [-bbox3d_xyz[0], -bbox3d_xyz[0], -bbox3d_xyz[0], -bbox3d_xyz[0],  bbox3d_xyz[0],  bbox3d_xyz[0],  bbox3d_xyz[0],  bbox3d_xyz[0]],
            [-bbox3d_xyz[1], -bbox3d_xyz[1],  bbox3d_xyz[1],  bbox3d_xyz[1], -bbox3d_xyz[1], -bbox3d_xyz[1],  bbox3d_xyz[1],  bbox3d_xyz[1]],
            [-bbox3d_xyz[2],  bbox3d_xyz[2],  bbox3d_xyz[2], -bbox3d_xyz[2], -bbox3d_xyz[2],  bbox3d_xyz[2],  bbox3d_xyz[2], -bbox3d_xyz[2]]
            ]).T / 2 
        self.bbox3d_diameter = np.linalg.norm(bbox3d_xyz)
        
        self.obj_ply_path = os.path.join(model_dir, 'obj_{:06}.ply'.format(self.obj_classID))
        self.obj_pointcloud = py3d_io.load_ply(self.obj_ply_path)[0].numpy() * self.to_meter_scale # convert to m

        yolo_detection_dir = os.path.join(DATASPACE_DIR, 'bop_dataset/lm_yolo_detection/val')
        obj_yolo_detect_name = '08{:02d}-lm{}-others'.format(self.obj_classID, self.obj_classID)
        obj_yolo_label_dir = os.path.join(yolo_detection_dir, obj_yolo_detect_name, 'labels')
        
        self.poses_file = os.path.join(self.obj_dir, 'scene_gt.json')
        with open(self.poses_file, 'r') as f:
            self.poses_info = json.load(f)

        if self.load_gt_bbox:
            self.bboxes_file = os.path.join(self.obj_dir, 'scene_gt_info.json')
            with open(self.bboxes_file, 'r') as f:
                self.bboxes_info = json.load(f)
            self.gt_bboxes = dict()

        self.poses = list()
        self.image_IDs = list()
        self.allo_poses = list()
        self.image_paths = list()
        self.yolo_bboxes = dict()
        image_subset_lists = gs_utils.read_list_data_from_txt(os.path.join(self.obj_dir, f'{self.subset_mode}.txt'))
        for idx, img_inst in enumerate(image_subset_lists):
            image_ID = int(img_inst)
            image_path = os.path.join(self.obj_dir, 'rgb', '{:06d}.png'.format(image_ID))
            pose_RT = self.poses_info[str(image_ID)][0]
            obj_pose = np.eye(4)
            obj_pose[:3, :3] = np.array(pose_RT['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
            obj_pose[:3, 3] = np.array(pose_RT['cam_t_m2c'], dtype=np.float32).reshape(3) * self.to_meter_scale # convert to m

            self.poses.append(obj_pose)
            self.image_paths.append(image_path)

            allo_pose = obj_pose.copy() 
            allo_pose[:3, :3] = gdr_utils.egocentric_to_allocentric(allo_pose)[:3, :3]
            self.allo_poses.append(allo_pose)

            self.image_IDs.append(image_ID)

            if self.load_gt_bbox:
                gt_x1, gt_y1, gt_bw, gt_bh = self.bboxes_info[str(image_ID)][0]['bbox_visib']
                gt_x2 = gt_x1 + gt_bw
                gt_y2 = gt_y1 + gt_bh
                self.gt_bboxes[image_ID] = np.array([gt_x1, gt_y1, gt_x2, gt_y2])

            if self.load_yolo_det:
                yolo_bbox_path = os.path.join(obj_yolo_label_dir, '{:06d}.txt'.format(image_ID + 1)) # yolo_results starts from 1
                if os.path.exists(yolo_bbox_path):
                    yolo_box = np.loadtxt(yolo_bbox_path)
                    assert yolo_box.shape[0] != 0, f"img id:{image_ID} no box detected!"
                    if len(yolo_box.shape) == 2:
                        want_id = np.argsort(yolo_box[:,5])[0]
                        yolo_box = yolo_box[want_id]
                    x_c_n, y_c_n, w_n, h_n = yolo_box[1:5]
                    x0_n, y0_n = x_c_n - w_n / 2, y_c_n - h_n / 2
                    x1_n, y1_n = x_c_n + w_n / 2, y_c_n + h_n / 2
                    self.yolo_bboxes[image_ID] = np.array([x0_n, y0_n, x1_n, y1_n])
                # else:
                #     self.yolo_bboxes[image_ID] = np.array([0, 0, 1.0, 1.0])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        data_dict = dict()
        camK = self.camK
        pose = self.poses[idx]
        image_ID = self.image_IDs[idx]
        allo_pose = self.allo_poses[idx]
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path), dtype=np.uint8) / 255.0

        data_dict['camK'] = torch.as_tensor(camK, dtype=torch.float32) 
        data_dict['pose'] = torch.as_tensor(pose, dtype=torch.float32)
        data_dict['image'] = torch.as_tensor(image, dtype=torch.float32)    
        data_dict['allo_pose'] = torch.as_tensor(allo_pose, dtype=torch.float32)

        data_dict['image_ID'] = image_ID
        data_dict['image_path'] = image_path

        if self.use_gt_mask:
            mask_path = os.path.join(self.obj_dir, 'mask_visib', f'{image_ID:06d}_000000.png')
            data_dict['gt_mask_path'] = mask_path
        
        if self.obj_database_dir is not None:
            data_dict['coseg_mask_path'] = os.path.join(self.obj_database_dir, 'pred_coseg_mask', '{:06d}.png'.format(image_ID))

        if self.load_yolo_det and self.yolo_bboxes.get(image_ID, None) is not None:
            img_hei, img_wid = image.shape[:2]
            x0_n, y0_n, x1_n, y1_n = self.yolo_bboxes[image_ID]
            x0_n, x1_n = x0_n * img_wid, x1_n * img_wid
            y0_n, y1_n = y0_n * img_hei, y1_n * img_hei
            
            bbox_xyxy = np.array([x0_n, y0_n, x1_n, y1_n])
            data_dict['yolo_bbox'] = torch.as_tensor(bbox_xyxy, dtype=torch.float32)

            bbox_scale = max(x1_n - x0_n, y1_n - y0_n)
            bbox_center = np.array([(x0_n + x1_n) / 2, (y0_n + y1_n) / 2])
            data_dict['bbox_scale'] = torch.as_tensor(bbox_scale, dtype=torch.float32)
            data_dict['bbox_center'] = torch.as_tensor(bbox_center, dtype=torch.float32)
            
        if self.load_gt_bbox:
            x1, y1, x2, y2 = self.gt_bboxes[image_ID]
            bbox_scale = max(x2 - x1, y2 - y1)
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            data_dict['gt_bbox_scale'] = torch.as_tensor(bbox_scale, dtype=torch.float32)
            data_dict['gt_bbox_center'] = torch.as_tensor(bbox_center, dtype=torch.float32)


        return data_dict

    def collate_fn(self, batch):
        """
        batchify the data
        """
        new_batch = dict()
        for each_dat in batch:
            for key, val in each_dat.items():
                if key not in new_batch:
                    new_batch[key] = list()
                new_batch[key].append(val)

        for key, val in new_batch.items():
            new_batch[key] = torch.stack(val, dim=0)

        return new_batch


class LOWTEXTUREVideo_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, obj_name, subset_mode='train', num_grid_points=4096, obj_database_dir=None, 
                 num_refer_views=-1, 
                 use_gt_mask=False,
                 use_binarized_mask=False,
                 load_yolo_det=False):
        assert(subset_mode in ['train', 'test']), f'{subset_mode} is not a valid subset mode [train, test]'
        assert(obj_name in LOWTEXTUREVideo_OBJECTS.keys()), f'{obj_name} is not in the LowTexture {LOWTEXTUREVideo_OBJECTS.keys()}'
        self.subset_mode = subset_mode
        self.data_root = data_root
        self.obj_name = obj_name
        self.num_grid_points = num_grid_points
        self.obj_dir_name = LOWTEXTUREVideo_OBJECTS[self.obj_name]
        self.obj_dir = os.path.join(self.data_root, self.obj_dir_name)

        self.is_symmetric = False
        self.obj_database_dir = obj_database_dir
        self.use_binarized_mask = use_binarized_mask

        self.obj_bbox3d = gs_utils.read_numpy_data_from_txt(os.path.join(self.obj_dir, 'box3d_corners.txt')) # in meter
        min_3D_corner = self.obj_bbox3d.min(axis=0)
        max_3D_corner = self.obj_bbox3d.max(axis=0)
        obj_bbox3D_dims = max_3D_corner - min_3D_corner
        grid_cube_size = (np.prod(obj_bbox3D_dims, axis=0) / self.num_grid_points)**(1/3)
        xnum, ynum, znum = np.ceil(obj_bbox3D_dims / grid_cube_size).astype(np.int64)
        xmin, ymin, zmin = min_3D_corner
        xmax, ymax, zmax = max_3D_corner
        zgrid, ygrid, xgrid = np.meshgrid(np.linspace(zmin, zmax, znum),
                                            np.linspace(ymin, ymax, ynum), 
                                            np.linspace(xmin, xmax, xnum), 
                                            indexing='ij')
        self.bbox3d_grid_points = np.stack([xgrid, ygrid, zgrid], axis=-1).reshape(-1, 3)
        self.bbox3d_diameter = np.linalg.norm(obj_bbox3D_dims)

        if self.subset_mode == 'train':
            self.obj_seq_path = os.path.join(self.obj_dir, f'{self.obj_name}-1')
        else:
            self.obj_seq_path = os.path.join(self.obj_dir, f'{self.obj_name}-2')
 
        intrinsic_path = os.path.join(self.obj_seq_path, 'intrinsics.txt')
        try:
            self.camK = np.eye(3)
            intrin_params = dict()
            with open(intrinsic_path) as f:
                txt_dat = f.readlines()
                for _line in txt_dat:
                    _key, _val = _line.strip().split(':')
                    intrin_params[_key] = float(_val)
            self.camK [0, 0] = intrin_params['fx']
            self.camK [1, 1] = intrin_params['fy']
            self.camK [0, 2] = intrin_params['cx']
            self.camK [1, 2] = intrin_params['cy']
        except:
            self.camK = gs_utils.read_numpy_data_from_txt(intrinsic_path)
        
        self.pose_dir = os.path.join(self.obj_seq_path, 'poses_ba')
        self.video_path = os.path.join(self.obj_seq_path, 'Frames.m4v')
        self.video_frames = media.read_video(self.video_path) # NxHxWx3    
        IMG_NUM, IMG_HEI, IMG_WID = self.video_frames.shape[:3]
        self.poses = list()
        self.allo_poses = list()
        self.image_IDs = list()
        for frame_idx in range(IMG_NUM):
            pose_path = os.path.join(self.pose_dir, f'{frame_idx}.txt')
            if not os.path.exists(pose_path):
                continue
            obj_pose = gs_utils.read_numpy_data_from_txt(pose_path)
            self.poses.append(obj_pose)
            allo_pose = obj_pose.copy() 
            allo_pose[:3, :3] = gdr_utils.egocentric_to_allocentric(allo_pose)[:3, :3]
            self.allo_poses.append(allo_pose)

            self.image_IDs.append(frame_idx)
        
        self.coseg_mask_dir = os.path.join(self.obj_seq_path, 'video_coseg_mask')
        self.obj_cad_model_path = os.path.join(LOWTEXTURE_SCNNED_OBJEJCT_PATH, self.obj_dir_name, 'model.obj')
        if os.path.exists(self.obj_cad_model_path):
            self.obj_pointcloud = py3d_io.load_obj(self.obj_cad_model_path)[0].numpy()
            try:
                self.diameter = float(np.loadtxt(os.path.join(LOWTEXTURE_SCNNED_OBJEJCT_PATH, self.obj_dir_name, 'diameter.txt')))
            except:
                self.diameter = gs_utils.torch_compute_diameter_from_pointcloud(self.obj_pointcloud)
        else:
            self.obj_pointcloud = None
            self.diameter = self.bbox3d_diameter

        
    
    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        data_dict = dict()
        camK = self.camK
        pose = self.poses[idx]
        image_ID = self.image_IDs[idx]
        allo_pose = self.allo_poses[idx]
        image = np.array(self.video_frames[idx]) / 255.0
        
        data_dict['image_ID'] = image_ID
        data_dict['camK'] = torch.as_tensor(camK, dtype=torch.float32) 
        data_dict['pose'] = torch.as_tensor(pose, dtype=torch.float32)
        data_dict['image'] = torch.as_tensor(image, dtype=torch.float32)    
        data_dict['allo_pose'] = torch.as_tensor(allo_pose, dtype=torch.float32)

        if self.obj_database_dir is not None:
            data_dict['coseg_mask_path'] = os.path.join(self.obj_database_dir, 'pred_coseg_mask', '{:06d}.png'.format(image_ID))

        return data_dict

    def collate_fn(self, batch):
        """
        batchify the data
        """
        new_batch = dict()
        for each_dat in batch:
            for key, val in each_dat.items():
                if key not in new_batch:
                    new_batch[key] = list()
                new_batch[key].append(val)

        for key, val in new_batch.items():
            new_batch[key] = torch.stack(val, dim=0)

        return new_batch


class VideoTrack_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, obj_name, subset_mode='train', num_grid_points=4096, obj_database_dir=None, 
                 num_refer_views=-1, 
                 use_gt_mask=False,
                 load_yolo_det=False):
        assert(subset_mode in ['train', 'test']), f'{subset_mode} is not a valid subset mode [train, test]'
        assert(obj_name in TRACKING_OBJECTS.keys()), f'{obj_name} is not in the LowTexture {TRACKING_OBJECTS.keys()}'
        self.subset_mode = subset_mode
        self.data_root = data_root
        self.obj_name = obj_name
        self.num_grid_points = num_grid_points
        self.obj_dir_name = TRACKING_OBJECTS[self.obj_name]
        self.obj_dir = os.path.join(self.data_root, self.obj_dir_name)

        self.is_symmetric = False
        self.obj_database_dir = obj_database_dir

        self.obj_bbox3d = gs_utils.read_numpy_data_from_txt(os.path.join(self.obj_dir, 'box3d_corners.txt')) # in meter
        min_3D_corner = self.obj_bbox3d.min(axis=0)
        max_3D_corner = self.obj_bbox3d.max(axis=0)
        obj_bbox3D_dims = max_3D_corner - min_3D_corner
        grid_cube_size = (np.prod(obj_bbox3D_dims, axis=0) / self.num_grid_points)**(1/3)
        xnum, ynum, znum = np.ceil(obj_bbox3D_dims / grid_cube_size).astype(np.int64)
        xmin, ymin, zmin = min_3D_corner
        xmax, ymax, zmax = max_3D_corner
        zgrid, ygrid, xgrid = np.meshgrid(np.linspace(zmin, zmax, znum),
                                            np.linspace(ymin, ymax, ynum), 
                                            np.linspace(xmin, xmax, xnum), 
                                            indexing='ij')
        self.bbox3d_grid_points = np.stack([xgrid, ygrid, zgrid], axis=-1).reshape(-1, 3)
        self.bbox3d_diameter = np.linalg.norm(obj_bbox3D_dims)

        if self.subset_mode == 'train':
            self.obj_seq_path = os.path.join(self.obj_dir, f'{self.obj_name}-refer')
        else:
            self.obj_seq_path = os.path.join(self.obj_dir, f'{self.obj_name}-query')

        intrinsic_path = os.path.join(self.obj_seq_path, 'intrinsics.txt')
        try:
            self.camK = np.eye(3)
            intrin_params = dict()
            with open(intrinsic_path) as f:
                txt_dat = f.readlines()
                for _line in txt_dat:
                    _key, _val = _line.strip().split(':')
                    intrin_params[_key] = float(_val)
            self.camK [0, 0] = intrin_params['fx']
            self.camK [1, 1] = intrin_params['fy']
            self.camK [0, 2] = intrin_params['cx']
            self.camK [1, 2] = intrin_params['cy']
        except:
            self.camK = gs_utils.read_numpy_data_from_txt(intrinsic_path)
        
        self.pose_dir = os.path.join(self.obj_seq_path, 'poses_ba')
        self.video_path = os.path.join(self.obj_seq_path, 'Frames.m4v')
        self.video_frames = media.read_video(self.video_path) # NxHxWx3    
        IMG_NUM, IMG_HEI, IMG_WID = self.video_frames.shape[:3]
        self.poses = list()
        self.allo_poses = list()
        self.image_IDs = list()
        for frame_idx in range(IMG_NUM):
            pose_path = os.path.join(self.pose_dir, f'{frame_idx}.txt')
            if not os.path.exists(pose_path):
                continue
            obj_pose = gs_utils.read_numpy_data_from_txt(pose_path)
            self.poses.append(obj_pose)
            allo_pose = obj_pose.copy() 
            allo_pose[:3, :3] = gdr_utils.egocentric_to_allocentric(allo_pose)[:3, :3]
            self.allo_poses.append(allo_pose)

            self.image_IDs.append(frame_idx)
        
        self.coseg_mask_dir = os.path.join(self.obj_seq_path, 'video_coseg_mask')
        self.obj_cad_model_path = os.path.join(LOWTEXTURE_SCNNED_OBJEJCT_PATH, self.obj_dir_name, 'model.obj')
        if os.path.exists(self.obj_cad_model_path):
            self.obj_pointcloud = py3d_io.load_obj(self.obj_cad_model_path)[0].numpy()
            try:
                self.diameter = float(np.loadtxt(os.path.join(LOWTEXTURE_SCNNED_OBJEJCT_PATH, self.obj_dir_name, 'diameter.txt')))
            except:
                self.diameter = gs_utils.torch_compute_diameter_from_pointcloud(self.obj_pointcloud)
        else:
            self.obj_pointcloud = None
            self.diameter = self.bbox3d_diameter

        
    
    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        data_dict = dict()
        camK = self.camK
        pose = self.poses[idx]
        image_ID = self.image_IDs[idx]
        allo_pose = self.allo_poses[idx]
        image = np.array(self.video_frames[idx]) / 255.0
        
        data_dict['image_ID'] = image_ID
        data_dict['camK'] = torch.as_tensor(camK, dtype=torch.float32) 
        data_dict['pose'] = torch.as_tensor(pose, dtype=torch.float32)
        data_dict['image'] = torch.as_tensor(image, dtype=torch.float32)    
        data_dict['allo_pose'] = torch.as_tensor(allo_pose, dtype=torch.float32)

        if self.obj_database_dir is not None:
            data_dict['coseg_mask_path'] = os.path.join(self.obj_database_dir, 'pred_coseg_mask', '{:06d}.png'.format(image_ID))

        return data_dict

    def collate_fn(self, batch):
        """
        batchify the data
        """
        new_batch = dict()
        for each_dat in batch:
            for key, val in each_dat.items():
                if key not in new_batch:
                    new_batch[key] = list()
                new_batch[key].append(val)

        for key, val in new_batch.items():
            new_batch[key] = torch.stack(val, dim=0)

        return new_batch

datasetCallbacks = {
    'LINEMOD_GEN6D': {'OBJECTS': LINEMOID_OBJECTS, 
                'DATASETLOADER': LINEMOD_Dataset_GEN6D, 
                'DATAROOT': os.path.join(DATASPACE_DIR, 'LINEMOD_Gen6D'),
                },
    
    'LINEMOD_SUBSET': {'OBJECTS': SUBSET_LINEMOID_OBJECTS, 
                'DATASETLOADER': LINEMOD_Dataset_GEN6D, 
                'DATAROOT': os.path.join(DATASPACE_DIR, 'LINEMOD_Gen6D'),
                },

    'LINEMOD': {'OBJECTS': LINEMOID_OBJECTS, 
                'DATASETLOADER': LINEMOD_Dataset_BOP, 
                'DATAROOT': os.path.join(DATASPACE_DIR, 'bop_dataset/lm'),
                },
    
    "LOWTEXTUREVideo": {'OBJECTS': LOWTEXTUREVideo_OBJECTS, 
                        'DATASETLOADER': LOWTEXTUREVideo_Dataset, 
                        'DATAROOT': os.path.join(DATASPACE_DIR, 'onepose_dataset/lowtexture_test_data'),
                        },

    "VideoTrack": {'OBJECTS': TRACKING_OBJECTS, 
                        'DATASETLOADER': VideoTrack_Dataset, 
                        'DATAROOT': os.path.join(DATASPACE_DIR, 'onepose_dataset/lowtexture_test_data'),
                        },                    
}




"""
tensorboard --logdir 0700-toyrobot-others-binamask/ --port 9009
tensorboard --logdir 0701-yellowduck-others-binamask/ --port 9009
# tensorboard --logdir 0702-sheep-others-binamask/ --port 9009
tensorboard --logdir 0703-fakebanana-others-binamask/ --port 9009
tensorboard --logdir 0706-teabox-box-binamask/ --port 9009
tensorboard --logdir 0707-orange-others-binamask/ --port 9009
tensorboard --logdir 0708-greenteapot-others-binamask/ --port 9009
tensorboard --logdir 0710-lecreusetcup-others-binamask/ --port 9009
tensorboard --logdir 0712-insta-others-binamask/ --port 9009
tensorboard --logdir 0713-batterycharger-others-binamask/ --port 9009
tensorboard --logdir 0714-catmodel-others-binamask/ --port 9009
tensorboard --logdir 0715-logimouse-others-binamask/ --port 9009
tensorboard --logdir 0718-goldtea-others-binamask/ --port 9009
tensorboard --logdir 0719-yellowbluebox-box-binamask/ --port 9009
tensorboard --logdir 0720-narcissustea-others-binamask/ --port 9009
tensorboard --logdir 0721-camera-others-binamask/ --port 9009
tensorboard --logdir 0722-ugreenbox-others-binamask/ --port 9009
tensorboard --logdir 0723-headphonecontainer-others-binamask/ --port 9009
# tensorboard --logdir 0724-vitamin-others-binamask/ --port 9009
tensorboard --logdir 0725-airpods-others-binamask/ --port 9009
tensorboard --logdir 0726-cup-others-binamask/ --port 9009
tensorboard --logdir 0727-shiningscan-box-binamask/ --port 9009
tensorboard --logdir 0728-sensenut-box-binamask/ --port 9009
tensorboard --logdir 0729-flowertea-others-binamask/ --port 9009
tensorboard --logdir 0730-blackcolumcontainer-others-binamask/ --port 9009
tensorboard --logdir 0731-whitesonycontainer-others-binamask/ --port 9009
tensorboard --logdir 0732-moliere-others-binamask/ --port 9009
tensorboard --logdir 0733-mouse-others-binamask/ --port 9009
tensorboard --logdir 0735-facecream-others-binamask/ --port 9009
tensorboard --logdir 0736-david-others-binamask/ --port 9009
tensorboard --logdir 0737-pelikancontainer-box-binamask/ --port 9009
tensorboard --logdir 0740-marseille-others-binamask/ --port 9009
tensorboard --logdir 0742-hikrobotbox-box-binamask/ --port 9009
tensorboard --logdir 0743-blackcharger-others-binamask/ --port 9009
tensorboard --logdir 0744-fan-others-binamask/ --port 9009
tensorboard --logdir 0745-ape-others-binamask/ --port 9009
tensorboard --logdir 0746-fakecam-others-binamask/ --port 9009
tensorboard --logdir 0748-penboxvert-others-binamask/ --port 9009


"""

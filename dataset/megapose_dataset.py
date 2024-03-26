import os
import sys
import cv2
import PIL
import copy
import mmcv
import torch
import logging
import hashlib
import numpy as np
from pytorch3d import ops as py3d_ops
from pytorch3d import transforms as py3d_transform

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJ_ROOT)

from dataset import misc
from three import meshutils
from misc_utils import gs_utils

logger = logging.getLogger(__name__)
CUR_FILE_DIR = os.path.dirname(__file__)

class MegaPose_Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir,
                 img_scale=224,
                 query_view_num=1,
                 refer_view_num=8,
                 rand_view_num=16,
                 nnb_Rmat_threshold=30,
                 DZI_camera_dist_ratio=0.25,
                 DZI_pad_ratio=0.5,
                 DZI_scale_ratio=0.25,
                 DZI_shift_ratio=0.25,
                 color_aug_prob=0.5,
                 query_longside_scale=672, 
                 visib_fraction_ratio=0.90,
                 min_amount_of_instances=2000,
                 color_type='rgb',
                 use_cache=True,
                 mask_morph=True,
                 filter_invalid=True,
                 mask_morph_kernel_size=3,
                 ):
        self.data_dir = data_dir
        self.img_scale = img_scale
        self.rand_view_num = rand_view_num
        self.refer_view_num = refer_view_num
        self.query_view_num = query_view_num
        self.nnb_Rmat_threshold = nnb_Rmat_threshold
        self.query_longside_scale = query_longside_scale

        assert(query_longside_scale in [224, 448, 672, 896, 1120]), "query_longside_scale must be one of [224, 448, 672, 896, 1120]"
        
        self.visib_fraction_ratio = visib_fraction_ratio
        self.min_amount_of_instances = min_amount_of_instances
        
        self.COLOR_TYPE = color_type
        self.COLOR_AUG_PROB = color_aug_prob
        self.DZI_PAD_RATIO = DZI_pad_ratio
        self.DZI_SCALE_RATIO = DZI_scale_ratio  # wh scale
        self.DZI_SHIFT_RATIO = DZI_shift_ratio  # center shift
        self.DZI_camera_dist_ratio = DZI_camera_dist_ratio

        self.mask_morph = mask_morph
        self.mask_morph_kernel_size = mask_morph_kernel_size
        self.filter_invalid = filter_invalid
        
        self.use_cache = use_cache
        self.dataset_name = 'train_pbr'
        cache_name = 'megapose1050_gso1000'
        self.cache_dir = os.path.join(CUR_FILE_DIR, ".cache")  # .cache

        self.obj_diameters = dict()
        self.obj_pointclouds = dict()
        self.obj_3D_bounding_boxes = dict()
        
        self.OBJ_IDs_to_NAMEs_DICT = dict()
        self.OBJ_NAMEs_to_IDs_DICT = dict()
        cad_model_dir = os.path.join(self.data_dir, 'models')
        valid_mesh_names = mmcv.load(os.path.join(self.data_dir, 'valid_meshes.json'))
        valid_mesh_info_path = os.path.join(self.data_dir, 'valid_meshes_diameters.json')
        valid_mesh_bbox3D_path = os.path.join(self.data_dir, 'valid_meshes_3D_bboxes.json')

        for obj_idx, obj_class_name in enumerate(valid_mesh_names):
            self.OBJ_IDs_to_NAMEs_DICT[obj_idx] = obj_class_name
            self.OBJ_NAMEs_to_IDs_DICT[obj_class_name] = obj_idx
            if not os.path.exists(valid_mesh_info_path):
                cad_model_path = os.path.join(cad_model_dir, obj_class_name, 'meshes/model.ply')
                obj = meshutils.Object3D(cad_model_path)
                self.obj_diameters[obj_idx] = obj.bounding_diameter

            if not os.path.exists(valid_mesh_bbox3D_path):
                cad_model_path = os.path.join(cad_model_dir, obj_class_name, 'meshes/model.ply')
                obj = meshutils.Object3D(cad_model_path)
                obj_pointclouds = obj.vertices
                self.obj_3D_bounding_boxes[obj_idx] = np.stack([obj_pointclouds.min(0),
                                                                obj_pointclouds.max(0)], axis=0)
        if self.use_cache:
            if not os.path.exists(valid_mesh_info_path):
                mmcv.dump(self.obj_diameters, valid_mesh_info_path)
                logger.info("Dumped obj_diameters to {}".format(valid_mesh_info_path))
            else:
                obj_diameters = mmcv.load(valid_mesh_info_path)
                logger.info("load obj_diameters from {}".format(valid_mesh_info_path))
                self.obj_diameters = {int(k): float(v) for k, v in obj_diameters.items()}
            
            if not os.path.exists(valid_mesh_bbox3D_path):
                mmcv.dump(self.obj_3D_bounding_boxes, valid_mesh_bbox3D_path)
                logger.info("Dumped obj_bbox3D to {}".format(valid_mesh_bbox3D_path))
            else:
                obj_3D_bounding_boxes = mmcv.load(valid_mesh_bbox3D_path)
                logger.info("load obj_bbox3D from {}".format(valid_mesh_bbox3D_path))
                self.obj_3D_bounding_boxes = {int(k): np.asarray(v) for k, v in obj_3D_bounding_boxes.items()}



        refer_hashed_file_name = hashlib.md5(
            "dataset_dicts_{}_{}".format(cache_name, self.dataset_name).encode("utf-8")
        ).hexdigest()
        refer_cache_path = os.path.join(self.cache_dir, 
            "dataset_dicts_{}_{}_{}.pkl".format(cache_name, self.dataset_name, refer_hashed_file_name))
        
        flatten_hashed_dict_name = "flatten_dataset_dicts_{}_alloRT_vis{}_min{}_{}".format(
            self.dataset_name, 
            int(100*self.visib_fraction_ratio), 
            self.min_amount_of_instances,
            f'flatten_{cache_name}_dict')
        flatten_hashed_dict_code = hashlib.md5(flatten_hashed_dict_name.encode("utf-8")).hexdigest()

        flatten_hashed_list_name = "flatten_dataset_dicts_{}_alloRT_vis{}_min{}_{}".format(
            self.dataset_name, 
            int(100*self.visib_fraction_ratio), 
            self.min_amount_of_instances,
            f'flatten_{cache_name}_list')
        flatten_hashed_list_code = hashlib.md5(flatten_hashed_list_name.encode("utf-8")).hexdigest()

        cached_flatten_dict_path = os.path.join(self.cache_dir, 
                                                f'{flatten_hashed_dict_name}_{flatten_hashed_dict_code}.pkl')                                                
        cached_flatten_list_path = os.path.join(self.cache_dir, 
                                                f'{flatten_hashed_list_name}_{flatten_hashed_list_code}.pkl')
        
        if self.use_cache and os.path.exists(cached_flatten_dict_path) and os.path.exists(cached_flatten_list_path):
            print("load flattened dicts from {}".format(cached_flatten_dict_path))
            self.reference_dataset_dicts = mmcv.load(cached_flatten_dict_path)
            self.reference_dataset_lists = mmcv.load(cached_flatten_list_path)
            print("load flattened lists from {}".format(cached_flatten_list_path))
        
        elif self.use_cache and os.path.exists(refer_cache_path):
            print("load cached dataset dicts from {}".format(refer_cache_path))
            # logger.info("load cached dataset dicts from {}".format(refer_cache_path))
            self.reference_dataset_dicts = mmcv.load(refer_cache_path)
            (self.reference_dataset_lists, 
                self.reference_dataset_dicts) = misc.flat_dataset_dicts_with_allo_pose(
                self.reference_dataset_dicts,
                visib_fract_threshold=self.visib_fraction_ratio,
                min_amount_of_instances=self.min_amount_of_instances,
                ) # flatten the image-level dict to instance-level dict
            mmcv.mkdir_or_exist(os.path.dirname(cached_flatten_list_path))
            mmcv.dump(self.reference_dataset_lists, cached_flatten_list_path, protocol=4)
            logger.info("Dumped reference_dataset_lists to {}".format(cached_flatten_list_path))
            
            mmcv.mkdir_or_exist(os.path.dirname(cached_flatten_dict_path))
            mmcv.dump(self.reference_dataset_dicts, cached_flatten_dict_path, protocol=4)
            logger.info("Dumped reference_dataset_dicts to {}".format(cached_flatten_dict_path))
        
        else:
            print("generate dataset dict ....")
            self.reference_dataset_dicts = list() # the whole dataset information
            self.reference_dataset_image_count = 0
            self.reference_dataset_instance_count = 0
            self.total_image_count = 0
            self.total_instance_count = 0
            refer_dataset_dir = os.path.join(self.data_dir, self.dataset_name)

            for ref_scene_idx in sorted(os.listdir(refer_dataset_dir)):
                try:
                    ref_scene_idx = int(ref_scene_idx)
                except:
                    continue
                if (ref_scene_idx + 1) % 50 == 0:
                    print("processing scene: {}, total_img:{}, total_inst:{}, valid_images: {}, valid_instances: {}".format(
                        ref_scene_idx + 1, 
                        self.total_image_count,
                        self.total_instance_count,
                        self.reference_dataset_image_count, 
                        self.reference_dataset_instance_count
                    ))
                ref_scene_dir = os.path.join(refer_dataset_dir, '{:08d}'.format(ref_scene_idx))
                
                refer_rgb_dir = os.path.join(ref_scene_dir, 'rgb')
                refer_depth_dir = os.path.join(ref_scene_dir, 'depth')
                refer_mask_dir = os.path.join(ref_scene_dir, 'mask_visib')
                refer_pose_data = mmcv.load(os.path.join(ref_scene_dir, 'scene_gt.json'))
                refer_camK_data = mmcv.load(os.path.join(ref_scene_dir, 'scene_camera.json'))
                refer_bbox_data = mmcv.load(os.path.join(ref_scene_dir, 'scene_gt_info.json'))
                
                for view_idx, view_dat in refer_pose_data.items():
                    view_idx = int(view_idx)
                    view_rgb_path = os.path.join(refer_rgb_dir, '{:08d}.png'.format(view_idx))
                    if not os.path.exists(view_rgb_path):
                        view_rgb_path = os.path.join(refer_rgb_dir, '{:08d}.jpg'.format(view_idx))
                    if not os.path.exists(view_rgb_path):
                        # print('Error: {} not exists!'.format(view_rgb_path))
                        continue
                    
                    self.total_image_count += 1

                    view_inst_bboxes = refer_bbox_data[str(view_idx)]
                    view_depth_scale = np.float32(refer_camK_data[str(view_idx)]['depth_scale'])
                    view_depth_path = os.path.join(refer_depth_dir, '{:08d}.png'.format(view_idx))
                    view_camK = np.array(refer_camK_data[str(view_idx)]['cam_K'], dtype=np.float32).reshape(3, 3)

                    view_record = {
                            'camK': view_camK,
                            "image_id": view_idx,
                            'rgb_path': view_rgb_path,
                            'depth_path': view_depth_path,    
                            'depth_scale': view_depth_scale,
                            "dataset_name": self.dataset_name,
                        }
                    view_instance_records = list()
                    view_instance_counter = dict()
                    for inst_idx, inst_dat in enumerate(view_dat):
                        inst_objID = int(inst_dat['obj_id'])
                        if inst_objID not in self.OBJ_IDs_to_NAMEs_DICT:
                            continue
                        inst_R = np.array(inst_dat['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
                        inst_t = np.array(inst_dat['cam_t_m2c'], dtype=np.float32).reshape(3)

                        self.total_instance_count += 1

                        ref_mask_path = os.path.join(refer_mask_dir, '{:08d}_{:08d}.png'.format(view_idx, inst_idx))

                        inst_pose = np.eye(4)
                        inst_pose[:3, 3] = inst_t
                        inst_pose[:3, :3] = inst_R                        
                        inst_visib_fract = view_inst_bboxes[inst_idx]['visib_fract']
                        if inst_visib_fract < self.visib_fraction_ratio:
                            continue

                        x1, y1, x2, y2 = map(int, view_inst_bboxes[inst_idx]['bbox_visib'])

                        inst_bbox = np.array([x1, y1, x2, y2])
                        mask_single = mmcv.imread(ref_mask_path, "unchanged").astype(bool).astype(np.uint8)
                        area = mask_single.sum()
                        if area < 500:  # filter out too small or nearly invisible instances
                            continue
                        if self.mask_morph:
                            kernel = np.ones((self.mask_morph_kernel_size, self.mask_morph_kernel_size))
                            mask_single = cv2.morphologyEx(mask_single.astype(np.uint8), cv2.MORPH_CLOSE, kernel) # remove holes
                            mask_single = cv2.morphologyEx(mask_single, cv2.MORPH_OPEN, kernel)  # remove outliers 

                        inst_record = {
                            'pose': inst_pose,
                            'bbox': inst_bbox,
                            'objID': inst_objID,
                            "bbox_mode": 'XYXY_ABS',
                            'visib_fract': inst_visib_fract,
                            "segmentation": misc.binary_mask_to_rle(mask_single, compressed=True),
                        }
                        
                        if inst_objID not in view_instance_counter:
                            view_instance_counter[inst_objID] = 0
                        view_instance_counter[inst_objID] += 1

                        view_instance_records.append(inst_record)
                    
                    if len(view_instance_records) == 0:  # filter im without anno
                        continue
                    
                    multi_instances_exist = False
                    for inst_objID, inst_count in view_instance_counter.items():
                        if inst_count >= 2:
                            multi_instances_exist = True
                    if multi_instances_exist:
                        continue
                    
                    view_record['annotations'] = view_instance_records
                    self.reference_dataset_dicts.append(view_record)
                    self.reference_dataset_image_count += 1
                    self.reference_dataset_instance_count += len(view_instance_records)
        
            if self.use_cache:
                mmcv.mkdir_or_exist(os.path.dirname(refer_cache_path))
                mmcv.dump(self.reference_dataset_dicts, refer_cache_path, protocol=4)
                logger.info("Dumped dataset_dicts to {}".format(refer_cache_path))

                (self.reference_dataset_lists, 
                    self.reference_dataset_dicts) = misc.flat_dataset_dicts_with_allo_pose(
                    self.reference_dataset_dicts,
                    visib_fract_threshold=self.visib_fraction_ratio,
                    min_amount_of_instances=self.min_amount_of_instances,
                    ) # flatten the image-level dict to instance-level dict
                mmcv.mkdir_or_exist(os.path.dirname(cached_flatten_list_path))
                mmcv.dump(self.reference_dataset_lists, cached_flatten_list_path, protocol=4)
                logger.info("Dumped reference_dataset_lists to {}".format(cached_flatten_list_path))
                
                mmcv.mkdir_or_exist(os.path.dirname(cached_flatten_dict_path))
                mmcv.dump(self.reference_dataset_dicts, cached_flatten_dict_path, protocol=4)
                logger.info("Dumped reference_dataset_dicts to {}".format(cached_flatten_dict_path))
        
        self.reference_dataset_image_count = len(self.reference_dataset_dicts)
        self.selected_objIDs = list(self.reference_dataset_dicts.keys())
        self.view_conuter = len(self.reference_dataset_lists)
        self.obj_conuter = len(self.selected_objIDs)
        print('Total instances: {}, objects: {}'.format(self.view_conuter, self.obj_conuter))

    def __len__(self):
        return self.view_conuter
    
    def __getitem__(self, idx):
        while True:
            try:
                dataset_dict = self.fetch_data(idx)
                if dataset_dict is None:
                    idx = np.random.randint(0, len(self))
                    continue
                else:
                    break
            except Exception as e:
                print(str(idx) + ', loading exception ocurred: ', e)
                idx = np.random.randint(0, len(self))
                continue
        return dataset_dict
    
    # def __getitem__(self, idx):
    #     dataset_dict = self.fetch_data(idx)
    #     return dataset_dict

    def load_multiview_refer_data(self, inst_objID, indices, cam_dist_factor):
        dataset_dict = dict()
        for sample_idx in indices:
            output = self.reference_dataset_dicts[inst_objID]['data'][sample_idx]
            output = self.read_refer_entry_data(output, cam_dist_factor)
            for key, val in output.items():
                if key not in dataset_dict:
                    dataset_dict[key] = list()
                dataset_dict[key].append(val)

        for key, val in dataset_dict.items():
            if isinstance(val[0], torch.Tensor):
                dataset_dict[key] = torch.stack(val, dim=0)
            else:
                dataset_dict[key] = val
        
        return dataset_dict
    

    def load_multiview_query_data(self, inst_objID, indices, cam_dist_factor):
        que_data_dict = dict()
        nnb_data_dict = dict()
        for sample_idx in indices:
            obj_data = self.reference_dataset_dicts[inst_objID]

            que_data = self.read_query_entry_data(obj_data['data'][sample_idx])

            que_allo_R = que_data['allo_RT'][:3, :3]
            obj_allo_Rs = torch.as_tensor(obj_data['allo_pose'][:, :3, :3], dtype=torch.float32)  # Nx3x3

            Rtrace = torch.einsum('mij,jk->mik', obj_allo_Rs[:, :3, :3], que_allo_R[:3, :3].T)
            Rcosim = torch.einsum('mii->m', Rtrace) / 2.0 - 0.5
            ranked_Rdist_vals, ranked_Rdist_inds = torch.topk(Rcosim, k=len(Rcosim), dim=0, largest=True)        
            que_basin_R_threshold = torch.cos(torch.tensor(self.nnb_Rmat_threshold / 180 * torch.pi))
            Rrad_within_basin = ranked_Rdist_vals >= que_basin_R_threshold # [1.0, -1.0]
            nnb_Rinds = ranked_Rdist_inds[Rrad_within_basin]
            random_ind = torch.randperm(len(nnb_Rinds))[0]
            
            selected_nnb_ind = nnb_Rinds[random_ind]
            nnb_data = obj_data['data'][selected_nnb_ind]
            nnb_data = self.read_refer_entry_data(nnb_data, cam_dist_factor)

            for key, val in que_data.items():
                if key not in que_data_dict:
                    que_data_dict[key] = list()
                que_data_dict[key].append(val)

            for key, val in nnb_data.items():
                if key not in nnb_data_dict:
                    nnb_data_dict[key] = list()
                nnb_data_dict[key].append(val)

        for key, val in que_data_dict.items():
            if isinstance(val[0], torch.Tensor):
                que_data_dict[key] = torch.stack(val, dim=0)
            else:
                que_data_dict[key] = val

        for key, val in nnb_data_dict.items():
            if isinstance(val[0], torch.Tensor):
                nnb_data_dict[key] = torch.stack(val, dim=0)
            else:
                nnb_data_dict[key] = val
        
        return que_data_dict, nnb_data_dict

    def fetch_data(self, idx):
        dataset_dict = dict()
        inst_objID = self.reference_dataset_lists[idx]['inst_annos']['objID']
        obj_allo_Rs = self.reference_dataset_dicts[inst_objID]['allo_pose'][:, :3, :3]
        obj_allo_Rs = torch.as_tensor(obj_allo_Rs, dtype=torch.float32)  # Nx3x3
        # obj_viewpoints = obj_allo_Rs[:, 2, :3] # Nx3
        obj_viewpoints = py3d_transform.matrix_to_axis_angle(obj_allo_Rs) # Nx3x3 -> Nx3
        camera_dist_factor = 1 + self.DZI_camera_dist_ratio * np.random.random_sample() # [1, 1.5]

        query_inds = py3d_ops.sample_farthest_points(
            obj_viewpoints[None, ...], K=self.query_view_num, random_start_point=True)[1].squeeze(0)
        
        query_dict, qnnb_dict = self.load_multiview_query_data(inst_objID, query_inds, camera_dist_factor)
        dataset_dict['query_dict'] = query_dict
        dataset_dict['qnnb_dict'] = qnnb_dict

        sample_inds = py3d_ops.sample_farthest_points(obj_viewpoints[None, ...], 
                                                      K=self.refer_view_num + self.rand_view_num, 
                                                      random_start_point=True)[1].squeeze(0)
        ref_inds = sample_inds[:self.refer_view_num]
        rand_inds = sample_inds[self.refer_view_num:]

        refer_dict = self.load_multiview_refer_data(inst_objID, ref_inds, camera_dist_factor)
        dataset_dict['refer_dict'] = refer_dict

        rand_dict = self.load_multiview_refer_data(inst_objID, rand_inds, camera_dist_factor)
        dataset_dict['rand_dict'] = rand_dict

        return dataset_dict
      
    def read_refer_entry_data(self, dataset_dict, cam_dist_factor):
        dataset_dict = copy.deepcopy(dataset_dict)
        inst_annos = dataset_dict.pop("inst_annos")

        obj_instID = int(inst_annos['objID'])
        camK = dataset_dict['camK'].astype(np.float32)
        obj_RT = inst_annos['pose'].astype(np.float32)
        obj_diameter = self.obj_diameters[obj_instID]/1000.0 # in meters

        allo_RT = obj_RT.copy()
        allo_RT[:3, :3] = gs_utils.egocentric_to_allocentric(allo_RT)[:3, :3]

        image = np.array(mmcv.imread(dataset_dict['rgb_path'], 'color', self.COLOR_TYPE), dtype=np.uint8)

        if np.random.rand() < self.COLOR_AUG_PROB:    # augment the synthetic+real image randomly   
            image = self.rgb_add_noise(image)     ### augment the imag 

        image = image.astype(np.float32) / 255.0
        orig_hei, orig_wid = image.shape[:2]
        mask = misc.cocosegm2mask(inst_annos["segmentation"], orig_hei, orig_wid).astype(np.bool_).astype(np.float32)

        camera_dist = cam_dist_factor * max(camK[0, 0], camK[1, 1]) / self.img_scale

        scaled_T = obj_RT[:3, 3] / obj_diameter
        obj_2D_center = camK @ scaled_T
        zoom_bbox_center = obj_2D_center[:2] / obj_2D_center[2:3]  #
        zoom_bbox_scale = camera_dist * self.img_scale / scaled_T[2]

        zoom_image = misc.crop_resize_by_warp_affine(
            image, zoom_bbox_center, zoom_bbox_scale, self.img_scale, interpolation=cv2.INTER_LINEAR)
        zoom_mask = misc.crop_resize_by_warp_affine(
            mask, zoom_bbox_center, zoom_bbox_scale, self.img_scale, interpolation=cv2.INTER_NEAREST)
        
        allo_RT = torch.as_tensor(allo_RT, dtype=torch.float32)
        zoom_mask = torch.as_tensor(zoom_mask, dtype=torch.float32).unsqueeze(0)
        zoom_image = torch.as_tensor(zoom_image, dtype=torch.float32).permute(2, 0, 1)

        return {'zoom_image': zoom_image, 
                'zoom_mask': zoom_mask, 
                'allo_RT': allo_RT}
    
    def read_query_entry_data(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        inst_annos = dataset_dict.pop("inst_annos")
        obj_RT = inst_annos['pose'].astype(np.float32)

        allo_RT = obj_RT.copy()
        allo_RT[:3, :3] = gs_utils.egocentric_to_allocentric(allo_RT)[:3, :3]

        bbox_mode = inst_annos['bbox_mode']
        rgb_path = dataset_dict['rgb_path']
        image = np.array(mmcv.imread(rgb_path, 'color', self.COLOR_TYPE), dtype=np.uint8)
        
        if np.random.rand() < self.COLOR_AUG_PROB:    # augment the synthetic+real image randomly   
            image = self.rgb_add_noise(image)     ### augment the imag 

        image = image.astype(np.float32) / 255.0
        orig_hei, orig_wid = image.shape[:2]
        if bbox_mode == 'XYXY_ABS':
            bbox = inst_annos['bbox']
        else:
            print('bbox_mode not supported!')
            raise NotImplementedError
        mask = misc.cocosegm2mask(inst_annos["segmentation"], orig_hei, orig_wid)

        raw_long_size = max(orig_hei, orig_wid)
        raw_short_size = min(orig_hei, orig_wid)
        raw_aspect_ratio = raw_short_size / raw_long_size
        if orig_hei < orig_wid:
            new_wid = self.query_longside_scale
            new_hei = int(new_wid * raw_aspect_ratio)
        else:
            new_hei = self.query_longside_scale
            new_wid = int(new_hei * raw_aspect_ratio)
        
        rescaled_image = cv2.resize(image, (new_wid, new_hei), interpolation=cv2.INTER_LINEAR)
        rescaled_mask = cv2.resize(mask, (new_wid, new_hei), interpolation=cv2.INTER_LINEAR)
        
        dzi_bbox_center, dzi_bbox_scale = misc.aug_bbox_DZI(
            bbox, orig_hei, orig_wid, 
            scale_ratio=self.DZI_SCALE_RATIO, 
            shift_ratio=self.DZI_SHIFT_RATIO, 
            pad_ratio=self.DZI_PAD_RATIO,
        )
        
        dzi_image = misc.crop_resize_by_warp_affine(
            image, dzi_bbox_center, dzi_bbox_scale, self.img_scale, interpolation=cv2.INTER_LINEAR)
        dzi_mask = misc.crop_resize_by_warp_affine(
            mask, dzi_bbox_center, dzi_bbox_scale, self.img_scale, interpolation=cv2.INTER_NEAREST)
        
        
        rescaled_image = torch.as_tensor(rescaled_image, dtype=torch.float32).permute(2, 0, 1)
        rescaled_mask = torch.as_tensor(rescaled_mask, dtype=torch.float32).unsqueeze(0)
        dzi_image = torch.as_tensor(dzi_image, dtype=torch.float32).permute(2, 0, 1)
        dzi_mask = torch.as_tensor(dzi_mask, dtype=torch.float32).unsqueeze(0)

        allo_RT = torch.as_tensor(allo_RT, dtype=torch.float32)        

        return {
            'allo_RT': allo_RT,
            'dzi_image': dzi_image, 
            'dzi_mask': dzi_mask, 
            'rescaled_image': rescaled_image, 
            'rescaled_mask': rescaled_mask,
            'rgb_path': rgb_path,
            }
    
    def collate_fn(self, batch):
        """
        batchify the data
        """
        new_batch = dict()
        for each_dat in batch:
            for key, val in each_dat.items():
                if isinstance(val, dict):
                    if key not in new_batch:
                        new_batch[key] = dict()
                    for sub_key, sub_val in val.items():
                        if sub_key not in new_batch[key]:
                            new_batch[key][sub_key] = list()
                        new_batch[key][sub_key].append(sub_val)
                else:
                    if key not in new_batch:
                        new_batch[key] = list()
                    new_batch[key].append(val)
        
        for key, val in new_batch.items():
            if isinstance(val, list):
                if isinstance(val[0], torch.Tensor):
                    new_batch[key] = torch.stack(val, dim=0)
                else:
                    new_batch[key] = [s_v[0] for s_v in val]
            
            elif isinstance(val, dict):
                for sub_key, sub_val in val.items():
                    try:
                        if isinstance(sub_val[0], torch.Tensor):
                            new_batch[key][sub_key] = torch.stack(sub_val, dim=0)   
                        else:
                            new_batch[key][sub_key] = [s_v[0] for s_v in sub_val]  
                    except:
                        pass
        return new_batch

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = np.random
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)
    
    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img
    
    def apply_megapose_augmentation(self, x):
        if isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x)
        for aug in self.RGB_AUG:
            x = aug(x)
        x = np.asarray(x)
        return x

if __name__ == '__main__':

    img_scale = 244    
    refer_view_num = 8
    rand_view_num = 16    
    query_longside_scale = 672
    print('PROJ_ROOT: ', PROJ_ROOT)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = os.path.join(PROJ_ROOT, 'dataspace/bop_dataset/MegaPose')
    dataset = MegaPose_Dataset(data_dir=data_dir,
                                rand_view_num=rand_view_num,
                                refer_view_num=refer_view_num,
                                query_longside_scale=query_longside_scale, 
                                )
    print('num_images: ', len(dataset))

    batch_size = 1
    train_sampler = None
    data_loader = torch.utils.data.DataLoader(dataset,
                                              num_workers=4, 
                                              batch_size=batch_size, 
                                              shuffle=True,
                                              pin_memory=False,
                                              drop_last=False,
                                              collate_fn=dataset.collate_fn
                                              )
    total_batches = len(data_loader)  # total batches for all epochs
    print('batch number: ', total_batches)
    print('num_objects: ', len(dataset.selected_objIDs))

    for batch_data in data_loader:

        query_dict = batch_data['query_dict']
        refer_dict = batch_data['refer_dict']
        qnnb_dict = batch_data['qnnb_dict']
        rand_dict = batch_data['rand_dict']

        que_full_image = query_dict['rescaled_image']
        que_full_mask = query_dict['rescaled_mask']
        que_dzi_image = query_dict['dzi_image']
        que_dzi_mask = query_dict['dzi_mask']

        ref_zoom_image = refer_dict['zoom_image']
        ref_zoom_mask = refer_dict['zoom_mask']

        rand_zoom_image = rand_dict['zoom_image']
        rand_zoom_mask = rand_dict['zoom_mask']

        qnnb_zoom_image = qnnb_dict['zoom_image']
        qnnb_zoom_mask = qnnb_dict['zoom_mask']

        print('que_full_image: ', que_full_image.shape)
        print('que_full_mask: ', que_full_mask.shape)
        print('que_dzi_image: ', que_dzi_image.shape)
        print('que_dzi_mask: ', que_dzi_mask.shape)

        print('ref_zoom_image: ', ref_zoom_image.shape)
        print('ref_zoom_mask: ', ref_zoom_mask.shape)
        
        print('rand_zoom_image: ', rand_zoom_image.shape)
        print('rand_zoom_mask: ', rand_zoom_mask.shape)
        
        print('qnnb_zoom_image: ', qnnb_zoom_image.shape)
        print('qnnb_zoom_mask: ', qnnb_zoom_mask.shape)

        break




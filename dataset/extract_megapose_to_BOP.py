import os
import io
import sys
import cv2
import json
import time
import copy
import mmcv
import torch
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
import imageio.v2 as imageio
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Iterator, List, Optional, Union

import megapose.utils.tensor_collection as tc
from megapose.lib3d.transform import Transform
from megapose.utils.webdataset import tarfile_samples, tarfile_to_samples


ListBbox = List[int]
ListPose = List[List[float]]
Resolution = Tuple[int, int]
SingleDataJsonType = Union[str, float, ListPose, int, ListBbox, Any]
DataJsonType = Union[Dict[str, SingleDataJsonType], List[SingleDataJsonType]]
SceneObservationTensorCollection = tc.PandasTensorCollection

def transform_to_list(T: Transform) -> ListPose:
    return [T.quaternion.coeffs().tolist(), T.translation.tolist()]

@dataclass
class ObjectData:
    # NOTE (Yann): bbox_amodal, bbox_modal, visib_fract should be moved to SceneObservation
    label: str
    TWO: Optional[Transform] = None
    unique_id: Optional[int] = None
    bbox_amodal: Optional[np.ndarray] = None  # (4, ) array [xmin, ymin, xmax, ymax]
    bbox_modal: Optional[np.ndarray] = None  # (4, ) array [xmin, ymin, xmax, ymax]
    visib_fract: Optional[float] = None
    TWO_init: Optional[
        Transform
    ] = None  # Some pose estimation datasets (ModelNet) provide an initial pose estimate
    #  NOTE: This should be loaded externally

    def to_json(self) -> Dict[str, SingleDataJsonType]:
        d: Dict[str, SingleDataJsonType] = dict(label=self.label)
        for k in ("TWO", "TWO_init"):
            if getattr(self, k) is not None:
                d[k] = transform_to_list(getattr(self, k))
        for k in ("bbox_amodal", "bbox_modal"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k).tolist()
        for k in ("visib_fract", "unique_id"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d

    @staticmethod
    def from_json(d: DataJsonType) -> "ObjectData":
        assert isinstance(d, dict)
        label = d["label"]
        assert isinstance(label, str)
        data = ObjectData(label=label)
        for k in ("TWO", "TWO_init"):
            if k in d:
                item = d[k]
                assert isinstance(item, list)
                quat_list, trans_list = item
                assert isinstance(quat_list, list)
                assert isinstance(trans_list, list)
                quat = tuple(quat_list)
                trans = tuple(trans_list)
                setattr(data, k, Transform(quat, trans))
        for k in ("unique_id", "visib_fract"):
            if k in d:
                setattr(data, k, d[k])
        for k in ("bbox_amodal", "bbox_modal"):
            if k in d:
                setattr(data, k, np.array(d[k]))
        return data

@dataclass
class CameraData:
    K: Optional[np.ndarray] = None
    resolution: Optional[Resolution] = None
    TWC: Optional[Transform] = None
    camera_id: Optional[str] = None
    TWC_init: Optional[
        Transform
    ] = None  # Some pose estimation datasets (ModelNet) provide an initial pose estimate
    # NOTE: This should be loaded externally

    def to_json(self) -> str:
        d: Dict[str, SingleDataJsonType] = dict()
        for k in ("TWC", "TWC_init"):
            if getattr(self, k) is not None:
                d[k] = transform_to_list(getattr(self, k))
        for k in ("K",):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k).tolist()
        for k in ("camera_id", "resolution"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return json.dumps(d)

    @staticmethod
    def from_json(data_str: str) -> "CameraData":
        d: DataJsonType = json.loads(data_str)
        assert isinstance(d, dict)
        data = CameraData()
        for k in ("TWC", "TWC_init"):
            if k in d:
                item = d[k]
                assert isinstance(item, list)
                quat_list, trans_list = item
                assert isinstance(quat_list, list)
                assert isinstance(trans_list, list)
                quat = tuple(quat_list)
                trans = tuple(trans_list)
                setattr(data, k, Transform(tuple(quat), tuple(trans)))
        for k in ("camera_id",):
            if k in d:
                setattr(data, k, d[k])
        for k in ("K",):
            if k in d:
                setattr(data, k, np.array(d[k]))
        if "resolution" in d:
            assert isinstance(d["resolution"], list)
            h, w = d["resolution"]
            assert isinstance(h, int)
            assert isinstance(w, int)
            data.resolution = (h, w)
        return data

@dataclass
class ObservationInfos:
    scene_id: str
    view_id: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(data_str: str) -> "ObservationInfos":
        d = json.loads(data_str)
        assert "scene_id" in d
        assert "view_id" in d
        return ObservationInfos(scene_id=d["scene_id"], view_id=d["view_id"])

@dataclass
class SceneObservation:
    rgb: Optional[np.ndarray] = None  # (h,w,3) uint8 numpy array
    depth: Optional[np.ndarray] = None  # (h, w), np.float32
    segmentation: Optional[np.ndarray] = None  # (h, w), np.uint32 (important);
    # contains objects unique ids. int64 are not handled and can be dangerous when used with PIL
    infos: Optional[ObservationInfos] = None
    object_datas: Optional[List[ObjectData]] = None
    camera_data: Optional[CameraData] = None
    binary_masks: Optional[
        Dict[int, np.ndarray]
    ] = None  # dict mapping unique id to (h, w) np.bool_

    @staticmethod
    def collate_fn(
#         batch: List[SceneObservation], object_labels: Optional[List[str]] = None
        batch, object_labels=None
    ) -> Dict[Any, Any]:
        """Collate a batch of SceneObservation objects.

        Args:
            object_labels: If passed in parse only those object labels.

        Returns:
            A dict with fields
                cameras: PandasTensorCollection
                rgb: torch.tensor [B,3,H,W] torch.uint8
                depth: torch.tensor [B,1,H,W]
                im_infos: List[dict]
                gt_detections: SceneObservationTensorCollection
                gt_data: SceneObservationTensorCollection


        """
        if object_labels is not None:
            object_labels = set(object_labels)

        cam_infos, K = [], []
        im_infos = []
        gt_data = []
        gt_detections = []
        initial_data = []
        batch_im_id = -1
        rgb_images = []
        depth_images = []

        for n, data in enumerate(batch):
            # data is of type SceneObservation
            batch_im_id += 1
            im_info = dict(
                scene_id=data.infos.scene_id,
                view_id=data.infos.view_id,
                batch_im_id=batch_im_id,
            )
            im_infos.append(im_info)

            K.append(data.camera_data.K)
            cam_info = dict(
                TWC=data.camera_data.TWC,
                resolution=data.camera_data.resolution,
            )
            cam_infos.append(cam_info)

            # [3,H,W]
            rgb = torch.as_tensor(data.rgb).permute(2, 0, 1).to(torch.uint8)
            rgb_images.append(rgb)
            if data.depth is not None:
                depth = np.expand_dims(data.depth, axis=0)
            else:
                depth = np.array([])

            depth_images.append(depth)

            gt_data_ = data.as_pandas_tensor_collection(object_labels=object_labels)
            gt_data_.infos["batch_im_id"] = batch_im_id  # Add batch_im_id
            gt_data.append(gt_data_)

            initial_data_ = None
            if hasattr(gt_data_, "poses_init"):
                initial_data_ = copy.deepcopy(gt_data_)
                initial_data_.poses = initial_data_.poses_init
                initial_data.append(initial_data_)

            # Emulate detection data
            gt_detections_ = copy.deepcopy(gt_data_)
            gt_detections_.infos["score"] = 1.0  # Add score field
            gt_detections.append(gt_detections_)

        gt_data = tc.concatenate(gt_data)
        gt_detections = tc.concatenate(gt_detections)
        if initial_data:
            initial_data = tc.concatenate(initial_data)
        else:
            initial_data = None

        cameras = tc.PandasTensorCollection(
            infos=pd.DataFrame(cam_infos),
            K=torch.as_tensor(np.stack(K)),
        )
        return dict(
            cameras=cameras,
            rgb=torch.stack(rgb_images),  # [B,3,H,W]
            depth=torch.as_tensor(np.stack(depth_images)),  # [B,1,H,W] or [B,0]
            im_infos=im_infos,
            gt_detections=gt_detections,
            gt_data=gt_data,
            initial_data=initial_data,
        )

    def as_pandas_tensor_collection(
        self,
        object_bbox_id = None,
        object_labels: Optional[List[str]] = None,
    ) -> SceneObservationTensorCollection:
        """Convert SceneData to a PandasTensorCollection representation."""
        obs = self

        assert obs.camera_data is not None
        assert obs.object_datas is not None

        infos = []
        TWO = []
        bboxes = []
        masks = []
        TWC = torch.as_tensor(obs.camera_data.TWC.matrix).float()

        TWO_init = []
        TWC_init = None
        if obs.camera_data.TWC_init is not None:
            TWC_init = torch.as_tensor(obs.camera_data.TWC_init.matrix).float()

        if object_bbox_id is not None:
            obj_data = obs.object_datas[object_bbox_id]
            info = dict(
                label=obj_data.label,
                scene_id=obs.infos.scene_id,
                view_id=obs.infos.view_id,
                visib_fract=getattr(obj_data, "visib_fract", 1),
            )
            infos.append(info)
            TWO.append(torch.tensor(obj_data.TWO.matrix).float())
            bboxes.append(torch.tensor(obj_data.bbox_modal).float())

            if obs.binary_masks is not None:
                binary_mask = torch.tensor(obs.binary_masks[obj_data.unique_id]).float()
                masks.append(binary_mask)

            if obs.segmentation is not None:
                binary_mask = np.zeros_like(obs.segmentation, dtype=np.bool_)
                binary_mask[obs.segmentation == obj_data.unique_id] = 1
                binary_mask = torch.as_tensor(binary_mask).float()
                masks.append(binary_mask)

            if obj_data.TWO_init:
                TWO_init.append(torch.tensor(obj_data.TWO_init.matrix).float())
        else:
            for n, obj_data in enumerate(obs.object_datas):
                if object_labels is not None and obj_data.label not in object_labels:
                    continue
                info = dict(
                    label=obj_data.label,
                    scene_id=obs.infos.scene_id,
                    view_id=obs.infos.view_id,
                    visib_fract=getattr(obj_data, "visib_fract", 1),
                )
                infos.append(info)
                TWO.append(torch.tensor(obj_data.TWO.matrix).float())
                bboxes.append(torch.tensor(obj_data.bbox_modal).float())

                if obs.binary_masks is not None:
                    binary_mask = torch.tensor(obs.binary_masks[obj_data.unique_id]).float()
                    masks.append(binary_mask)

                if obs.segmentation is not None:
                    binary_mask = np.zeros_like(obs.segmentation, dtype=np.bool_)
                    binary_mask[obs.segmentation == obj_data.unique_id] = 1
                    binary_mask = torch.as_tensor(binary_mask).float()
                    masks.append(binary_mask)

                if obj_data.TWO_init:
                    TWO_init.append(torch.tensor(obj_data.TWO_init.matrix).float())

        TWO = torch.stack(TWO)
        bboxes = torch.stack(bboxes)
        infos = pd.DataFrame(infos)
        if len(masks) > 0:
            masks = torch.stack(masks)
        else:
            masks = None

        B = len(infos)

        TCW = torch.linalg.inv(TWC)  # [4,4]

        # [B,4,4]
        TCO = TCW.unsqueeze(0) @ TWO
        TCO_init = None
        if len(TWO_init):
            TCO_init = torch.linalg.inv(TWC_init).unsqueeze(0) @ torch.stack(TWO_init)
        K = torch.tensor(obs.camera_data.K).unsqueeze(0).expand([B, -1, -1])

        data = tc.PandasTensorCollection(
            infos=infos,
            TCO=TCO,
            bboxes=bboxes,
            poses=TCO,
            K=K,
            masks=masks
        )

        # Only register the mask tensor if it is not None
        if masks is not None:
            data.register_tensor("masks", masks)
        if TCO_init is not None:
            data.register_tensor("TCO_init", TCO_init)
            data.register_tensor("poses_init", TCO_init)
        return data

def load_scene_ds_obs(
    sample: Dict[str, Union[bytes, str]],
    depth_scale: float = 1000.0,
    load_depth: bool = False,
    label_format: str = "{label}",
) -> SceneObservation:

    assert isinstance(sample["rgb.png"], bytes)
    assert isinstance(sample["segmentation.png"], bytes)
    assert isinstance(sample["depth.png"], bytes)
    assert isinstance(sample["camera_data.json"], bytes)
    assert isinstance(sample["infos.json"], bytes)

    rgb = np.array(imageio.imread(io.BytesIO(sample["rgb.png"])))
    segmentation = np.array(imageio.imread(io.BytesIO(sample["segmentation.png"])))
    segmentation = np.asarray(segmentation, dtype=np.uint32)
    depth = None
    if load_depth:
        depth = imageio.imread(io.BytesIO(sample["depth.png"]))
        depth = np.asarray(depth, dtype=np.float32)
        depth /= depth_scale

    object_datas_json: List[DataJsonType] = json.loads(sample["object_datas.json"])
    object_datas = [ObjectData.from_json(d) for d in object_datas_json]
    for obj in object_datas:
        obj.label = label_format.format(label=obj.label)

    camera_data = CameraData.from_json(sample["camera_data.json"])
    infos = ObservationInfos.from_json(sample["infos.json"])

    return SceneObservation(
        rgb=rgb,
        depth=depth,
        segmentation=segmentation,
        infos=infos,
        object_datas=object_datas,
        camera_data=camera_data,
    )


if __name__ == "__main__":
    megapose_data_root = "dataspace/MegaPose"
    local_dataspace = 'dataspace/MegaPose/train_pbr'

    sub_dataset = os.path.join(megapose_data_root, 'webdatasets/gso_1M')
    subtar_files = sorted([str(x) for x in Path(sub_dataset).iterdir() if x.suffix == ".tar"])
    print('Total number of scenes: ', len(subtar_files))

    valid_mesh_file_path = os.path.join(megapose_data_root, "google_scanned_objects/valid_meshes.json")
    all_valid_mesh_names = mmcv.load(valid_mesh_file_path)
    OBJ_CLASSES_IDXES_DICT = dict()
    for obj_idx, obj_class_name in enumerate(all_valid_mesh_names):
        OBJ_CLASSES_IDXES_DICT[obj_class_name] = obj_idx
    len(OBJ_CLASSES_IDXES_DICT)

    scene_counter = 0
    image_counter = 0
    instance_counter = 0
    visib_fract_thresh = 0.05
    num_scenes = len(subtar_files)
    start_timer = time.time()

    def yield_tar_dict(file):
        yield dict(url=file)
        
    def extract_samples(scene_url, scene_key, load_depth=False):
        sample: Dict[str, Union[bytes, str]] = dict()
        with tarfile.open(scene_url) as tar_f:
            for item_key in (
                "rgb.png", "depth.png", "segmentation.png",
                "infos.json", "camera_data.json", "object_datas.json"
            ):                
                tar_file_f = tar_f.extractfile(f"{scene_key}.{item_key}")
                sample[item_key] = tar_file_f.read()
            obs = load_scene_ds_obs(sample, load_depth=load_depth)
        return obs
            
    for six, src_tar_file in enumerate(subtar_files):    
        scene_idx = int(src_tar_file.split('/')[-1].split('.')[0])
        scene_dir = os.path.join(local_dataspace, '{:08d}'.format(scene_idx))

        scene_rgb_dir = os.path.join(scene_dir, 'rgb')
        scene_depth_dir = os.path.join(scene_dir, 'depth')
        scene_mask_dir = os.path.join(scene_dir, 'mask_visib')

        if not os.path.exists(scene_rgb_dir):
            os.makedirs(scene_rgb_dir)
        if not os.path.exists(scene_depth_dir):
            os.makedirs(scene_depth_dir)
        if not os.path.exists(scene_mask_dir):
            os.makedirs(scene_mask_dir)

        scene_gt_pose_path = os.path.join(scene_dir, 'scene_gt.json')
        scene_gt_bbox_path = os.path.join(scene_dir, 'scene_gt_info.json')
        scene_gt_camera_path = os.path.join(scene_dir, 'scene_camera.json')
        
        src_samples = tarfile_samples(yield_tar_dict(src_tar_file))
        scene_counter += 1
        
        scene_gt_camKs = dict()
        scene_gt_poses = dict()
        scene_gt_bboxes = dict()
        
        for view_idx, view_sample in enumerate(src_samples):
            view_info = load_scene_ds_obs(view_sample).as_pandas_tensor_collection()
            
            scene_key = view_sample['__key__']
            scene_url = view_sample['__url__']
                
            obs_entry = extract_samples(scene_url=scene_url, scene_key=scene_key, load_depth=True)
            view_color = obs_entry.rgb
            view_depth = (obs_entry.depth * 1000).astype(np.uint16)
            
            scene_gt_camKs[str(view_idx)] = {"cam_K": view_info.K[0].reshape(-1).tolist(), "depth_scale": 1.0}
            
            rgb_path = os.path.join(scene_rgb_dir, '{:08d}.png'.format(view_idx))
            dep_path = os.path.join(scene_depth_dir, '{:08d}.png'.format(view_idx))
            mmcv.imwrite(view_color, rgb_path)
            mmcv.imwrite(view_depth, dep_path)
            image_counter += 1

            view_poses = list()
            view_bboxes = list()
            view_inst_num = 0
            for bix, label in enumerate(view_info.infos.label):
                visib_fract = view_info.infos.visib_fract[bix]
                scene_id = view_info.infos.scene_id[bix]
                view_id = view_info.infos.view_id[bix]
                pose = view_info.poses[bix]
                
                if visib_fract < visib_fract_thresh:
                    continue
                
                view_data = obs_entry.as_pandas_tensor_collection(object_bbox_id=bix)
                obj_label = view_data.infos.label[0][4:] # starts with gso_
                obj_mask = (255 * view_data.masks[0]).type(torch.uint8).numpy()
                obj_pose = view_data.poses[0].type(torch.float32)
                obj_bbox = view_data.bboxes[0].type(torch.int16)
                
                pose_dict = {
                    "cam_R_m2c": obj_pose[:3, :3].reshape(-1).tolist(),
                    "cam_t_m2c": obj_pose[:3, 3].reshape(-1).tolist(),
                    "obj_id": OBJ_CLASSES_IDXES_DICT[obj_label],
                    "obj_label": obj_label,
                }
                bbox_dict = {
                    "bbox_visib": obj_bbox.tolist(),
                    "visib_fract": visib_fract,
                }
                
                view_poses.append(pose_dict)
                view_bboxes.append(bbox_dict)
                
                msk_path = os.path.join(scene_mask_dir, '{:08d}_{:08d}.png'.format(view_idx, view_inst_num))
                mmcv.imwrite(obj_mask, msk_path)
                view_inst_num += 1
                instance_counter += 1
                
            scene_gt_poses[str(view_idx)] = view_poses
            scene_gt_bboxes[str(view_idx)] = view_bboxes
            
        mmcv.dump(scene_gt_poses, scene_gt_pose_path)
        mmcv.dump(scene_gt_bboxes, scene_gt_bbox_path)
        mmcv.dump(scene_gt_camKs, scene_gt_camera_path)

        if six % 10 == 0:
            time_cost = time.time() - start_timer
            time_stamp = time.strftime('%m:%d_%H:%M:%S', time.localtime())
            print(f'{scene_counter}/{num_scenes}, views: {image_counter}, insts: {instance_counter}, ', f'cost: {time_cost}', f'time: {time_stamp}')

    print(f'Done! {image_counter}, {instance_counter}, total time cost: {time.time() - start_timer}')
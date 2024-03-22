import os
import cv2
import tqdm
import numpy as np
import os.path as osp
from pathlib import Path
from transforms3d import affines, quaternions


def parse_box(box_path):
    with open(box_path, 'r') as f:
        lines = f.readlines()
    data = [float(e) for e in lines[1].strip().split(',')]
    position = data[:3]
    quaternion = data[6:]
    rot_mat = quaternions.quat2mat(quaternion)
    T_ow = affines.compose(position, rot_mat, np.ones(3))
    return T_ow

def get_bbox3d(box_path):
    assert Path(box_path).exists()
    with open(box_path, 'r') as f:
        lines = f.readlines()
    box_data = [float(e) for e in lines[1].strip().split(',')]
    ex, ey, ez = box_data[3: 6]
    bbox_3d = np.array([
        [-ex, -ey, -ez],
        [ex,  -ey, -ez],
        [ex,  -ey, ez],
        [-ex, -ey, ez],
        [-ex,  ey, -ez],
        [ ex,  ey, -ez],
        [ ex,  ey, ez],
        [-ex,  ey, ez]
    ]) * 0.5
    bbox_3d_homo = np.concatenate([bbox_3d, np.ones((8, 1))], axis=1)
    return bbox_3d, bbox_3d_homo



def data_process_anno(data_dir):
    arkit_box_path = osp.join(data_dir, 'Box.txt')
    arkit_pose_path = osp.join(data_dir, 'ARposes.txt')
    arkit_intrin_path = osp.join(data_dir, 'Frames.txt')

    out_pose_dir = osp.join(data_dir, 'poses')
    Path(out_pose_dir).mkdir(parents=True, exist_ok=True)
    out_intrin_path = osp.join(data_dir, 'intrinsics.txt')
    out_bbox3D_path = osp.join(osp.dirname(data_dir), 'box3d_corners.txt')

    ##### read the ARKit 3D bounding box and convert to box corners
    bbox_3d, bbox_3d_homo = get_bbox3d(arkit_box_path)
    np.savetxt(out_bbox3D_path, bbox_3d)

    ##### read the ARKit camera intrinsics
    with open(arkit_intrin_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 0 and l[0] != '#']
    arkit_camk = np.array([[float(e) for e in l.split(',')] for l in lines])
    fx, fy, cx, cy = np.average(arkit_camk, axis=0)[2:]
    with open(out_intrin_path, 'w') as f:
        f.write('fx: {0}\nfy: {1}\ncx: {2}\ncy: {3}'.format(fx, fy, cx, cy))

    ##### read the ARKit camera poses
    T_O2W = parse_box(arkit_box_path) # 3D object bounding box is defined w.r.t. the world coordinate system

    with open(arkit_pose_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        index = 0
        for line in tqdm.tqdm(lines):
            if len(line) == 0 or line[0] == '#':
                continue

            eles = line.split(',')
            data = [float(e) for e in eles]

            position = data[1:4]
            quaternion = data[4:]
            rot_mat = quaternions.quat2mat(quaternion)
            rot_mat = rot_mat @ np.array([
                [1,  0,  0],
                [0, -1,  0],
                [0,  0, -1]])
            T_C2W = affines.compose(position, rot_mat, np.ones(3))
            T_W2C = np.linalg.inv(T_C2W)
            T_O2C = T_W2C @ T_O2W

            pose_save_path = osp.join(out_pose_dir, '{}.txt'.format(index))
            np.savetxt(pose_save_path, T_O2C)
            index += 1
            



if __name__ == "__main__":
    args = parse_args()
    data_dir = args.scanned_object_path
    assert osp.exists(data_dir), f"Scanned object path:{data_dir} not exists!"
    seq_dirs = os.listdir(data_dir)

    for seq_dir in seq_dirs:
        if '-refer' in seq_dir:
            ################ Parse scanned reference sequence ################
            print('=> Processing train sequence: ', seq_dir)
            video_path = osp.join(data_dir, seq_dir, 'Frames.m4v')
            print('=> parse video: ', video_path)
            data_process_anno(osp.join(data_dir, seq_dir), downsample_rate=1, hw=512)

        elif '-query' in seq_dir:
            ################ Parse scanned test sequence ################
            print('=> Processing test sequence: ', seq_dir)
            data_process_test(osp.join(data_dir, seq_dir), downsample_rate=1)
            pass

        else:
            
            continue


# python parse_scanned_data.py --scanned_object_path /home/dingding/Workspace/Others/OnePose_Plus_Plus/data/demo/teacan
import os.path as osp
import numpy as np
import math
import torch
import json
import copy
import transforms3d
import scipy.sparse as ssp
import os
import cv2
import pickle
from tqdm import tqdm

from virtualmarker.core.config import cfg, update_config, init_experiment_dir
from virtualmarker.utils.coord_utils import cam2pixel_matrix, bbox_clip_xyxy, bbox_xywh_to_xyxy, bbox_xyxy_to_xywh
from virtualmarker.dataset.joints_dataset import JointsDataset


class HP3D(JointsDataset):
    bbox_3d_shape = (2000, 2000, 2000)
    def __init__(self, mode, args, transform=None, master=None):
        super().__init__(mode, args, '3DHP', transform, master)
    
        # 3DHP joint set
        self.hp3d_joint_num = 28
        self.hp3d_joints_name = ('Spine3', 'Spine4', 'Spine2', 'Spine', 'Pelvis',                         # 4
                   'Neck', 'Head', 'Head_top',                                              # 7
                   'L_Clavicle', 'L_Shoulder', 'L_Elbow',                          # 10
                   'L_Wrist', 'L_Hand', 'R_Clavicle',                             # 13
                   'R_Shoulder', 'R_Elbow', 'R_Wrist',                          # 16
                   'R_Hand', 'L_Hip', 'L_Knee',                                   # 19
                   'L_Ankle', 'L_Foot', 'L_Toe',                                   # 22
                   'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot', 'R_Toe')     # 27
        self.hp3d_to_h36m = [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]
        self.hp3d_to_h36m_test = [14, 8, 9, 10, 11, 12, 13, 15, 1, 0, 0, 5, 6, 7, 2, 3, 4]
        self.hp3d_joint_num = cfg.dataset.num_joints-17 if cfg.dataset.num_joints != 17 else cfg.dataset.num_joints

        self.datalist = self.lazy_load_data()


    def lazy_load_data(self):
        lazy_annot_path = osp.join(self.annot_path, 'HP3D_{}.pkl'.format(self.data_split))
        if osp.exists(lazy_annot_path):
            if self.master:
                print('Lazy load annotations of 3DHP from ' + lazy_annot_path)
            with open(lazy_annot_path, 'rb') as f:
                datalist = pickle.load(f)
        else:
            datalist = self.load_data_json()
            try:
                with open(lazy_annot_path, 'wb') as f:
                    pickle.dump(datalist, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                if self.master:
                    print(e)
                    print('Skip writing to .pkl file.')
        for item in datalist:
            item['img_name'] = osp.join(self.img_dir, item['img_name'])
        return datalist

    def load_data_json(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        datalist = []
        with open(osp.join(self.annot_path, f'annotation_mpi_inf_3dhp_{self.data_split}.json'), 'r') as fid:
            database = json.load(fid)

        # iterate through the annotations
        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v

            img_name = osp.join('mpi_inf_3dhp_{}_set'.format(self.data_split), ann['file_name'])

            width, height = ann['width'], ann['height']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                bbox_xywh_to_xyxy(ann['bbox']), width, height)
            bbox = np.array(bbox_xyxy_to_xywh((xmin, ymin, xmax, ymax)))

            intrinsic_param = np.array(ann['cam_param']['intrinsic_param'], dtype=np.float32)

            focal_l = np.array([intrinsic_param[0, 0], intrinsic_param[1, 1]], dtype=np.float32)
            center = np.array([intrinsic_param[0, 2], intrinsic_param[1, 2]], dtype=np.float32)

            joint_cam = np.array(ann['keypoints_cam'])

            joint_img = cam2pixel_matrix(joint_cam, intrinsic_param)[:, :2]

            root_cam = joint_cam[4]

            cam_param = {'focal': focal_l, 'center': center, 'Rot': np.eye(3).astype(np.float32), 'transl': np.zeros((3,)).astype(np.float32)}

            # take h36m joints
            if self.data_split == 'test':
                joint_img = joint_img[self.hp3d_to_h36m_test]
                joint_cam = joint_cam[self.hp3d_to_h36m_test]
            else:
                joint_img = joint_img[self.hp3d_to_h36m]
                joint_cam = joint_cam[self.hp3d_to_h36m]

            # add nose joint
            joint_img[9] = (joint_img[8] + joint_img[10]) / 2
            joint_cam[9] = (joint_cam[8] + joint_cam[10]) / 2

            # check truncation
            underflow_x = (joint_img[:, 0] < 0).sum()
            underflow_y = (joint_img[:, 1] < 0).sum()
            overflow_x = (joint_img[:, 0] > height).sum()
            overflow_y = (joint_img[:, 1] > width).sum()
            if overflow_x > 5 or overflow_y > 5 or (overflow_x+overflow_y) > 8:
                continue
            if underflow_x > 5 or underflow_y > 5 or (underflow_x+underflow_y) > 8:
                continue

            datalist.append({
                'img_name': img_name,
                'img_hw': (height, width),
                'bbox': bbox,
                'joint_img': joint_img[:, :2],
                'joint_cam': joint_cam,
                'root_cam': joint_cam[:1],
                'smpl_param': {'pose': [0.]*72, 'shape': [0.]*10, 'trans': [0.]*3},
                'cam_param': cam_param,
            })

        return datalist


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

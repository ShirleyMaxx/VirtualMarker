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

from core.config import cfg, update_config, init_experiment_dir
from dataset.joints_dataset import JointsDataset

class Human36M(JointsDataset):
    bbox_3d_shape = (2000, 2000, 2000)
    def __init__(self, mode, args, transform=None, master=None):
        super().__init__(mode, args, 'Human36M', transform, master)
        self.block_list = ['s_09_act_05_subact_02_ca', 's_09_act_10_subact_02_ca', 's_09_act_13_subact_01_ca']

        self.datalist = self.lazy_load_data()

    def lazy_load_data(self):
        lazy_annot_path = osp.join(self.annot_path, 'Human36M_{}.pkl'.format(self.data_split))
        if osp.exists(lazy_annot_path):
            if self.master:
                print('Lazy load annotations of Human36M from ' + lazy_annot_path)
            with open(lazy_annot_path, 'rb') as f:
                datalist = pickle.load(f)
        else:
            datalist = self.load_data()
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

    def load_data(self):
        if self.master:
            print('Load annotations of Human36M')
        ann_file_path = osp.join(self.annot_path, f'Sample5_Human36M_{self.data_split}_smpl_full.json')
        with open(ann_file_path, 'r') as f:
            database = json.load(f)

        datalist = []
        for idx, datum in enumerate(database):
            img_name = datum['file_name']

            skip = False
            for block_name in self.block_list:
                if block_name in img_name:
                    skip = True
            if skip:
                continue

            image_id = datum['image_id']
            bbox = datum['bbox']   # x_min, y_min, w, h

            focal_l, center = np.array(datum['cam_param']['f'], dtype=np.float32), np.array(
                datum['cam_param']['c'], dtype=np.float32)
            Rot, transl = np.array(datum['cam_param']['R'], dtype=np.float32), np.array(
                datum['cam_param']['t'], dtype=np.float32)

            root_cam = np.array(datum['root_coord'])

            smpl_shape = np.array(datum['betas'])
            smpl_pose = np.array(datum['thetas']).reshape(24, 3)
            smpl_global_t = np.array(datum['trans'])
            smpl_global_t[:] = 0

            smpl_param = {'pose': smpl_pose, 'shape': smpl_shape, 'trans': smpl_global_t}
            cam_param = {'focal': focal_l, 'center': center, 'Rot': Rot, 'transl': transl}

            datalist.append({
                'img_name': img_name,
                'img_hw': (datum['height'], datum['width']),
                'bbox': bbox,
                'joint_img': None,  
                'joint_cam': None,  
                'root_cam': root_cam,  
                'smpl_param': smpl_param, 
                'cam_param': cam_param, 
            })

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

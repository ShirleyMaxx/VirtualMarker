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
from pycocotools.coco import COCO
from tqdm import tqdm

from core.config import cfg, update_config, init_experiment_dir
from utils.coord_utils import process_bbox
from dataset.joints_dataset import JointsDataset

class MSCOCO(JointsDataset):
    bbox_3d_shape = (2000, 2000, 2000)
    def __init__(self, mode, args, transform=None, master=None):
        super().__init__(mode, args, 'COCO', transform, master)
        assert mode == 'train', 'COCO only support train set.'

        self.datalist = self.lazy_load_data()

    def lazy_load_data(self):
        lazy_annot_path = osp.join(self.annot_path, 'COCO_{}.pkl'.format(self.data_split))
        if osp.exists(lazy_annot_path):
            if self.master:
                print('Lazy load annotations of COCO from ' + lazy_annot_path)
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
            print('Load annotations of COCO')
        db = COCO(osp.join(self.annot_path, 'person_keypoints_' + self.data_split + '2017.json'))
        with open(osp.join(self.annot_path, 'coco_smplifyx_train.json')) as f:
            smplify_results = json.load(f)

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            img = db.loadImgs(ann['image_id'])[0]
            img_name = osp.join('train2017', img['file_name'])
            img_path = img_name

            if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                continue

            # bbox
            bbox = process_bbox(ann['bbox'])
            if bbox is None: continue

            # joint coordinates
            joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1, 3)
            joint_valid = (joint_img[:, 2].copy().reshape(-1, 1) > 0).astype(np.float32)
            joint_img[:, 2] = 0

            if str(aid) in smplify_results:
                smplify_result = smplify_results[str(aid)]
            else:
                continue

            smplify_result['cam_param'].update({'R': np.eye(3).astype(np.float32), 't': np.zeros((3,)).astype(np.float32)})
            smpl_param = smplify_result['smpl_param'] 
            cam_param = smplify_result['cam_param'] 
            focal_l, center, Rot, transl = cam_param['focal'], cam_param['princpt'], cam_param['R'], cam_param['t']
            cam_param = {'focal': np.array(focal_l).astype(np.float32), 'center': np.array(center).astype(np.float32), 'Rot': np.array(Rot).astype(np.float32), 'transl': np.array(transl).astype(np.float32)}

            datalist.append({
                'img_name': img_name,
                'bbox': bbox,
                'root_cam': None,  
                'smpl_param': smpl_param, 
                'cam_param': cam_param, 
            })

        return datalist


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

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

class SURREAL(JointsDataset):
    bbox_3d_shape = (2000, 2000, 2000)
    def __init__(self, mode, args, transform=None, master=None):
        super().__init__(mode, args, 'SURREAL', transform, master)
        self.datalist = self.lazy_load_data()

    def lazy_load_data(self):
        lazy_annot_path = osp.join(self.annot_path, 'SURREAL_{}.pkl'.format(self.data_split))
        if osp.exists(lazy_annot_path):
            if self.master:
                print('Lazy load annotations of SURREAL from ' + lazy_annot_path)
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
            print('Load annotations of SURREAL ')
        db = COCO(osp.join(self.annot_path, self.data_split.replace('test', 'val') + '.json'))

        datalist = []
        for iid in db.imgs.keys():

            img_ann = db.imgs[iid]
            img_id = img_ann['id']
            img_name = osp.join(self.data_split, img_ann['file_name'])
            cam_param = {'focal': img_ann['cam_param']['focal'], 'center': img_ann['cam_param']['princpt'], 'Rot': np.eye(3).astype(np.float32), 'transl': np.zeros((3,)).astype(np.float32)}

            ann_id = db.getAnnIds(img_id)
            ann = db.loadAnns(ann_id)[0]
            smpl_param = ann['smpl_param']
            joint_cam = np.array(ann['joint_cam'], dtype=np.float32).reshape(-1,3)

            bbox = process_bbox(ann['bbox'])
            if bbox is None: continue

            datalist.append({
                'img_name': img_name,
                'img_hw': (img_ann['height'], img_ann['width']),
                'bbox': bbox,
                'root_cam': None,
                'smpl_param': smpl_param,
                'cam_param': cam_param,
            })
        datalist = sorted(datalist, key=lambda x: x['img_name'])

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
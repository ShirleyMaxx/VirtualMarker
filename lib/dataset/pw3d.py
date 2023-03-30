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

class PW3D(JointsDataset):
    bbox_3d_shape = (2000, 2000, 2000)
    def __init__(self, mode, args, transform=None, master=None):
        super().__init__(mode, args, 'PW3D', transform, master)
        self.datalist = self.lazy_load_data()


    def lazy_load_data(self):
        lazy_annot_path = osp.join(self.annot_path, '3DPW_{}.pkl'.format(self.data_split))
        if osp.exists(lazy_annot_path):
            if self.master:
                print('Lazy load annotations of 3DPW from ' + lazy_annot_path)
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
            print('Load annotations of 3DPW ')
        db = COCO(osp.join(self.annot_path, '3DPW_' + self.data_split + '.json'))

        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']

            img_ann = db.loadImgs(image_id)[0]
            img_name = osp.join(img_ann['sequence'], img_ann['file_name'])
            cam_param = {'focal': img_ann['cam_param']['focal'], 'center': img_ann['cam_param']['princpt'], 'Rot': np.eye(3).astype(np.float32), 'transl': np.zeros((3,)).astype(np.float32)}

            smpl_param = ann['smpl_param']
            bbox = process_bbox(np.array(ann['bbox']))
            if bbox is None: continue

            root_cam = np.array(ann['root_cam'], dtype=np.float32)

            datalist.append({
                'img_name': img_name,
                'img_hw': (img_ann['height'], img_ann['width']),
                'bbox': bbox,
                'annot_id': aid,
                'root_cam': root_cam,
                'smpl_param': smpl_param,
                'cam_param': cam_param,
            })

        datalist = sorted(datalist, key=lambda x: x['annot_id'])
        datalist = sorted(datalist, key=lambda x: x['img_name'])

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
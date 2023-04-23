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
from virtualmarker.dataset.joints_dataset import JointsDataset


class Up_3D(JointsDataset):
    bbox_3d_shape = (2000, 2000, 2000)
    def __init__(self, mode, args, transform=None, master=None):
        super().__init__(mode, args, 'Up_3D', transform, master)
    
        self.datalist = self.load_data()

    def load_data(self):
        lazy_load_path = osp.join(self.annot_path, 'UP3D_train.pkl')
        if self.master:
            print(f'Load annotations of Up_3D from {lazy_load_path}')
        with open(lazy_load_path, 'rb') as f:
            datalist = pickle.load(f)

        for item in datalist:
            item['img_name'] = osp.join(self.img_dir, item['img_name'])
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return super().__getitem__(idx)

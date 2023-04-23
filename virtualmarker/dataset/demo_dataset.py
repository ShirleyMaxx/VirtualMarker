import os.path as osp
import numpy as np
import scipy.sparse as ssp
import cv2
import pickle
from torch.utils.data import Dataset
from torchvision import transforms

from virtualmarker.core.config import cfg, update_config, init_experiment_dir
from virtualmarker.utils.coord_utils import _box_to_center_scale, get_intrinsic_metrix, pixel2cam, cam2pixel
from virtualmarker.utils.aug_utils import get_affine_transform, augm_params

def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5 # fov: 55 degree


class DemoDataset(Dataset):
    def __init__(self, img_path_list, detection_list):
        self.input_joint_name = cfg.dataset.input_joint_set  
        self.detection_list = detection_list
        self.img_path_list = img_path_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])

        self.human36_joint_num = 17
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.vertex_num = 6890
        self.vm_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        if self.input_joint_name == 'vm':
            # virtual marker joint set
            self.vm_A, self.vm_B = None, None
            if cfg.model.mesh2vm.vm_path != '':
                self.vm_A, self.vm_B = ssp.load_npz(osp.join(cfg.model.mesh2vm.vm_path, f'vm_A{cfg.model.mesh2vm.vm_type}.npz')).A.astype(float),\
                    ssp.load_npz(osp.join(cfg.model.mesh2vm.vm_path, f'vm_B{cfg.model.mesh2vm.vm_type}.npz')).A.astype(float)
                self.vm_info = np.load(osp.join(cfg.model.mesh2vm.vm_path, 'vm_info.npz'))
                self.vm_flip_pairs = tuple(self.vm_info['vm_flip_pairs'].tolist())
                self.vm_flip_pairs_reindex = tuple([(vm_pair[0]+self.human36_joint_num, vm_pair[1]+self.human36_joint_num) for vm_pair in self.vm_flip_pairs])
                cfg.dataset.num_joints = 17+self.vm_info['vm_K']

            self.vm_joint_num = cfg.dataset.num_joints
            if self.vm_B is not None:
                self.vm_joint_num = self.vm_B.shape[1]    
            elif cfg.model.mesh2vm.vm_path != '':
                self.vm_joint_num -= 17

            self.selected_indices = [i for i in range(6890)]
            with open(osp.join(cfg.data_dir, cfg.dataset.smpl_indices_path), 'rb') as f:
                smpl_indices = pickle.load(f)
            for body_part in smpl_indices.keys():
                body_part_indices = list(smpl_indices[body_part].numpy())
                if body_part in cfg.model.mesh2vm.ignore_part:
                    for idx in body_part_indices:
                        self.selected_indices.remove(idx)

            # Selected vertices set
            self.verts_joint_num = len(self.selected_indices)
        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)
        self.joint_regressor = np.load(osp.join(cfg.data_dir, 'smpl', 'J_regressor_h36m_correct.npy'))

    def get_joint_setting(self, joint_category='human36'):
        joint_num =  cfg.dataset.num_joints
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')
        if self.input_joint_name == 'vm' and joint_num % 16:
            flip_pairs = tuple(list(self.human36_flip_pairs) + list(self.vm_flip_pairs_reindex))

        return joint_num, skeleton, flip_pairs

    def __len__(self):
        return len(self.detection_list)

    def __getitem__(self, idx):
        """
        self.detection_list: [[frame_id, x_min, y_min, x_max, y_max, pixel_root_x, pixel_root_y, depth]]
        """
        det_info = self.detection_list[idx]
        img_idx = int(det_info[0])
        img_path = self.img_path_list[img_idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        ori_img_height, ori_img_width = img.shape[:2]
        focal = estimate_focal_length(ori_img_height, ori_img_width)
        focal_l = np.array([focal, focal])
        center_pt = np.array([ori_img_width/2 , ori_img_height/2])
        intrinsic_param = get_intrinsic_metrix(focal_l, center_pt, inv=True).astype(np.float32)

        bbox = det_info[1:5]
        # xmin, ymin, width, height
        bbox = (bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])

        # use the estimated root depth
        root_img = np.array([det_info[5], det_info[6], det_info[7]/1700*focal])
        root_cam = pixel2cam(root_img[None,:], center_pt, focal_l)

        center = (float(bbox[0] + 0.5*bbox[2]), float(bbox[1] + 0.5*bbox[3]))
        aspect_ratio = float(cfg.model.input_shape[1]) / cfg.model.input_shape[0]  # w / h
        scale_mult = 1
        center, scale = _box_to_center_scale(bbox, aspect_ratio, scale_mult=scale_mult)
        # aug
        _, _, scale = augm_params(is_train=0, scale=scale)

        trans, inv_trans = get_affine_transform(center, scale, 0, (cfg.model.input_shape[1], cfg.model.input_shape[0]))
        img = cv2.warpAffine(img, trans, (cfg.model.input_shape[1], cfg.model.input_shape[0]), flags=cv2.INTER_LINEAR)
        img = self.transform(img).float()

        meta_data = {
            'idx': idx,
            'img_idx': img_idx,
            'img': img,
            'inv_trans': inv_trans.astype(np.float32),
            'intrinsic_param': intrinsic_param.astype(np.float32),
            'root_cam': root_cam[0].astype(np.float32), 
            'depth_factor': np.array([2000]).astype(np.float32),
            'focal_l': np.array(focal_l).astype(np.float32),
            'center_pt': np.array(center_pt).astype(np.float32),
        }

        return meta_data
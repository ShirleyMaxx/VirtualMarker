import os.path as osp
import numpy as np
np.set_printoptions(suppress=True)
import math
import torch
import json
import copy
import scipy.sparse as ssp
import os
import cv2
import pickle
import random
from pycocotools.coco import COCO
import argparse
from tqdm import tqdm
from tqdm.contrib import tzip
import time
import smplx

import sys
from torchvision import transforms
from PIL import Image
import imageio
from termcolor import colored

from virtualmarker.core.config import cfg, update_config, init_experiment_dir

from virtualmarker.utils.coord_utils import cam2pixel, process_bbox, scale_bbox, _box_to_center_scale, _center_scale_to_box, rigid_align, get_bbox, get_intrinsic_metrix
from virtualmarker.utils.aug_utils import j2d_processing, augm_params, j3d_processing, juvd_processing, flip_joints_3d
from virtualmarker.utils.preprocessing import load_img, im_to_torch, generate_integral_uvd_target
from virtualmarker.utils.vis import vis_joints_3d, render_mesh
from virtualmarker.utils.funcs_utils import save_obj
from virtualmarker.utils.smpl_utils import SMPL


class JointsDataset(torch.utils.data.Dataset):
    def __init__(self, mode, args, dataset_name, transform=None, master=None):
        self.transform = transform
        self.data_split = mode
        self.master = master
        self.dataset_name = dataset_name
        self.img_dir = osp.join(cfg.data_dir, dataset_name, 'images')
        self.annot_path = osp.join(cfg.data_dir, dataset_name, 'annotations')
        self.input_joint_name = cfg.dataset.input_joint_set  

        # SMPL parametric model
        self.mesh_model_dict = {
            'neutral': SMPL(
                osp.join(cfg.data_dir, 'smpl'),
                batch_size=1,
                create_transl=False,
                gender = 'neutral'),
            'female': SMPL(
                osp.join(cfg.data_dir, 'smpl'),
                batch_size=1,
                create_transl=False,
                gender = 'female'),
            'male': SMPL(
                osp.join(cfg.data_dir, 'smpl'),
                batch_size=1,
                create_transl=False,
                gender = 'male'),
        }
        self.smpl_faces = self.mesh_model_dict['neutral'].faces.astype(np.int)
        self.smpl_template = self.mesh_model_dict['neutral'].v_template

        # H36M joint set
        self.human36_joint_num = 17
        self.human36_joints_name = (
        'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
        'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.human36_flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.human36_skeleton = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
        self.human36_root_joint_idx = self.human36_joints_name.index('Pelvis')
        self.human36_eval_joint = (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16)
        self.joint_regressor = self.mesh_model_dict['neutral'].J_regressor_h36m
        self.vertex_num = 6890

        self.datalist, skip_img_path = [], []

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
            self.vm_skeleton = self.human36_skeleton


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
            self.verts_skeleton = self.human36_skeleton

        self.joint_num, self.skeleton, self.flip_pairs = self.get_joint_setting(self.input_joint_name)
        self.error_distribution = self.get_stat() if dataset_name == 'AMASS' else None
        self.color_factor = cfg.aug.color_factor
        self.occlusion = cfg.aug.occlusion
        self.aspect_ratio = float(cfg.model.input_shape[1]) / cfg.model.input_shape[0]  # w / h
        self.scale_mult = 1.25 if self.dataset_name != 'Human36M' else 1

    def get_joint_setting(self, joint_category='human36'):
        joint_num =  cfg.dataset.num_joints
        skeleton = eval(f'self.{joint_category}_skeleton')
        flip_pairs = eval(f'self.{joint_category}_flip_pairs')
        if self.input_joint_name == 'vm' and joint_num != 17:
            flip_pairs = tuple(list(self.human36_flip_pairs) + list(self.vm_flip_pairs_reindex))

        return joint_num, skeleton, flip_pairs

    def occlusion_aug(self, bbox, img_shape):
        xmin, ymin, _, _ = bbox
        xmax = xmin + bbox[2]
        ymax = ymin + bbox[3]
        imght, imgwidth = img_shape
        counter = 0
        while True:
            # force to break if no suitable occlusion
            if counter > 5:
                return 0, 0, 0, 0
            counter += 1

            area_min = 0.0
            area_max = 0.3
            synth_area = (random.random() * (area_max - area_min) + area_min) * (xmax - xmin) * (ymax - ymin)

            ratio_min = 0.5
            ratio_max = 1 / 0.5
            synth_ratio = (random.random() * (ratio_max - ratio_min) + ratio_min)

            synth_h = math.sqrt(synth_area * synth_ratio)
            synth_w = math.sqrt(synth_area / synth_ratio)
            synth_xmin = random.random() * ((xmax - xmin) - synth_w - 1) + xmin
            synth_ymin = random.random() * ((ymax - ymin) - synth_h - 1) + ymin

            if synth_xmin >= 0 and synth_ymin >= 0 and synth_xmin + synth_w < imgwidth and synth_ymin + synth_h < imght:
                synth_xmin = int(synth_xmin)
                synth_ymin = int(synth_ymin)
                synth_w = int(synth_w)
                synth_h = int(synth_h)
                break
        return synth_ymin, synth_h, synth_xmin, synth_w

    def get_stat(self):
        try:
            pose_param = pickle.load(open(cfg.dataset.amass_noise_path, 'rb'))
        except:
            return None
        return pose_param

    def generate_syn_error(self):
        noise = np.random.normal(loc=0., scale=3.5, size=(self.vertex_num, 3))
        return noise

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, bbox, smpl_param, cam_param  = data['img_name'], data['bbox'].copy(), data['smpl_param'].copy(), data['cam_param'].copy()
        img = load_img(img_path)
        ori_img = img.copy()
        img_shape = img.shape[:2]

        smpl_shape, smpl_pose, smpl_global_t = smpl_param['shape'], smpl_param['pose'], smpl_param['trans']
        smpl_gender = smpl_param['gender'] if self.dataset_name == 'SURREAL' else 'neutral'
        focal_l, center_pt, offset = cam_param['focal'], cam_param['center'], None
        if self.dataset_name == 'Up_3D':
            offset = cam_param['offset']

        root_cam = data['root_cam']
        if self.dataset_name == '3DHP':
            mesh_cam, joint_cam = root_cam+np.zeros((self.vertex_num, 3), dtype=np.float32), data['joint_cam']
        else:
            smpl_pose = torch.FloatTensor(smpl_pose).view(1, -1)
            smpl_shape = torch.FloatTensor(smpl_shape).view(1, -1)
            smpl_global_t = torch.FloatTensor(smpl_global_t).view(1, -1)
            smpl_output = self.mesh_model_dict[smpl_gender](betas=smpl_shape, pose2rot=True,
                                        body_pose=smpl_pose[:, 3:],
                                        global_orient=smpl_pose[:, :3],
                                        transl=smpl_global_t)
            mesh_cam = smpl_output.vertices[0].numpy()
            if self.dataset_name == 'Up_3D':
                Rot, transl = np.array(cam_param['Rot'],dtype=np.float32).reshape(3, 3), np.array(cam_param['transl'],dtype=np.float32).reshape(3)
                mesh_cam = np.dot(Rot, np.transpose(mesh_cam, (1,0))).transpose(1,0) + transl.reshape(1, 3)
            mesh_cam *= 1000.0
            joint_cam = np.dot(self.joint_regressor, mesh_cam)
            if self.dataset_name == 'Human36M':
                offset = root_cam - joint_cam[:1]
                joint_cam += offset
                mesh_cam += offset
            root_cam = joint_cam[:1]

        # virtual marker
        if self.input_joint_name == 'vm' and self.vm_B is not None:
            joint_cam_vm = np.dot(self.vm_B.T, mesh_cam[self.selected_indices])
            joint_cam = np.concatenate((joint_cam, joint_cam_vm), axis=0)


        # get joint_img
        joint_img = cam2pixel(joint_cam, focal_l, center_pt)
        joint_img_mesh = cam2pixel(mesh_cam, focal_l, center_pt)
        if self.dataset_name == 'Up_3D':
            joint_img = joint_img / np.mean([center_pt[1]*2/bbox[3], center_pt[0]*2/bbox[2]]) + np.array([bbox[0], bbox[1], 0])
            joint_img[..., 1] += offset
            joint_img_mesh = joint_img_mesh / np.mean([center_pt[1]*2/bbox[3], center_pt[0]*2/bbox[2]]) + np.array([bbox[0], bbox[1], 0])
            joint_img_mesh[..., 1] += offset
        joint_img = joint_img[:, :2]

        # root-align
        mesh_cam = mesh_cam - root_cam
        joint_cam = joint_cam - root_cam

        joint_uvd = np.zeros((joint_img.shape[0], 3, 2), dtype=np.float32)
        joint_uvd[:, :2, 0] = joint_img.copy()
        joint_uvd[:, 2:, 0] = joint_cam[:, -1:].copy()

        # make new bbox
        if cfg.dataset.use_tight_bbox:
            if self.dataset_name == '3DHP':
                bbox = get_bbox(joint_img)
            else:
                bbox = get_bbox(joint_img_mesh)
            bbox = process_bbox(bbox)
        assert bbox is not None, 'check bbox from ' + img_path

        center, scale = _box_to_center_scale(bbox, self.aspect_ratio, scale_mult=self.scale_mult)
        bbox = _center_scale_to_box(center, scale)

        # aug
        flip, rot, scale = augm_params(is_train=(self.data_split == 'train'), scale=scale)

        joint_img_transformed, trans, inv_trans = j2d_processing(joint_img.copy(), img_shape[1], (cfg.model.input_shape[1], cfg.model.input_shape[0]),
                                        bbox, center, scale, rot, flip, self.flip_pairs)        # 0 - 256

        # joint valid
        joint_valid = np.ones((joint_img.shape[0], 1), dtype=np.float32)
        joint_cam_valid = np.ones((joint_img.shape[0], 3), dtype=np.float32)
        mesh_valid = np.ones((len(mesh_cam), 1), dtype=np.float32)
        
        if not cfg.dataset.use_coco3d_up3d:
            if self.dataset_name == 'COCO' or self.dataset_name == 'Up_3D':
                joint_cam_valid[:, 2] = 0
                mesh_valid[:] = 0

        if self.dataset_name == '3DHP':
            if self.input_joint_name == 'vm':
                joint_valid[self.human36_joint_num:], joint_cam_valid[self.human36_joint_num:] = 0, 0
            mesh_valid[:] = 0

        joint_uvd[:, :, 1] = joint_valid.copy()
        joint_uvd = juvd_processing(joint_uvd, img_shape[1], trans, flip, self.flip_pairs)   # + aug

        intrinsic_param = get_intrinsic_metrix(cam_param['focal'], cam_param['center'], inv=True)
        joint_uvd, joint_uvd_valid, joint_uvd_heatmap = generate_integral_uvd_target(joint_uvd, joint_uvd.shape[0], cfg.model.input_shape[1], cfg.model.input_shape[0], self.bbox_3d_shape)
        joint_uvd_valid *= joint_cam_valid

        if self.data_split == 'train':
            if self.occlusion and random.random() <= 0.8:
                synth_ymin, synth_h, synth_xmin, synth_w = self.occlusion_aug(bbox, img_shape)
                img[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin + synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255
        if flip:
            img = img[:, ::-1, :]

        img = cv2.warpAffine(img, trans, (cfg.model.input_shape[1], cfg.model.input_shape[0]), flags=cv2.INTER_LINEAR)

        if self.data_split == 'train':
            c_high = 1 + self.color_factor
            c_low = 1 - self.color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)
        img = im_to_torch(img)
        # mean
        img[0].add_(-0.406)
        img[1].add_(-0.457)
        img[2].add_(-0.480)

        # std
        img[0].div_(0.225)
        img[1].div_(0.224)
        img[2].div_(0.229)

        meta_data = {
            'img': img,
            'flip': np.array(flip).astype(np.bool),
            'focal_l': np.array(focal_l).astype(np.float32),
            'center_pt': np.array(center_pt).astype(np.float32),
            'trans': trans.astype(np.float32),
            'inv_trans': inv_trans.astype(np.float32),
            'root_cam': root_cam[0].astype(np.float32), 
            'joint_cam': joint_cam.astype(np.float32), 
            'joint_cam_valid': joint_cam_valid.astype(np.float32), 
            'mesh_cam': mesh_cam.astype(np.float32), 
            'mesh_valid': mesh_valid.astype(np.bool),
            'intrinsic_param': intrinsic_param.astype(np.float32),
            'depth_factor': np.array([self.bbox_3d_shape[2]]).astype(np.float32),
            'joint_uvd': joint_uvd,
            'joint_uvd_valid': joint_uvd_valid.reshape((-1)),
        }
        return meta_data



    def compute_joint_err(self, pred_pose, gt_pose):
        actual_joint_num = pred_pose.shape[1]
        pred_pose, gt_pose = pred_pose.detach().cpu().numpy(), gt_pose.detach().cpu().numpy()
        joint_mean_error = np.power((np.power((pred_pose - gt_pose), 2)).sum(axis=2), 0.5).mean()

        return joint_mean_error

    def compute_per_joint_err(self, pred_pose, gt_pose):
        # pred_pose: [N, J, 3]
        num_samples = pred_pose.shape[0]

        mpjpe = np.mean(np.sqrt(np.square(pred_pose - gt_pose).sum(axis=2)), axis=1)  # (N, )

        pred_pose_rigid_align = []
        for n in range(num_samples):
            pred_pose_rigid_align.append(rigid_align(pred_pose[n], gt_pose[n]))
        pred_pose_rigid_align = np.array(pred_pose_rigid_align)

        pa_mpjpe = np.mean(np.sqrt(np.square(pred_pose_rigid_align - gt_pose).sum(axis=2)), axis=1)  # (N, )

        return np.mean(mpjpe), np.mean(pa_mpjpe)


    def compute_per_joint_err_dict(self, pred_pose, gt_pose):
        batch_size = pred_pose.shape[0]
        actual_joint_num = pred_pose.shape[1]
        assert pred_pose.shape[1] == gt_pose.shape[1]
        pred_pose = pred_pose.reshape(batch_size, actual_joint_num, 3)
        gt_pose = gt_pose.reshape(batch_size, actual_joint_num, 3)

        pred_pose_align = pred_pose - pred_pose[:, :1, :]
        gt_pose_align = gt_pose - gt_pose[:, :1, :]
        mpjpe_14j, pa_mpjpe_14j = self.compute_per_joint_err(pred_pose_align[:, self.human36_eval_joint, :], gt_pose_align[:, self.human36_eval_joint, :])

        metric_dict = {
            'MPJPE_14J': mpjpe_14j,
            'PA_MPJPE_14J': pa_mpjpe_14j,
            }
        return metric_dict

    def compute_per_mesh_err_dict(self, pred_mesh, gt_mesh, pred_pose_reg, gt_pose_reg, gt_pose_root=None, focal_l=None, center_pt=None, dataset_name=None):
        pred_mesh_align = pred_mesh - pred_pose_reg[:, :1, :]
        gt_mesh_align = gt_mesh - gt_pose_reg[:, :1, :]
        mpvpe_align = np.mean(np.mean(np.sqrt(np.square(pred_mesh_align - gt_mesh_align).sum(axis=2)), axis=1))

        metric_dict = {
            'MPVPE': mpvpe_align,
            }

        if cfg.test.save_obj:
            print(colored(f"==> Save mesh, rendered image/video to {cfg.vis_dir}.", 'green'))
            print(colored(f"==> Warning! Will take long time [here](lib/dataset/joints_dataset.py/#L372) if visualizing all the data. Early break if needed.", 'red'))
            if 'PW3D' == dataset_name:
                redun_img_path_list = [datum['img_name'] for datum in self.datalist]
                img_path_list = list(set(redun_img_path_list))
                img_path_list.sort(key=redun_img_path_list.index)
                videowriter = None
                for img_path in img_path_list:
                    videoname = osp.join(cfg.vis_dir, f"{img_path.split('/')[-2]}_rendered.mp4")
                    if not osp.isfile(videoname):
                        try:
                            videowriter.close()
                        except:
                            pass
                        videowriter = imageio.get_writer(videoname, fps=25)
                    ori_img = load_img(img_path).astype(np.uint8)   # C*H*W -> H*W*C
                    ori_img_height, ori_img_width = ori_img.shape[:2]
                
                    chosen_mask = np.array(redun_img_path_list) == img_path
                    mesh_to_render = (pred_mesh[chosen_mask] + gt_pose_root[chosen_mask, None]) / 1000
                    rgb, depth = render_mesh(ori_img_height, ori_img_width, mesh_to_render, self.smpl_faces, {'focal': focal_l[chosen_mask], 'princpt': center_pt[chosen_mask]})
                    valid_mask = (depth > 0)[:,:,None] 
                    rendered_img = rgb * valid_mask + ori_img * (1-valid_mask)
                    videowriter.append_data(rendered_img.astype(np.uint8))
                videowriter.close()
            else:
                for pred, gt, datum, root_cam, focal, center in tzip(pred_mesh, gt_mesh, self.datalist, gt_pose_root, focal_l, center_pt):
                    filename = datum['img_name'].split('/')[-1][:-4]

                    save_obj(pred / 1000, self.smpl_faces, osp.join(cfg.vis_dir, f'{filename}_pred_{cfg.dataset.input_joint_set}{cfg.dataset.num_joints}.obj'))
                    save_obj(gt / 1000, self.smpl_faces, osp.join(cfg.vis_dir, f'{filename}_gt.obj'))

                    # render mesh
                    ori_img = load_img(datum['img_name']).astype(np.uint8)   # C*H*W -> H*W*C
                    mesh_to_render = (pred + root_cam) / 1000
                    print('root_cam   ', root_cam)
                    rgb, depth = render_mesh(ori_img.shape[0], ori_img.shape[1], [mesh_to_render], self.smpl_faces, {'focal':  [focal], 'princpt': [center]})
                    valid_mask = (depth > 0)[:,:,None] 
                    rendered_img = rgb * valid_mask + ori_img[:,:,::-1] * (1-valid_mask)
                    render_path = osp.join(cfg.vis_dir, f'{filename}_rendered.png')
                    cv2.imwrite(render_path, np.uint8(rendered_img))
        return metric_dict
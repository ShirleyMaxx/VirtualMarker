from os import path as osp
import numpy as np
np.set_printoptions(suppress=True)
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from core.config import cfg
from models import simple3dmesh, noise_reduction
from utils.mesh import Mesh
from utils.funcs_utils import load_checkpoint


class Simple3DMesh_Post(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, 
                    mesh_num_joints=None, 
                    flip_pairs=None,
                    vm_A=None,
                    selected_indices=None,
                    mesh_template=None):
        super(Simple3DMesh_Post, self).__init__()

        self.simple3dmesh = simple3dmesh.get_model(mesh_num_joints=mesh_num_joints, flip_pairs=flip_pairs, vm_A=vm_A, selected_indices=selected_indices)
        if cfg.model.simple3dmesh.fix_network:
            for param in self.simple3dmesh.parameters():
                param.requires_grad = False

        self.mesh = Mesh(mesh_template)

        self.noise_reduction = noise_reduction.get_model(int(mesh_num_joints/4+0.5))
        if osp.isfile(cfg.model.simple3dmesh.noise_reduce_pretrained):
            state_dict = load_checkpoint(cfg.model.simple3dmesh.noise_reduce_pretrained)['model_state_dict']
            for key in list(state_dict.keys()):
                if 'noise_reduction.' in key:
                    state_dict[key.replace('noise_reduction.', '')] = state_dict.pop(key)
            self.noise_reduction.load_state_dict(state_dict, strict=True)
            print('Successfully load noise_reduction checkpoint from {}.'.format(cfg.model.simple3dmesh.noise_reduce_pretrained))

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False, flip_mask=None, is_train=True, gt_param_pose=None, gt_param_shape=None, gt_param_trans=None, is_up3d=None, cam_param=None):
        """Forward pass
        Inputs:
            x: image, size = (B, 3, 224, 224)
        Returns:
            pred_xyz_jts: camera 3d pose (joints + archetypes), size = (B, J+K, 3)
            confidence: confidence score for each body point in 3d pose, size = (B, J+K), for loss_{conf}
            pred_uvd_jts_flat: uvd 3d pose (joints + archetypes), size = (B, (J+K)*3), for loss_{pose}
            mesh3d: non-parametric 3d coordinates of mesh vertices, size = (B, V, 3), for loss_{mesh}
            mesh3d_post: non-parametric refined 3d coordinates of mesh vertices, size = (B, V, 3), for vis
        """
        batch_size = x.shape[0]

        # 3D pose & mesh, confidence
        pred_xyz_jts, pred_uvd_jts_flat, adaptive_A, confidence, mesh3d, _ = self.simple3dmesh(x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item, flip_output, flip_mask)       # (B, J+K, 3), (B, J+K)
        mesh3d_ret = mesh3d.clone()

        # prepare input for noise reduction
        # the input is the predicted mesh (subsampled by a factor of 4)
        # notice that we detach the mesh
        mesh3d = mesh3d.detach()
        # mesh3d downsample
        mesh3d = self.mesh.downsample(mesh3d)

        mesh3d_post = self.noise_reduction(mesh3d)
        # mesh3d upsample
        mesh3d_post = self.mesh.upsample(mesh3d_post)
        return pred_xyz_jts, pred_uvd_jts_flat, adaptive_A, confidence, mesh3d_ret, mesh3d_post


def get_model(mesh_num_joints, flip_pairs, vm_A=None, selected_indices=None, mesh_template=None):
    model = Simple3DMesh_Post(mesh_num_joints=mesh_num_joints, flip_pairs=flip_pairs, vm_A=vm_A, selected_indices=selected_indices, mesh_template=mesh_template)
    return model
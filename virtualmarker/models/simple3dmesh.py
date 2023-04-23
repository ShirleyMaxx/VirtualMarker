from os import path as osp
import numpy as np
np.set_printoptions(suppress=True)
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from virtualmarker.core.config import cfg
from virtualmarker.models import simple3dpose


class Simple3DMesh(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, 
                    mesh_num_joints=None, 
                    flip_pairs=None,
                    vm_A=None,
                    selected_indices=None):
        super(Simple3DMesh, self).__init__()

        self.joint_num = cfg.dataset.num_joints

        self.simple3dpose = simple3dpose.get_model(flip_pairs=flip_pairs)

        self.adaptive_A = nn.Sequential(
            nn.Linear(self.joint_num, mesh_num_joints*self.joint_num, bias = True),
            )

        if osp.isfile(cfg.model.simple3dmesh.pretrained):
            pretrained = cfg.model.simple3dmesh.pretrained
            print('==> try loading pretrained Image2Mesh model {}'.format(pretrained))
            pretrained_weight_dict = torch.load(pretrained)
            if 'model_state_dict' in pretrained_weight_dict:
                pretrained_state_dict = pretrained_weight_dict['model_state_dict']
            else:
                pretrained_state_dict = pretrained_weight_dict
            for key in list(pretrained_state_dict.keys()):
                if 'simple3dpose' not in key and 'adaptive_A' not in key:
                    pretrained_state_dict['simple3dpose.' + key] = pretrained_state_dict.pop(key)
            try:
                self.load_state_dict(pretrained_state_dict, strict=True)
                print('Successfully load pretrained simple3dmesh model.')
            except:
                try:
                    self.load_state_dict(pretrained_state_dict, strict=False)
                    print('Load part of pretrained simple3dmesh model {} (strict=False)'.format(pretrained))
                except:
                    print('Failed load pretrained simple3dmesh model {}'.format(pretrained))

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False, flip_mask=None, is_train=True):
        """Forward pass
        Inputs:
            x: image, size = (B, 3, 224, 224)
        Returns:
            pred_xyz_jts: camera 3d pose (joints + virtual markers), size = (B, J+K, 3)
            confidence: confidence score for each body point in 3d pose, size = (B, J+K), for loss_{conf}
            pred_uvd_jts_flat: uvd 3d pose (joints + virtual markers), size = (B, (J+K)*3), for loss_{pose}
            mesh3d: non-parametric 3d coordinates of mesh vertices, size = (B, V, 3), for loss_{mesh}
        """
        batch_size = x.shape[0]
        # 3D pose estimation, get confidence from 3D heatmaps
        pred_xyz_jts, confidence, pred_uvd_jts_flat, pred_root_xy_img = self.simple3dpose(x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item, flip_output, flip_mask)       # (B, J+K, 3), (B, J+K)

        confidence_ret = confidence.clone()
        pred_xyz_jts_ret = pred_xyz_jts.clone()
    
        # detach pose3d to mesh for faster convergence
        pred_xyz_jts = pred_xyz_jts.detach()
        confidence = confidence.detach()

        # get adaptive_A based on the estimation confidence 
        adaptive_A = self.adaptive_A(confidence.view(confidence.shape[0], -1))
        adaptive_A = adaptive_A.view(adaptive_A.size(0), -1, self.joint_num)   # B, V, J+K

        # get mesh by production of 3D pose & reconstruction matrix A
        mesh3d = torch.matmul(adaptive_A, pred_xyz_jts)     # B, V, 3

        return pred_xyz_jts_ret, pred_uvd_jts_flat, adaptive_A, confidence_ret, mesh3d, None, pred_root_xy_img


def get_model(mesh_num_joints, flip_pairs, vm_A=None, selected_indices=None):
    model = Simple3DMesh(mesh_num_joints=mesh_num_joints, flip_pairs=flip_pairs, vm_A=vm_A, selected_indices=selected_indices)
    return model
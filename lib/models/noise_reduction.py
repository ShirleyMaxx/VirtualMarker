"""
Adapted from https://github.com/nkolot/GraphCMR/blob/master/models/smpl_param_regressor.py
"""
from __future__ import division
import torch
import torch.nn as nn
from .layers.MLP import FCBlock, FCResBlock
from core.config import cfg
import time
from os import path as osp

class NoiseReduction(nn.Module):

    def __init__(self, hidden_dim=1024, mesh_num_joints=None):
        super(NoiseReduction, self).__init__()
        # mesh_num_joints is the number of vertices in the SMPL mesh
        ndim = 3
        self.layers = nn.Sequential(FCBlock(mesh_num_joints * ndim, hidden_dim),
                                    FCResBlock(hidden_dim, hidden_dim),
                                    FCResBlock(hidden_dim, hidden_dim),
                                    nn.Linear(hidden_dim, mesh_num_joints * 3))

        if osp.isfile(cfg.model.simple3dmesh.noise_reduce_pretrained):
            pretrained = cfg.model.simple3dmesh.noise_reduce_pretrained
            print('==> try loading pretrained noise_reduction model {}'.format(pretrained))
            pretrained_weight_dict = torch.load(pretrained)
            if 'model_state_dict' in pretrained_weight_dict:
                pretrained_state_dict = pretrained_weight_dict['model_state_dict']
            else:
                pretrained_state_dict = pretrained_weight_dict
            for key in list(pretrained_state_dict.keys()):
                if 'noise_reduction.' in key:
                    pretrained_state_dict[key.replace('noise_reduction.', '')] = pretrained_state_dict.pop(key)
            try:
                self.load_state_dict(pretrained_state_dict, strict=True)
                print('Successfully load pretrained noise_reduction model.')
            except:
                try:
                    self.load_state_dict(pretrained_state_dict, strict=False)
                    print('Load part of pretrained noise_reduction model {} (strict=False)'.format(pretrained))
                except:
                    print('Failed load pretrained noise_reduction model {}'.format(pretrained))


    def forward(self, x):
        """Forward pass.
        Input:
            x: size = (B, V*6)
        Returns:
            mesh: size = (B, V*3)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layers(x)
        return x.view(batch_size, -1, 3)


def get_model(mesh_num_joints):
    model = NoiseReduction(mesh_num_joints=mesh_num_joints)
    return model
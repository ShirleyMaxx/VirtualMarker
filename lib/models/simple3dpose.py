from os import path as osp
import numpy as np
np.set_printoptions(suppress=True)
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers.HRnet import HRNet

from core.config import cfg, update_config


def norm_heatmap(norm_type, heatmap):
    # Input tensor shape: [B,C,...]
    shape = heatmap.shape
    if norm_type == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)*cfg.model.simple3dpose.alpha
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError

class Simple3DPose(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, 
                    flip_pairs=None):
        super(Simple3DPose, self).__init__()
        self.deconv_dim = cfg.model.simple3dpose.num_deconv_filters
        self._norm_layer = norm_layer
        self.joint_num = cfg.dataset.num_joints
        self.actual_joint_num = self.joint_num
        self.norm_type = cfg.model.simple3dpose.extra_norm_type
        self.depth_dim = cfg.model.simple3dpose.extra_depth_dim
        self.height_dim = cfg.model.heatmap_shape[0]
        self.width_dim = cfg.model.heatmap_shape[1]

        self.flip_pairs_left = [pair[0] for pair in flip_pairs] if flip_pairs is not None else None
        self.flip_pairs_right = [pair[1] for pair in flip_pairs] if flip_pairs is not None else None

        backbone = HRNet
        self.preact = backbone()

        self.root_idx = 0


    def _make_deconv_layer(self):
        deconv_layers = []
        if self.height_dim == 80:
            deconv1 = nn.ConvTranspose2d(
                self.feature_channel, self.deconv_dim[0], kernel_size=7, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn1 = self._norm_layer(self.deconv_dim[0])
            deconv2 = nn.ConvTranspose2d(
                self.deconv_dim[0], self.deconv_dim[1], kernel_size=6, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn2 = self._norm_layer(self.deconv_dim[1])
            deconv3 = nn.ConvTranspose2d(
                self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn3 = self._norm_layer(self.deconv_dim[2])
        else:
            deconv1 = nn.ConvTranspose2d(
                self.feature_channel, self.deconv_dim[0], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn1 = self._norm_layer(self.deconv_dim[0])
            deconv2 = nn.ConvTranspose2d(
                self.deconv_dim[0], self.deconv_dim[1], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn2 = self._norm_layer(self.deconv_dim[1])
            deconv3 = nn.ConvTranspose2d(
                self.deconv_dim[1], self.deconv_dim[2], kernel_size=4, stride=2, padding=int(4 / 2) - 1, bias=False)
            bn3 = self._norm_layer(self.deconv_dim[2])

        deconv_layers.append(deconv1)
        deconv_layers.append(bn1)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv2)
        deconv_layers.append(bn2)
        deconv_layers.append(nn.ReLU(inplace=True))
        deconv_layers.append(deconv3)
        deconv_layers.append(bn3)
        deconv_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*deconv_layers)

    def _initialize(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def uvd_to_cam(self, uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor, return_relative=True):
        assert uvd_jts.dim() == 3 and uvd_jts.shape[2] == 3, uvd_jts.shape
        uvd_jts_new = uvd_jts.clone()
        assert torch.sum(torch.isnan(uvd_jts)) == 0, ('uvd_jts', uvd_jts)

        # remap uv coordinate to input space
        uvd_jts_new[:, :, 0] = (uvd_jts[:, :, 0] + 0.5) * cfg.model.input_shape[1]
        uvd_jts_new[:, :, 1] = (uvd_jts[:, :, 1] + 0.5) * cfg.model.input_shape[0]
        # remap d to mm
        uvd_jts_new[:, :, 2] = uvd_jts[:, :, 2] * depth_factor
        assert torch.sum(torch.isnan(uvd_jts_new)) == 0, ('uvd_jts_new', uvd_jts_new)

        dz = uvd_jts_new[:, :, 2]

        # transform in-bbox coordinate to image coordinate
        uv_homo_jts = torch.cat(
            (uvd_jts_new[:, :, :2], torch.ones_like(uvd_jts_new)[:, :, 2:]),
            dim=2)
        # batch-wise matrix multipy : (B,1,2,3) * (B,K,3,1) -> (B,K,2,1)
        uv_jts = torch.matmul(trans_inv.unsqueeze(1), uv_homo_jts.unsqueeze(-1))
        # transform (u,v,1) to (x,y,z)
        cam_2d_homo = torch.cat(
            (uv_jts, torch.ones_like(uv_jts)[:, :, :1, :]),
            dim=2)
        # batch-wise matrix multipy : (B,1,3,3) * (B,K,3,1) -> (B,K,3,1)
        xyz_jts = torch.matmul(intrinsic_param.unsqueeze(1), cam_2d_homo)
        xyz_jts = xyz_jts.squeeze(dim=3)
        # recover absolute z : (B,K) + (B,1)
        abs_z = dz + joint_root[:, 2].unsqueeze(-1)
        # multipy absolute z : (B,K,3) * (B,K,1)
        xyz_jts = xyz_jts * abs_z.unsqueeze(-1)

        if return_relative:
            # (B,K,3) - (B,1,3)
            xyz_jts = xyz_jts - joint_root.unsqueeze(1)

        # xyz_jts = xyz_jts / depth_factor.unsqueeze(-1)

        return xyz_jts

    def flip_uvd_coord(self, pred_jts, shift=False, flatten=True, flip_mask=None):
        if flatten:
            assert pred_jts.dim() == 2
            num_batches = pred_jts.shape[0]
            pred_jts = pred_jts.reshape(num_batches, self.actual_joint_num, 3)
        else:
            assert pred_jts.dim() == 3
            num_batches = pred_jts.shape[0]

        if flip_mask is None:
            flip_mask = torch.ones(num_batches).bool()
        # none of them needs flip
        if flip_mask.sum() == 0:
            return pred_jts
        flip_mask = flip_mask.cuda()

        # flip
        if shift:
            pred_jts[flip_mask, :, 0] = - pred_jts[flip_mask, :, 0]
        else:
            pred_jts[flip_mask, :, 0] = -1 / self.width_dim - pred_jts[flip_mask, :, 0]

        # flip_pair
        pred_jts_flip = pred_jts[flip_mask].clone()

        pred_jts_flip[:, self.flip_pairs_left], pred_jts_flip[:, self.flip_pairs_right] = \
            pred_jts_flip[:, self.flip_pairs_right].clone(), pred_jts_flip[:, self.flip_pairs_left].clone()

        pred_jts[flip_mask] = pred_jts_flip

        if flatten:
            pred_jts = pred_jts.reshape(num_batches, self.actual_joint_num * 3)

        return pred_jts

    def flip_confidence(self, confidence, flip_mask=None):
        num_batches = confidence.shape[0]
        if flip_mask is None:
            flip_mask = torch.ones(num_batches).bool()
        # none of them needs flip
        if flip_mask.sum() == 0:
            return confidence
        flip_mask = flip_mask.cuda()

        # flip_pair
        confidence_flip = confidence[flip_mask].clone()

        confidence_flip[:, self.flip_pairs_left], confidence_flip[:, self.flip_pairs_right] = \
            confidence_flip[:, self.flip_pairs_right].clone(), confidence_flip[:, self.flip_pairs_left].clone()

        confidence[flip_mask] = confidence_flip

        return confidence

    def forward(self, x, trans_inv, intrinsic_param, joint_root, depth_factor, flip_item=None, flip_output=False, flip_mask=None):
        """Forward pass
        Inputs:
            x: image, size = (B, 3, 224, 224)
        Returns:
            pred_xyz_jts: camera 3d pose (joints + archetypes), size = (B, J+K, 3)
            confidence: confidence score for each body point in 3d pose, size = (B, J+K), for loss_{conf}
            pred_uvd_jts_flat: uvd 3d pose (joints + archetypes), size = (B, (J+K)*3), for loss_{pose}
        """
        batch_size = x.shape[0]

        x0 = self.preact(x)     # (b, 512, 8, 8)
        out = x0.reshape((x0.shape[0], self.actual_joint_num, -1))
        out = norm_heatmap(self.norm_type, out)
        assert out.dim() == 3, out.shape


        if self.norm_type == 'sigmoid':
            maxvals, _ = torch.max(out, dim=2, keepdim=True)
        else:
            maxvals = torch.ones((*out.shape[:2], 1), dtype=torch.float, device=out.device)

        heatmaps = out / out.sum(dim=2, keepdim=True)

        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.actual_joint_num, self.depth_dim, self.height_dim, self.width_dim))      # B, J+K, D, H, W

        hm_x = heatmaps.sum((2, 3))
        hm_y = heatmaps.sum((2, 4))
        hm_z = heatmaps.sum((3, 4))

        device = torch.device('cuda')

        hm_x = hm_x * torch.arange(float(hm_x.shape[-1])).to(device)
        hm_y = hm_y * torch.arange(float(hm_y.shape[-1])).to(device)
        hm_z = hm_z * torch.arange(float(hm_z.shape[-1])).to(device)

        coord_x = hm_x.sum(dim=2, keepdim=True)
        coord_y = hm_y.sum(dim=2, keepdim=True)
        coord_z = hm_z.sum(dim=2, keepdim=True)

        pred_uvd_jts_coord = torch.cat((coord_x, coord_y, coord_z), dim=2).clone()     # B, J+K, 3

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        #  -0.5 ~ 0.5
        pred_uvd_jts = torch.cat((coord_x, coord_y, coord_z), dim=2)    # B, J+K, 3

        # NOTE that heatmap is (z, y, x) pred_uvd_jts is (x, y, z)
        pred_uvd_jts_ind = (pred_uvd_jts_coord[...,2]*self.depth_dim*self.height_dim + pred_uvd_jts_coord[...,1]*self.height_dim + pred_uvd_jts_coord[...,0]).unsqueeze(2).long()
        confidence = torch.gather(heatmaps.view(*heatmaps.shape[:2], -1), 2, pred_uvd_jts_ind).squeeze(-1)      # B, J+K

        if flip_item is not None:
            assert flip_output
            pred_uvd_jts_orig, confidence_orig = flip_item

            pred_uvd_jts = self.flip_uvd_coord(pred_uvd_jts, flatten=False, shift=True)
            confidence = self.flip_confidence(confidence)

            pred_uvd_jts = (pred_uvd_jts + pred_uvd_jts_orig.reshape(batch_size, self.actual_joint_num, 3)) / 2
            confidence = (confidence + confidence_orig) / 2

        pred_uvd_jts_flat = pred_uvd_jts.reshape((batch_size, self.actual_joint_num * 3)).clone()

        # use flip_mask to flip back thosed flipped
        if flip_mask is not None:
            pred_uvd_jts = self.flip_uvd_coord(pred_uvd_jts, flatten=False, shift=True, flip_mask=flip_mask)
            confidence = self.flip_confidence(confidence, flip_mask=flip_mask)

        #  -0.5 ~ 0.5
        # Rotate back
        pred_xyz_jts = self.uvd_to_cam(pred_uvd_jts, trans_inv, intrinsic_param, joint_root, depth_factor)
        assert torch.sum(torch.isnan(pred_xyz_jts)) == 0, ('pred_xyz_jts', pred_xyz_jts)

        pred_xyz_jts = pred_xyz_jts - pred_xyz_jts[:, self.root_idx, :].unsqueeze(1)    # B, J+K, 3
        
        # pred_xyz_jts = pred_xyz_jts.reshape((batch_size, -1))
        return pred_xyz_jts, confidence, pred_uvd_jts_flat


def get_model(flip_pairs):
    model = Simple3DPose(flip_pairs=flip_pairs)
    return model
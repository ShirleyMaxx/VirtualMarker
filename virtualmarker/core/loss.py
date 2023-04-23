import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from virtualmarker.core.config import cfg

class ConfCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, confidence, joint_valid):
        # confidence N, J
        # joint_valid N, J*3
        batch_size = confidence.shape[0]
        joint_valid = joint_valid.view(batch_size, -1, 3)[:, :, 2]  # N, J

        loss = (joint_valid * (-torch.log(confidence + 1e-6))).mean()

        return loss

class CoordLoss(nn.Module):
    def __init__(self, has_valid=False, reduction='mean'):
        super(CoordLoss, self).__init__()

        self.has_valid = has_valid
        self.criterion = nn.L1Loss(reduction=reduction).cuda()

    def forward(self, pred, target, target_valid=None):
        if self.has_valid:
            pred, target = pred * target_valid, target * target_valid

        loss = self.criterion(pred, target)

        return loss

class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()

class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()


def get_mesh_loss(faces=None):
    loss = CoordLoss(has_valid=True), NormalVectorLoss(faces), EdgeLengthLoss(faces), CoordLoss(has_valid=True), CoordLoss(has_valid=True)
    return loss

def get_loss(faces=None):
    # define loss function (criterion) and optimizer
    criterion_3d = CoordLoss(has_valid=True).cuda()
    criterion_3d_reg = CoordLoss(has_valid=True).cuda()
    criterion_conf = ConfCELoss()
    criterion_mesh = get_mesh_loss(faces)
    criterion_dict = {
        'joint_3d': criterion_3d,
        'joint_reg_3d': criterion_3d_reg,
        'conf': criterion_conf,
        'mesh_3d': criterion_mesh[0],
        'normal': criterion_mesh[1],
        'edge': criterion_mesh[2],
    }
    return criterion_dict

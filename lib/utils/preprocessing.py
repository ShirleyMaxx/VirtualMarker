import numpy as np
import cv2
import random
from core.config import cfg 
import math
import torch
import torch.nn.functional as F

def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:,:,::-1]
    # if not isinstance(img, np.ndarray):
    #     raise IOError("Fail to read %s" % path)

    return img

def im_to_torch(img):
    """Transform ndarray image to torch tensor.

    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: `(H, W, 3)`.

    Returns
    -------
    torch.Tensor
        A tensor with shape: `(3, H, W)`.

    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255
    return img

def generate_integral_uvd_target(joints_3d, num_joints, patch_height, patch_width, bbox_3d_shape):

    target_weight = np.ones((num_joints, 3), dtype=np.float32)
    target_weight[:, 0] = joints_3d[:, 0, 1]
    target_weight[:, 1] = joints_3d[:, 0, 1]
    target_weight[:, 2] = joints_3d[:, 0, 1]

    target = np.zeros((num_joints, 3), dtype=np.float32)
    target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
    target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
    target[:, 2] = joints_3d[:, 2, 0] / bbox_3d_shape[2]

    # for heatmap 3d
    target3d = np.zeros((num_joints, 3), dtype=np.float32)
    target3d[:, 2] = (joints_3d[:, 0, 0] / patch_width - 0.5)*cfg.model.heatmap_shape[0]
    target3d[:, 1] = (joints_3d[:, 1, 0] / patch_height - 0.5)*cfg.model.heatmap_shape[1]
    target3d[:, 0] = (joints_3d[:, 2, 0] / bbox_3d_shape[2])*cfg.model.simple3dpose.extra_depth_dim


    target_weight[target[:, 0] > 0.5] = 0
    target_weight[target[:, 0] < -0.5] = 0
    target_weight[target[:, 1] > 0.5] = 0
    target_weight[target[:, 1] < -0.5] = 0
    target_weight[target[:, 2] > 0.5] = 0
    target_weight[target[:, 2] < -0.5] = 0

    target = target.reshape((-1))
    # target_weight = target_weight.reshape((-1))
    return target, target_weight, target3d

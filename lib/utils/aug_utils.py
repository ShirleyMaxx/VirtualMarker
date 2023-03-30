import numpy as np
import cv2
import torch
import random

from core.config import cfg


def flip_img(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))

def flip_joints_2d(joints_2d, width, flip_pairs):
    """Flip 2d joints.

        Parameters
        ----------
        joints_3d : numpy.ndarray
            Joints in shape (num_joints, 3)
        flip_pairs : list
            List of joint pairs.

        Returns
        -------
        numpy.ndarray
            Flipped 3d joints with shape (num_joints, 3)

    """
    joints = joints_2d.copy()
    # flip horizontally
    joints[:, 0] = width - joints[:, 0] - 1
    # change left-right parts
    if flip_pairs is not None:
        for lr in flip_pairs:
            joints[lr[0]], joints[lr[1]] = joints[lr[1]].copy(), joints[lr[0]].copy()

    return joints

def flip_xyz_joints_3d(joints_3d, flip_pairs):
    """Flip 3d xyz joints.

        Parameters
        ----------
        joints_3d : numpy.ndarray
            Joints in shape (num_joints, 3)
        flip_pairs : list
            List of joint pairs.

        Returns
        -------
        numpy.ndarray
            Flipped 3d joints with shape (num_joints, 3)

    """
    assert joints_3d.ndim in (2, 3)

    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0] = -1 * joints[:, 0]
    # change left-right parts
    if flip_pairs is not None:
        for pair in flip_pairs:
            joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints

def flip_joints_3d(joints_3d, width, flip_pairs):
    """Flip 3d joints.

        Parameters
        ----------
        joints_3d : numpy.ndarray
            Joints in shape (num_joints, 3, 2)
        width : int
            Image width.
        flip_pairs : list
            List of joint pairs.

        Returns
        -------
        numpy.ndarray
            Flipped 3d joints with shape (num_joints, 3, 2)

    """
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    if flip_pairs is not None:
        for pair in flip_pairs:
            joints[pair[0], :, 0], joints[pair[1], :, 0] = \
                joints[pair[1], :, 0], joints[pair[0], :, 0].copy()
            joints[pair[0], :, 1], joints[pair[1], :, 1] = \
                joints[pair[1], :, 1], joints[pair[0], :, 1].copy()

    joints[:, :, 0] *= joints[:, :, 1]
    return joints

def j2d_processing(kp, width, res, bbox, center, scale, rot, f, flip_pairs):
    """Process gt 2D keypoints and apply all augmentation transforms."""
    # flip the x coordinates
    if f:
        kp = flip_joints_2d(kp, width, flip_pairs)
        center[0] = width - center[0] - 1
    trans, inv_trans = get_affine_transform(center, scale, rot, res)

    nparts = kp.shape[0]
    # print(kp)
    for i in range(nparts):
        kp[i, :2] = affine_transform(kp[i, :2].copy(), trans)
    # print('after     ', kp)
    # assert 0
    kp = kp.astype('float32')
    return kp, trans, inv_trans

def j3d_processing(S, r, f, flip_pairs):
    """Process gt 3D keypoints and apply all augmentation transforms."""
    # in-plane rotation
    # flip the x coordinates
    if f:
        S = flip_xyz_joints_3d(S, flip_pairs)

    # rot_mat = np.eye(3)
    # if not r == 0:
    #     rot_rad = -r * np.pi / 180
    #     sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    #     rot_mat[0, :2] = [cs, -sn]
    #     rot_mat[1, :2] = [sn, cs]
    # S = np.einsum('ij,kj->ki', rot_mat, S)


    S = S.astype('float32')

    return S

def juvd_processing(S, width, trans, f, flip_pairs):
    """Process gt 3D keypoints and apply all augmentation transforms."""
    # in-plane rotation
    # flip the x coordinates
    if f:
        S = flip_joints_3d(S, width, flip_pairs)

    for i in range(S.shape[0]):
        if S[i, 0, 1] > 0.0:
            S[i, 0:2, 0] = affine_transform(S[i, 0:2, 0], trans)


    S = S.astype('float32')

    return S

def augm_params(is_train, scale):
    if not is_train:
        return 0, 0, scale

    """Get augmentation parameters."""
    flip = 0  # flipping
    rot = 0  # rotation
    # We flip with probability 1/2
    if cfg.aug.flip and random.uniform(0, 1) <= 0.5:
        flip = 1

    # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
    # scale = np.clip(np.random.randn(), -1.0, 1.0) * cfg.aug.scale_factor + 1.0
    # rot = np.clip(np.random.randn(), -2.0,
    #             2.0) * cfg.aug.rotate_factor if random.random() <= 0.6 else 0

    sf = cfg.aug.scale_factor
    scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

    rf = cfg.aug.rotate_factor
    rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

    return flip, rot, scale

def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32)):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, inv_trans

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

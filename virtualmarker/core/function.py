import os.path as osp
import numpy as np
import cv2, os
import math
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from collections import defaultdict
import pickle
from itertools import islice
import json
import time
import scipy.sparse as ssp
import matplotlib.pyplot as plt

from virtualmarker.core.base import prepare_network
from virtualmarker.core.config import cfg
from virtualmarker.utils.funcs_utils import lr_check, save_obj
from virtualmarker.utils.vis import vis_joints_3d, render_mesh, denormalize_image
from virtualmarker.utils.aug_utils import get_affine_transform, flip_img, augm_params


class Simple3DMeshTrainer:
    def __init__(self, args, load_path, writer=None, master=None):
        self.batch_generator, self.dataset_list, self.model, self.loss, self.optimizer, self.lr_scheduler, self.loss_history, self.error_history,\
            self.dataset, self.sampler = prepare_network(args, load_path=load_path, is_train=True, master=master)

        self.main_dataset = self.dataset_list[0]
        self.joint_num = cfg.dataset.num_joints
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.train.vis_freq
        self.writer = writer
        self.device = args.device

        self.J_regressor = torch.Tensor(self.main_dataset.joint_regressor).cuda()
        self.selected_indices = self.main_dataset.selected_indices
        self.vm_B = torch.Tensor(self.main_dataset.vm_B).cuda()
        self.edge_add_epoch = cfg.train.edge_loss_start

    def train(self, epoch, n_iters_total, master):
        self.model.train()
        
        metric_dict = defaultdict(list)

        lr_check(self.optimizer, epoch, master)

        running_loss = 0.0
        batch_generator = tqdm(self.batch_generator, dynamic_ncols=True) if master else self.batch_generator
        for i, meta in enumerate(batch_generator):
            for k, _ in meta.items():
                meta[k] = meta[k].cuda()
                if k == 'img':
                    meta[k] = meta[k].requires_grad_()

            imgs = meta['img'].cuda()
            inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
            gt_pose_root = meta['root_cam'].cuda()
            depth_factor = meta['depth_factor'].cuda()
            flip_mask = meta['flip'].cuda().reshape(-1) if cfg.aug.flip else None
            batch_size = imgs.shape[0]

            joint_uvd_valid, joint_cam_valid, mesh_valid = meta['joint_uvd_valid'].cuda(), meta['joint_cam_valid'].cuda(), meta['mesh_valid'].cuda()
            gt_pose, gt_uvd_pose, gt_mesh = meta['joint_cam'].cuda(), meta['joint_uvd'].cuda(), meta['mesh_cam'].cuda()
            
            _, pred_uvd_pose, _, confidence, pred_mesh, _, _ = self.model(imgs, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_item=None, flip_mask=flip_mask)
            pred_pose = torch.cat((torch.matmul(self.J_regressor, pred_mesh), torch.matmul(self.vm_B.T[None], pred_mesh[:, self.selected_indices])), dim=1)

            joint3d_loss = self.loss['joint_3d'](pred_uvd_pose, gt_uvd_pose, joint_uvd_valid)
            joint3d_reg_loss = self.loss['joint_reg_3d'](pred_pose, gt_pose, joint_cam_valid)
            conf_loss = self.loss['conf'](confidence, joint_uvd_valid)
            mesh3d_loss = self.loss['mesh_3d'](pred_mesh, gt_mesh, mesh_valid)
            mesh3d_normal_loss = self.loss['normal'](pred_mesh, gt_mesh)
            mesh3d_edge_loss = self.loss['edge'](pred_mesh, gt_mesh)

            loss = cfg.loss.loss_weight_joint3d * joint3d_loss + \
                cfg.loss.loss_weight_joint3d_reg * joint3d_reg_loss + \
                cfg.loss.loss_weight_conf * conf_loss + \
                cfg.loss.loss_weight_mesh3d * mesh3d_loss + \
                cfg.loss.loss_weight_normal * mesh3d_normal_loss

            metric_dict['joint3d_loss'].append(joint3d_loss.item())
            metric_dict['joint3d_reg_loss'].append(joint3d_reg_loss.item())
            metric_dict['conf_loss'].append(conf_loss.item())
            metric_dict['mesh3d_loss'].append(mesh3d_loss.item())
            metric_dict['mesh3d_normal_loss'].append(mesh3d_normal_loss.item())
            metric_dict['mesh3d_edge_loss'].append(mesh3d_edge_loss.item())
            
            if epoch > self.edge_add_epoch:
                loss += cfg.loss.loss_weight_edge * mesh3d_edge_loss
            metric_dict['total_loss'].append(loss.item())

            self.optimizer['3d'].zero_grad()
            self.optimizer['mesh'].zero_grad()
            loss.backward()
            self.optimizer['3d'].step()
            self.optimizer['mesh'].step()

            running_loss += float(loss.detach().item())

            if master:
                if i % self.print_freq == 0:
                    batch_generator.set_description(f'Epoch{epoch}_({i}/{len(self.batch_generator)}) => '
                                                    f'3d: {joint3d_loss.detach():.4f}, '
                                                    f'3d_r: {joint3d_reg_loss.detach():.4f}, '
                                                    f'conf: {conf_loss.detach():.4f}, '
                                                    f'mesh: {mesh3d_loss.detach():.4f}, '
                                                    f'norm: {mesh3d_normal_loss.detach():.4f}, '
                                                    f'edge: {mesh3d_edge_loss.detach():.4f}, '
                                                    f'total: {loss.detach():.4f} ')
                if i % self.vis_freq == 0:
                    vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='train_{:08}_joint3d_gt.jpg'.format(i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=min(batch_size//3, 4))
                    vis_joints_3d(imgs.detach().cpu().numpy(), pred_pose.detach().cpu().numpy(), None, file_name='train_{:08}_joint3d_pred.jpg'.format(i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=min(batch_size//3, 4))
                    vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='train_{:08}_mesh_gt.jpg'.format(i), nrow=min(batch_size//3, 4))
                    vis_joints_3d(imgs.detach().cpu().numpy(), pred_mesh.detach().cpu().numpy(), None, file_name='train_{:08}_mesh_pred.jpg'.format(i), nrow=min(batch_size//3, 4))


                for title, value in metric_dict.items():
                    self.writer.add_scalar("{}/{}".format('train', title), value[-1], n_iters_total)
                n_iters_total += 1

        self.loss_history.append(running_loss / len(self.batch_generator))

        if master:
            print(f'Epoch{epoch} Loss: {self.loss_history[-1]:.4f}')
        return n_iters_total

class Simple3DMeshTester:
    def __init__(self, args, load_path='', writer=None, master=None):
        self.val_loader_list, self.val_dataset, self.model, _, _, _, _, _, _, _ = \
            prepare_network(args, load_path=load_path, is_train=False, master=master)

        self.joint_num = self.val_dataset[0].joint_num
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.test.vis_freq
        self.writer = writer
        self.device = args.device

        self.J_regressor = torch.Tensor(self.val_dataset[0].joint_regressor).cuda()
        self.selected_indices = self.val_dataset[0].selected_indices
        self.vm_B = torch.Tensor(self.val_dataset[0].vm_B).cuda()

    def test(self, epoch, master, world_size, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        for dataset_name, val_dataset, val_loader in zip(cfg.dataset.test_list, self.val_dataset, self.val_loader_list):
            results = defaultdict(list)
            metric_dict = defaultdict(list)

            joint_error = 0.0
            if master:
                print('=> Evaluating on ', dataset_name, ' ... ')
            loader = tqdm(val_loader, dynamic_ncols=True) if master else val_loader
            with torch.no_grad():
                for i, meta in enumerate(loader):
                    for k, _ in meta.items():
                        meta[k] = meta[k].cuda()

                    imgs = meta['img'].cuda()
                    batch_size = imgs.shape[0]
                    inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                    depth_factor, gt_pose_root = meta['depth_factor'].cuda(), meta['root_cam'].cuda()

                    gt_pose, gt_mesh = meta['joint_cam'].cuda(), meta['mesh_cam'].cuda()

                    _, pred_uvd_pose, _, confidence, pred_mesh, _, _ = self.model(imgs, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_mask=None, is_train=False)

                    results['gt_pose'].append(gt_pose.detach().cpu().numpy())
                    results['gt_mesh'].append(gt_mesh.detach().cpu().numpy())

                    # flip_test
                    if isinstance(imgs, list):
                        imgs_flip = [flip_img(img.clone()) for img in imgs]
                    else:
                        imgs_flip = flip_img(imgs.clone())

                    _, _, _, _, pred_mesh_flip, _, _ = self.model(imgs_flip, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_item=(pred_uvd_pose, confidence), flip_output=True, flip_mask=None, is_train=False)

                    pred_pose_flip = torch.cat((torch.matmul(self.J_regressor, pred_mesh_flip), torch.matmul(self.vm_B.T[None], pred_mesh_flip[:, self.selected_indices])), dim=1)

                    results['pred_pose_flip'].append(pred_pose_flip.detach().cpu().numpy())
                    results['pred_mesh_flip'].append(pred_mesh_flip.detach().cpu().numpy())
                    results['gt_pose_root'].append(gt_pose_root.detach().cpu().numpy())
                    results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                    results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())

                    j_error = val_dataset.compute_joint_err(pred_pose_flip, gt_pose)

                    if master:
                        if i % self.print_freq == 0:
                            loader.set_description(f'{eval_prefix}({i}/{len(val_loader)}) => joint error: {j_error:.4f}')
                        if cfg.test.vis and i % self.vis_freq == 0:
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_gt.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=min(batch_size//3, 4))
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_gt.jpg'.format(dataset_name, i), nrow=min(batch_size//3, 4))
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_pose_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_pred_flip.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind, nrow=min(batch_size//3, 4))
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_mesh_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_pred_flip.jpg'.format(dataset_name, i), nrow=min(batch_size//3, 4))

                        joint_error += j_error
                for term in results.keys():
                    results[term] = np.concatenate(results[term])
            
                self.joint_error = joint_error / max(len(val_loader),1)

                if master:
                    joint_flip_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose_flip'], results['gt_pose'])
                    mesh_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_flip'], results['gt_mesh'],\
                        results['pred_pose_flip'], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name)

                    msg = ''
                    msg += f'\n{eval_prefix}'
                    for metric_key in joint_flip_error_dict.keys():
                        metric_dict[metric_key+'_REG'].append(joint_flip_error_dict[metric_key].item())
                        msg += f' | {metric_key:12}: {joint_flip_error_dict[metric_key]:3.2f}'
                    msg += f'\n{eval_prefix}'
                    for metric_key in mesh_flip_error_dict.keys():
                        metric_dict[metric_key].append(mesh_flip_error_dict[metric_key].item()) 
                        msg += f' | {metric_key:12}: {mesh_flip_error_dict[metric_key]:3.2f}'
                    print(msg)

                    for title, value in metric_dict.items():
                        self.writer.add_scalar("{}_{}/{}_epoch".format('val', dataset_name, title), value[-1], epoch)

                    # saving metric
                    metric_path = osp.join(cfg.metric_dir, "{}_metric_e{}_valset.json".format(dataset_name, epoch))
                    with open(metric_path, 'w') as fout:
                        json.dump(metric_dict, fout, indent=4, sort_keys=True)
                    print(f'=> writing metric dict to {metric_path}')

class Simple3DMeshPostTester:
    def __init__(self, args, load_path='', writer=None, master=None):
        self.val_loader_list, self.val_dataset, self.model, _, _, _, _, _, _, _ = \
            prepare_network(args, load_path=load_path, is_train=False, master=master)

        self.joint_num = self.val_dataset[0].joint_num
        self.draw_skeleton = True
        self.skeleton_kind = 'human36m'
        self.print_freq = cfg.train.print_freq
        self.vis_freq = cfg.test.vis_freq
        self.writer = writer
        self.device = args.device

        # initialize
        self.J_regressor = torch.Tensor(self.val_dataset[0].joint_regressor).cuda()
        self.selected_indices = self.val_dataset[0].selected_indices
        self.vm_B = torch.Tensor(self.val_dataset[0].vm_B).cuda()
        # noise_reduction
        assert cfg.model.simple3dmesh.noise_reduce, 'only support noise_reduce is True'


    def test(self, epoch, master, world_size, current_model=None):
        if current_model:
            self.model = current_model
        self.model.eval()

        eval_prefix = f'Epoch{epoch} ' if epoch else ''
        for dataset_name, val_dataset, val_loader in zip(cfg.dataset.test_list, self.val_dataset, self.val_loader_list):
            results = defaultdict(list)
            metric_dict = defaultdict(list)

            joint_error = 0.0
            print('=> Evaluating on ', dataset_name, ' ... ')
            loader = tqdm(val_loader, dynamic_ncols=True) if master else val_loader
            with torch.no_grad():
                for i, meta in enumerate(loader):
                    for k, _ in meta.items():
                        meta[k] = meta[k].cuda()
                    imgs = meta['img'].cuda()
                    inv_trans, intrinsic_param = meta['inv_trans'].cuda(), meta['intrinsic_param'].cuda()
                    depth_factor, gt_pose_root = meta['depth_factor'].cuda(), meta['root_cam'].cuda()

                    gt_pose, gt_mesh = meta['joint_cam'].cuda(), meta['mesh_cam'].cuda()

                    _, pred_uvd_pose, _, confidence, pred_mesh, pred_mesh_post, _ = self.model(imgs, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_mask=None, is_train=False)

                    results['gt_pose'].append(gt_pose.detach().cpu().numpy())
                    results['gt_mesh'].append(gt_mesh.detach().cpu().numpy())

                    # flip_test
                    if isinstance(imgs, list):
                        imgs_flip = [flip_img(img.clone()) for img in imgs]
                    else:
                        imgs_flip = flip_img(imgs.clone())

                    _, _, _, _, pred_mesh_flip, pred_mesh_post_flip, _ = self.model(imgs_flip, inv_trans, intrinsic_param, gt_pose_root, depth_factor, flip_item=(pred_uvd_pose, confidence), flip_output=True, flip_mask=None, is_train=False)

                    pred_pose_flip = torch.cat((torch.matmul(self.J_regressor, pred_mesh_post_flip), torch.matmul(self.vm_B.T[None], pred_mesh_post_flip[:, self.selected_indices])), dim=1)

                    results['pred_mesh_flip'].append(pred_mesh_flip.detach().cpu().numpy())
                    results['pred_mesh_post_flip'].append(pred_mesh_post_flip.detach().cpu().numpy())
                    results['pred_pose_flip'].append(pred_pose_flip.detach().cpu().numpy())
                    results['gt_pose_root'].append(gt_pose_root.detach().cpu().numpy())
                    results['focal_l'].append(meta['focal_l'].detach().cpu().numpy())
                    results['center_pt'].append(meta['center_pt'].detach().cpu().numpy())
    
                    j_error = val_dataset.compute_joint_err(pred_pose_flip, gt_pose)

                    if master:
                        if i % self.print_freq == 0:
                            loader.set_description(f'{eval_prefix}({i}/{len(val_loader)}) => joint error: {j_error:.4f}')
                        if cfg.test.vis and i % self.vis_freq == 0:
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_pose.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_gt.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind)
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_pose_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_joint3d_pred_flip.jpg'.format(dataset_name, i), draw_skeleton=self.draw_skeleton, dataset_name=self.skeleton_kind)
                            vis_joints_3d(imgs.detach().cpu().numpy(), gt_mesh.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_gt.jpg'.format(dataset_name, i))
                            vis_joints_3d(imgs.detach().cpu().numpy(), pred_mesh_post_flip.detach().cpu().numpy(), None, file_name='val_{}_{:08}_mesh_pred_post_flip.jpg'.format(dataset_name, i))
                        joint_error += j_error

                for term in results.keys():
                    results[term] = np.concatenate(results[term])
            
                self.joint_error = joint_error / len(val_loader)

                if master:
                    joint_post_flip_error_dict = val_dataset.compute_per_joint_err_dict(results['pred_pose_flip'], results['gt_pose'])
                    mesh_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_flip'], results['gt_mesh'],\
                        results['pred_pose_flip'], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name)
                    mesh_post_flip_error_dict = val_dataset.compute_per_mesh_err_dict(results['pred_mesh_post_flip'], results['gt_mesh'],\
                        results['pred_pose_flip'], results['gt_pose'], results['gt_pose_root'], results['focal_l'], results['center_pt'], dataset_name=dataset_name)

                    msg = ''
                    msg += f'\n{eval_prefix} post '
                    for metric_key in joint_post_flip_error_dict.keys():
                        metric_dict[metric_key+'_POST'].append(joint_post_flip_error_dict[metric_key].item())
                        msg += f' | {metric_key:12}: {joint_post_flip_error_dict[metric_key]:3.2f}'
                    msg += f'\n{eval_prefix} '
                    for metric_key in mesh_flip_error_dict.keys():
                        metric_dict[metric_key].append(mesh_flip_error_dict[metric_key].item()) 
                        msg += f' | {metric_key:12}: {mesh_flip_error_dict[metric_key]:3.2f}'
                    msg += f'\n{eval_prefix} post '
                    for metric_key in mesh_post_flip_error_dict.keys():
                        metric_dict[metric_key+'_POST'].append(mesh_post_flip_error_dict[metric_key].item()) 
                        msg += f' | {metric_key:12}: {mesh_post_flip_error_dict[metric_key]:3.2f}'

                    print(msg)

                    for title, value in metric_dict.items():
                        self.writer.add_scalar("{}_{}/{}_epoch".format('val', dataset_name, title), value[-1], epoch)

                    # saving metric
                    metric_path = osp.join(cfg.metric_dir, "{}_metric_e{}_valset.json".format(dataset_name, epoch))
                    with open(metric_path, 'w') as fout:
                        json.dump(metric_dict, fout, indent=4, sort_keys=True)
                    print(f'=> writing metric dict to {metric_path}')

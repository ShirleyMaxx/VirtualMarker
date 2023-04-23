import os
import sys
import time
import math
import numpy as np
import cv2
import shutil
import logging
from collections import OrderedDict
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from virtualmarker.core.config import cfg


def lr_check(optimizer, epoch, master):
    base_epoch = 5

    if master:
        msg = f"Current epoch {epoch}, "
        if cfg.model.name == 'simple3dmesh':
            for keys, opt in optimizer.items():
                for param_group in opt.param_groups:
                    curr_lr = param_group['lr']   
                msg += f"{keys}_lr: {curr_lr}, "
        else:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
            msg += f"lr: {curr_lr}"
        print(msg)

def lr_warmup(optimizer, lr, epoch, base):
    lr = lr * (epoch / base)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        self.acc += time.time() - self.t0  # cacluate time diff

    def reset(self):
        self.acc = 0

    def print(self):
        return round(self.acc, 2)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def stop():
    sys.exit()


def check_data_pararell(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model):
    if cfg.model.name == "simple3dmesh":
        opt_dict = {}
        lr_schd_dict = {}
        lr_dict = {}
        # opt for pose branch
        params_3d = [{'params': model.simple3dpose.parameters(), 'lr': cfg.train.simple3dpose_lr}]
        opt_3d = optim.Adam(params_3d)
        lr_schd_3d = get_scheduler(opt_3d, cfg.train.simple3dpose_lr_step, cfg.train.simple3dpose_lr_factor)
        opt_dict.update({'3d': opt_3d})
        lr_schd_dict.update({'3d': lr_schd_3d})
        lr_dict.update({'3d': cfg.train.simple3dpose_lr})
        # opt for mesh branch
        params_mesh = [{'params': model.adaptive_A.parameters(), 'lr': cfg.train.lr}]
        opt_mesh = optim.Adam(params_mesh)
        lr_schd_mesh = get_scheduler(opt_mesh, cfg.train.lr_step, cfg.train.lr_factor)
        opt_dict.update({'mesh': opt_mesh})
        lr_schd_dict.update({'mesh': lr_schd_mesh})
        lr_dict.update({'mesh': cfg.train.lr})
    else:
        opt = optim.Adam(model.parameters(), lr=cfg.train.lr)
        lr_schd = get_scheduler(opt, cfg.train.lr_step, cfg.train.lr_factor)
        opt_dict = opt
        lr_schd_dict = lr_schd
        lr_dict = cfg.train.lr

    return opt_dict, lr_schd_dict, lr_dict


def get_scheduler(optimizer, lr_step, lr_factor, scheduler_type='step'):
    scheduler = None
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=lr_factor)

    return scheduler


def save_checkpoint(states, epoch, is_best=None):
    file_name = f'checkpoint{epoch}.pth.tar'
    output_dir = cfg.checkpoint_dir
    if states['epoch'] == cfg.train.end_epoch:
        file_name = 'final.pth.tar'
    torch.save(states, os.path.join(output_dir, file_name))

    if is_best:
        torch.save(states, os.path.join(output_dir, 'best.pth.tar'))


def load_checkpoint(load_path, master=True):
    try:
        print(f"Fetch model weight from {load_path}")
        checkpoint = torch.load(load_path, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint %s exists!\n"%(load_path), e)

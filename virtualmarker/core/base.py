import os.path as osp
import numpy as np
import cv2
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from collections import Counter
import pickle
from termcolor import colored

from virtualmarker.dataset import *
import virtualmarker.models as models
from virtualmarker.dataset.multiple_datasets import MultipleDatasets
from virtualmarker.core.loss import get_loss
from virtualmarker.core.config import cfg
from virtualmarker.utils.funcs_utils import get_optimizer, load_checkpoint, count_parameters


def get_dataloader(args, dataset_names, is_train=True, transform=None, master=None):
    is_distributed = args.is_distributed
    dataset_list, dataloader_list = [], []

    for name in dataset_names:
        dataset_split = 'train' if is_train else 'test'
        batch_per_dataset = cfg[dataset_split].batch_size * args.gpus
        
        print(f"==> Preparing {name} ({dataset_split}) Dataloader...")
        dataset = eval(f'{name}')(dataset_split.lower(), args=args, transform=transform, master=master)
        
        print("# of {} {} data: {}".format(dataset_split, name, len(dataset)))
        if dataset_split == 'train':
            sampler = DistributedSampler(dataset) if (is_distributed and not is_train) else None
        else:
            sampler = None
        dataloader = DataLoader(dataset,
                                batch_size=batch_per_dataset,
                                shuffle=cfg[dataset_split].shuffle and (sampler is None),
                                sampler=sampler,
                                num_workers=cfg.dataset.workers,
                                pin_memory=True, drop_last=False)
        dataset_list.append(dataset)
        dataloader_list.append(dataloader)

    if not is_train:
        return dataset_list, dataloader_list, None, None
    else:
        trainset_loader = MultipleDatasets(dataset_list)
        sampler = DistributedSampler(trainset_loader) if is_distributed else None
        dataloader = DataLoader(dataset=trainset_loader, batch_size=batch_per_dataset, \
                                    shuffle=cfg[dataset_split].shuffle and (sampler is None), sampler=sampler, \
                                    num_workers=cfg.dataset.workers,
                                    pin_memory=True, drop_last=False)
        return dataset_list, dataloader, trainset_loader, sampler


def prepare_network(args, load_path='', is_train=True, master=None):
    is_distributed = args.is_distributed
    dataset_names = cfg.dataset.train_list if is_train else cfg.dataset.test_list
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    ])
    dataset_list, dataloader, dataset, sampler = get_dataloader(args, dataset_names, is_train, transform, master=master)
    model, criterion, optimizer, lr_scheduler = None, None, None, None
    loss_history, test_error_history = [], {'joint': []}
    model_name = cfg.model.name

    main_dataset = dataset_list[0]
    
    print(f"==> Preparing {model_name} model...")

    if model_name == 'simple3dmesh':
        vm_A = main_dataset.vm_A if hasattr(main_dataset, "vm_A") else None
        selected_indices = main_dataset.selected_indices if hasattr(main_dataset, "selected_indices") else None
        if cfg.model.simple3dmesh.noise_reduce:
            model = models.simple3dmesh_post.get_model(main_dataset.vertex_num, main_dataset.flip_pairs, \
                vm_A=vm_A, \
                selected_indices=selected_indices, \
                mesh_template=main_dataset.smpl_template)
        else:
            model = models.simple3dmesh.get_model(main_dataset.vertex_num, main_dataset.flip_pairs, \
                vm_A=vm_A, \
                selected_indices=selected_indices)

    print('# of model parameters: {}'.format(count_parameters(model)))

    if is_train:
        criterion = get_loss(faces=main_dataset.smpl_faces)
        optimizer, lr_scheduler, _ = get_optimizer(model=model)

    if load_path != '' and (not is_train or args.resume_training):
        print('==> Loading checkpoint')
        checkpoint = load_checkpoint(load_path, master)

        if 'model_state_dict' in checkpoint.keys():
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        try:
            if model_name == 'simple3dmesh' and cfg.model.simple3dmesh.noise_reduce:
                model.simple3dmesh.load_state_dict(state_dict, strict=True)
            else:
                model.load_state_dict(state_dict, strict=True)
            print(colored(f'Successfully load checkpoint from {load_path}.', 'green'))
        except:
            print(colored(f'Failed to load checkpoint in {load_path}', 'red'))

        if is_train:
            if model_name == 'simple3dmesh':
                for keys, opt in optimizer.items():
                    opt.load_state_dict(checkpoint[f'optim_state_dict_{keys}'])
                    for state in opt.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
                    curr_lr = 0.0

                    for param_group in opt.param_groups:
                        curr_lr = param_group['lr']        

                for keys, lr_schd in lr_scheduler.items():
                    lr_state = checkpoint['scheduler_state_dict_{}'.format(keys)]
                    lr_state['milestones'], lr_state['gamma'] = Counter(cfg.train.lr_step), cfg.train.lr_factor
                    lr_schd.load_state_dict(lr_state)

            else:
                optimizer.load_state_dict(checkpoint['optim_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
                curr_lr = 0.0

                for param_group in optimizer.param_groups:
                    curr_lr = param_group['lr']

                lr_state = checkpoint['scheduler_state_dict']
                # update lr_scheduler
                lr_state['milestones'], lr_state['gamma'] = Counter(cfg.train.lr_step), cfg.train.lr_factor
                lr_scheduler.load_state_dict(lr_state)

            loss_history = checkpoint['train_log']
            test_error_history = checkpoint['test_log']
            cfg.train.begin_epoch = checkpoint['epoch'] + 1
            print(colored('===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}'
                .format(cfg.train.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']), 'green'))
    # multi-gpu
    if is_distributed:
        # sync bn in multi-gpus
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(args.device)
        model = DistributedDataParallel(model, device_ids=[args.device], output_device=args.local_rank)
    else:
        model = torch.nn.DataParallel(model, device_ids=args.device)
        model = model.cuda()

    return dataloader, dataset_list, model, criterion, optimizer, lr_scheduler, loss_history, test_error_history, dataset, sampler

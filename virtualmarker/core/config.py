import os
import os.path as osp
import shutil

import yaml
from easydict import EasyDict as edict
import datetime

from tensorboardX import SummaryWriter

def init_dirs(dir_list):
    for dir in dir_list:
        if os.path.exists(dir) and os.path.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)


cfg = edict()

def init_experiment_dir(cur_path='.', data_path='.', experiment_name=''):

    """ Directory """
    cfg.root_dir = cur_path
    cfg.exp_dir = osp.join(cfg.root_dir, 'experiment')
    cfg.data_dir = osp.join(data_path, 'data')

    KST = datetime.timezone(datetime.timedelta(hours=8))
    save_folder = 'exp_' + str(datetime.datetime.now(tz=KST))[5:-16]
    save_folder = save_folder.replace(" ", "_").replace(":", "_")
    if experiment_name == '':
        save_folder_path = osp.join('experiment', save_folder)
    else:
        save_folder_path = osp.join('experiment', experiment_name, save_folder)

    cfg.output_dir = osp.join(cfg.root_dir, save_folder_path)
    cfg.tb_dir = osp.join(cfg.output_dir, 'tb')
    cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
    cfg.metric_dir = osp.join(cfg.output_dir, 'metric')
    cfg.checkpoint_dir = osp.join(cfg.output_dir, 'checkpoint')

    print("Experiment Data on {}".format(cfg.output_dir))
    init_dirs([cfg.output_dir, cfg.tb_dir, cfg.metric_dir, cfg.vis_dir, cfg.checkpoint_dir])

    # tensorboard
    writer = SummaryWriter(cfg.tb_dir)
    return writer


""" Dataset """
cfg.dataset = edict()
cfg.dataset.train_list = ['Human36M', 'COCO']
cfg.dataset.test_list = ['PW3D']
cfg.dataset.input_joint_set = 'vm'
cfg.dataset.num_joints = 81
cfg.dataset.workers = 64
cfg.dataset.use_coco3d_up3d = False
cfg.dataset.smpl_indices_path = 'smpl/smpl_indices.pkl'
cfg.dataset.use_tight_bbox = True


""" Model """
cfg.model = edict()
cfg.model.name = 'simple3dmesh'
cfg.model.input_shape = (256, 256)
cfg.model.heatmap_shape = (64, 64)
cfg.model.bbox_3d_shape = (2000, 2000, 2000)

cfg.model.mesh2vm = edict()
cfg.model.mesh2vm.vm_path = './assets/64'
cfg.model.mesh2vm.vm_type = '_sym'
cfg.model.mesh2vm.vm_K = 64
cfg.model.mesh2vm.initial_B_path = ''
cfg.model.mesh2vm.ignore_part = ['head']

cfg.model.simple3dpose = edict()
cfg.model.simple3dpose.num_deconv_filters = [256, 256, 256]
cfg.model.simple3dpose.backbone = 'HRNet'   
cfg.model.simple3dpose.extra_norm_type = 'softmax'
cfg.model.simple3dpose.extra_depth_dim = 64
cfg.model.simple3dpose.alpha = 15

cfg.model.hrnet = edict()
cfg.model.hrnet.final_conv_kernel = 1
cfg.model.hrnet.pretrained_layers = ['*']
cfg.model.hrnet.pretrained = 'models/pytorch/pose_coco/pose_hrnet_w48_384x288.pth'
cfg.model.hrnet.stage2 = edict()
cfg.model.hrnet.stage2.num_channels = [48, 96]
cfg.model.hrnet.stage2.block = 'BASIC'
cfg.model.hrnet.stage2.num_modules = 1
cfg.model.hrnet.stage2.num_branches = 2
cfg.model.hrnet.stage2.num_blocks = [4, 4]
cfg.model.hrnet.stage2.fuse_method = 'SUM'
cfg.model.hrnet.stage3 = edict()
cfg.model.hrnet.stage3.num_channels = [48, 96, 192]
cfg.model.hrnet.stage3.block = 'BASIC'
cfg.model.hrnet.stage3.num_modules = 4
cfg.model.hrnet.stage3.num_branches = 3
cfg.model.hrnet.stage3.num_blocks = [4, 4, 4]
cfg.model.hrnet.stage3.fuse_method = 'SUM'
cfg.model.hrnet.stage4 = edict()
cfg.model.hrnet.stage4.num_channels = [48, 96, 192, 384]
cfg.model.hrnet.stage4.block = 'BASIC'
cfg.model.hrnet.stage4.num_modules = 3
cfg.model.hrnet.stage4.num_branches = 4
cfg.model.hrnet.stage4.num_blocks = [4, 4, 4, 4]
cfg.model.hrnet.stage4.fuse_method = 'SUM'

cfg.model.simple3dmesh = edict()
cfg.model.simple3dmesh.pretrained = ''
cfg.model.simple3dmesh.noise_reduce = False
cfg.model.simple3dmesh.noise_reduce_pretrained = 'experiment/noise_reduction_pretrain/final.pth.tar'
cfg.model.simple3dmesh.fix_network = False


""" Train Detail """
cfg.train = edict()
cfg.train.print_freq = 10
cfg.train.vis_freq = 1000
cfg.train.batch_size = 64
cfg.train.shuffle = True
cfg.train.begin_epoch = 1
cfg.train.end_epoch = 40
cfg.train.edge_loss_start = 7
cfg.train.scheduler = 'step'
cfg.train.lr = 1e-3
cfg.train.lr_step = [30]
cfg.train.lr_factor = 0.1
cfg.train.simple3dpose_lr = 0.0005
cfg.train.simple3dpose_lr_step = [30]
cfg.train.simple3dpose_lr_factor = 0.5
cfg.train.simple3dmesh_lr = 0.00025
cfg.train.simple3dmesh_lr_step = [5]
cfg.train.simple3dmesh_lr_factor = 0.1
cfg.train.optimizer = 'adam'
cfg.train.resume_weight_path = ''

""" Augmentation """
cfg.aug = edict()
cfg.aug.flip = False
cfg.aug.rotate_factor = 0 
cfg.aug.scale_factor = 0  
cfg.aug.color_factor = 0
cfg.aug.occlusion = True

""" Test Detail """
cfg.test = edict()
cfg.test.batch_size = 64
cfg.test.vis_freq = 500
cfg.test.shuffle = False
cfg.test.weight_path = ''
cfg.test.vis = True
cfg.test.save_obj = False
cfg.test.save_result = False

cfg.loss = edict()
cfg.loss.loss_weight_joint3d = 1e-3
cfg.loss.loss_weight_joint3d_reg = 1e-3
cfg.loss.loss_weight_joint3d_reg_post = 1e-3
cfg.loss.loss_weight_conf = 1
cfg.loss.loss_weight_mesh3d = 1
cfg.loss.loss_weight_mesh3d_post = 1
cfg.loss.loss_weight_normal = 1e-1
cfg.loss.loss_weight_edge = 1e-2
cfg.loss.loss_weight_beta = 1
cfg.loss.loss_weight_theta = 1



def _update_dict(cfg, k, v):
    for vk, vv in v.items():
        if vk in cfg:
            if isinstance(vv, dict):
                _update_dict(cfg[vk], vk, vv)
            else:
                cfg[vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, yaml.SafeLoader))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    _update_dict(cfg[k], k, v)
                else:
                    if k == 'SCALES':
                        cfg[k][0] = (tuple(v))
                    else:
                        cfg[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))




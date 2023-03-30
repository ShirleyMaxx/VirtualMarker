<div align="center">

  <h1 align="center">3D Human Mesh Estimation from Virtual Markers <br> (CVPR 2023)</h1>
  
</div>

<div align="left">

  <a>![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)</a>
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://github.com/ShirleyMaxx/VirtualMarker/blob/main/LICENSE">![License](https://img.shields.io/github/license/metaopt/torchopt?label=license&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjI0IiBoZWlnaHQ9IjI0IiBmaWxsPSIjZmZmZmZmIj48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0xMi43NSAyLjc1YS43NS43NSAwIDAwLTEuNSAwVjQuNUg5LjI3NmExLjc1IDEuNzUgMCAwMC0uOTg1LjMwM0w2LjU5NiA1Ljk1N0EuMjUuMjUgMCAwMTYuNDU1IDZIMi4zNTNhLjc1Ljc1IDAgMTAwIDEuNUgzLjkzTC41NjMgMTUuMThhLjc2Mi43NjIgMCAwMC4yMS44OGMuMDguMDY0LjE2MS4xMjUuMzA5LjIyMS4xODYuMTIxLjQ1Mi4yNzguNzkyLjQzMy42OC4zMTEgMS42NjIuNjIgMi44NzYuNjJhNi45MTkgNi45MTkgMCAwMDIuODc2LS42MmMuMzQtLjE1NS42MDYtLjMxMi43OTItLjQzMy4xNS0uMDk3LjIzLS4xNTguMzEtLjIyM2EuNzUuNzUgMCAwMC4yMDktLjg3OEw1LjU2OSA3LjVoLjg4NmMuMzUxIDAgLjY5NC0uMTA2Ljk4NC0uMzAzbDEuNjk2LTEuMTU0QS4yNS4yNSAwIDAxOS4yNzUgNmgxLjk3NXYxNC41SDYuNzYzYS43NS43NSAwIDAwMCAxLjVoMTAuNDc0YS43NS43NSAwIDAwMC0xLjVIMTIuNzVWNmgxLjk3NGMuMDUgMCAuMS4wMTUuMTQuMDQzbDEuNjk3IDEuMTU0Yy4yOS4xOTcuNjMzLjMwMy45ODQuMzAzaC44ODZsLTMuMzY4IDcuNjhhLjc1Ljc1IDAgMDAuMjMuODk2Yy4wMTIuMDA5IDAgMCAuMDAyIDBhMy4xNTQgMy4xNTQgMCAwMC4zMS4yMDZjLjE4NS4xMTIuNDUuMjU2Ljc5LjRhNy4zNDMgNy4zNDMgMCAwMDIuODU1LjU2OCA3LjM0MyA3LjM0MyAwIDAwMi44NTYtLjU2OWMuMzM4LS4xNDMuNjA0LS4yODcuNzktLjM5OWEzLjUgMy41IDAgMDAuMzEtLjIwNi43NS43NSAwIDAwLjIzLS44OTZMMjAuMDcgNy41aDEuNTc4YS43NS43NSAwIDAwMC0xLjVoLTQuMTAyYS4yNS4yNSAwIDAxLS4xNC0uMDQzbC0xLjY5Ny0xLjE1NGExLjc1IDEuNzUgMCAwMC0uOTg0LS4zMDNIMTIuNzVWMi43NXpNMi4xOTMgMTUuMTk4YTUuNDE4IDUuNDE4IDAgMDAyLjU1Ny42MzUgNS40MTggNS40MTggMCAwMDIuNTU3LS42MzVMNC43NSA5LjM2OGwtMi41NTcgNS44M3ptMTQuNTEtLjAyNGMuMDgyLjA0LjE3NC4wODMuMjc1LjEyNi41My4yMjMgMS4zMDUuNDUgMi4yNzIuNDVhNS44NDYgNS44NDYgMCAwMDIuNTQ3LS41NzZMMTkuMjUgOS4zNjdsLTIuNTQ3IDUuODA3eiI+PC9wYXRoPjwvc3ZnPgo=)</a>
  [![arXiv](https://img.shields.io/badge/arXiv-2303.11726-b31b1b.svg)](https://arxiv.org/pdf/2303.11726.pdf)
  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/3d-human-mesh-estimation-from-virtual-markers-1/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=3d-human-mesh-estimation-from-virtual-markers-1)

</div>


<p align="center">
  <img src="demo/quality_results.png"/>
</p>
<p align="middle">
  <img src="demo/demo_result1.gif" height="120" /> 
  <img src="demo/demo_result2.gif" height="120" /> 
  <img src="demo/demo_result3.gif" height="120" /> 
  <img src="demo/demo_result4.gif" height="120" /> 
  <img src="demo/demo_result5.gif" height="120" /> 
</p>

## Introduction

This is the offical [Pytorch](https://pytorch.org/) implementation of our paper:
<h3 align="center">3D Human Mesh Estimation from Virtual Markers (CVPR 2023)</h3>

<h4 align="center" style="text-decoration: none;">
  <a href="https://shirleymaxx.github.io/", target="_blank"><b>Xiaoxuan Ma</b></a>
  ,
  <a href="https://scholar.google.com/citations?user=DoUvUz4AAAAJ&hl=en", target="_blank"><b>Jiajun Su</b></a>
  ,
  <a href="https://www.chunyuwang.org/", target="_blank"><b>Chunyu Wang</b></a>
  ,
  <a href="https://wentao.live/", target="_blank"><b>Wentao Zhu</b></a>
  ,
  <a href="https://cfcs.pku.edu.cn/english/people/faculty/yizhouwang/index.htm", target="_blank"><b>Yizhou Wang</b></a>

</h4>
<h4 align="center">
  <a href="https://arxiv.org/pdf/2303.11726.pdf", target="_blank">[arXiv]</a>
</h4>

Below is the learned virtual markers and the overall framework.

<p align="center">
  <img src="demo/virtualmarker.gif" height="160" />
  <img src="demo/pipeline.png" height="160" /> 
</p>


## TODO :white_check_mark:

- [ ] Provide inference code


## Installation

1. Clone this codebase as ${Project}.
2. Install dependences. This project is developed using >= python 3.8 on Ubuntu 16.04. NVIDIA GPUs are needed. We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment.

  ```bash
    # 1. Create a conda virtual environment.
    conda create -n pytorch python=3.8 -y
    conda activate pytorch

    # 2. Install PyTorch >= v1.6.0 following [official instruction](https://pytorch.org/). Please adapt the cuda version to yours.
    pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

    # 3. Install other packages. This project doesn't have any special or difficult-to-install dependencies.
    sh requirements.sh
  ```
3. Prepare SMPL layer. We use [smplx](https://github.com/vchoutas/smplx#installation).

   1. Install `smplx` package by `pip install smplx`.
   2. Download `basicModel_f_lbs_10_207_0_v1.0.0.pkl`, `basicModel_m_lbs_10_207_0_v1.0.0.pkl`, and `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) (female & male) and [here](http://smplify.is.tue.mpg.de/) (neutral) to `${Project}/data/smpl`. Please rename them as `SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`, and `SMPL_NEUTRAL.pkl`, respectively.
   3. Download others SMPL-related from [here](https://pkueducn-my.sharepoint.com/:f:/g/personal/maxiaoxuan_pku_edu_cn/Ekse-wqgeKVLoTm5lS-aKRABkE_wooh4E83SHEhDxb8H3g?e=s8DWEG) and put them to `${Project}/data/smpl`.
4. Download data following the **Data** section. In summary, your directory tree should be like this

  ```
    ${Project}
    ├── assets
    ├── command
    ├── configs
    ├── data 
    ├── demo 
    ├── experiment 
    ├── inputs 
    ├── lib 
    ├── main 
    ├── models 
    ├── README.md
    `── requirements.sh
  ```

  - `assets` contains the body virtual markers in `npz` format. Feel free to use them.
  - `command` contains the running scripts.
  - `configs` contains the configurations in `yml` format.
  - `data` contains soft links to images and annotations directories.
  - `lib` contains kernel codes for our method.
  - `main` contains high-level codes for training or testing the network.
  - `models` contains pre-trained weights. Download from [here](https://pkueducn-my.sharepoint.com/:f:/g/personal/maxiaoxuan_pku_edu_cn/EmtcUZXZAxtPsxIyoOrS5m0B-ox4dzS_9wBAgSyYbq_flQ?e=QVxc2E).
  - *`experiment` will be automatically made after running the code, it contains the outputs, including trained model weights, test metrics and visualized outputs.


## Train & Eval

### Data

The `data` directory structure should follow the below hierarchy. Please download the images from the official sites. Download all the processed annotation files from [here](https://pkueducn-my.sharepoint.com/:f:/g/personal/maxiaoxuan_pku_edu_cn/Eq8EbSOjkaRHkMkV_BbvDYYBkzSGTlDlZ_muWy7IscZWGA?e=3Rk6ub).

```
${Project}
|-- data
    |-- 3DHP
    |   |-- annotations
    |   `-- images
    |-- COCO
    |   |-- annotations
    |   `-- images
    |-- Human36M
    |   |-- annotations
    |   `-- images
    |-- PW3D
    |   |-- annotations
    |   `-- images
    |-- SURREAL
    |   |-- annotations
    |   `-- images
    |-- Up_3D
    |   |-- annotations
    |   `-- images
    `-- smpl
        |-- smpl_indices.pkl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_MALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- mesh_downsampling.npz
        |-- J_regressor_extra.npy
        `-- J_regressor_h36m_correct.npy
```

### Train

Every experiment is defined by `config` files. Configs of the experiments in the paper can be found in the `./configs` directory. You can use the scripts under `command` to run.

To train the model, simply run the script below. Specific configurations can be modified in the corresponding `configs/simple3dmesh_train/baseline.yml` file. Default setting is using 4 GPUs (16G V100). Multi-GPU training is implemented with PyTorch's [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel). Results can be seen in `experiment` directory or in the tensorboard.

We conduct mix-training on H3.6M and 3DPW datasets. To get the reported results on 3DPW dataset, please first run `train_h36m.sh` and then load the final weight to train on 3DPW by running `train_pw3d.sh`. We train a seperate model on SURREAL dataset using `train_surreal.sh`. 

```bash
sh command/simple3dmesh_train/train_h36m.sh
sh command/simple3dmesh_train/train_pw3d.sh
sh command/simple3dmesh_train/train_surreal.sh
```

### Evaluation

To evaluate the model, specify the model path `test.weight_path` in `configs/simple3dmesh_test/baseline_*.yml`. Argument `--mode test` should be set. Results can be seen in `experiment` directory or in the tensorboard.

```bash
sh command/simple3dmesh_test/test_h36m.sh
sh command/simple3dmesh_test/test_pw3d.sh
sh command/simple3dmesh_test/test_surreal.sh
```

### Model Zoo

| Test set | MPVE |     MPJPE     | PA-MPJPE |  Download | Config |  
|----------|----------|------------|------------|-------|-----------|
| Human3.6M    | 58.0 | 47.3 | 32.0 | [model](https://pkueducn-my.sharepoint.com/:f:/g/personal/maxiaoxuan_pku_edu_cn/EksPuX3YJlZNjAHgJhRKj1QBzWKC67_ao3ksbwLRK7VTQQ?e=rBOrrK) | [cfg](./configs/simple3dmesh_train/baseline_h36m.yml)    |
| 3DPW         | 77.9 | 67.5 | 41.3 | [model](https://pkueducn-my.sharepoint.com/:f:/g/personal/maxiaoxuan_pku_edu_cn/EjoJjNeDm1ZLq4AwDyBX39MBTl9SVDXdlhU4KPraxxipcA?e=tPREyy) | [cfg](./configs/simple3dmesh_train/baseline_pw3d.yml)    |
| SURREAL      | 44.7 | 36.9 | 28.9 | [model](https://pkueducn-my.sharepoint.com/:f:/g/personal/maxiaoxuan_pku_edu_cn/EtYVRm2N3MhJrzIwJaMa7HEBG-ijLD_0UK9pfKcqwVkSuA?e=438Ejk) | [cfg](./configs/simple3dmesh_train/baseline_surreal.yml)    |
| in-the-wild* | | | | [model](https://pkueducn-my.sharepoint.com/:f:/g/personal/maxiaoxuan_pku_edu_cn/Egq_U92SyMxJvjb0g3-M16YBd02iG8ZCg_dmPFM2e5XjMw?e=aM8efG) |  |

\* We further train a model for better inference performance on in-the-wild scenes by finetuning the 3DPW model on SURREAL dataset. 


## Citation
Cite as below if you find this repository is helpful to your project:
```bibtex
@article{ma20233d,
  title={3D Human Mesh Estimation from Virtual Markers},
  author={Ma, Xiaoxuan and Su, Jiajun and Wang, Chunyu and Zhu, Wentao and Wang, Yizhou},
  journal={arXiv preprint arXiv:2303.11726},
  year={2023}
}
```


## Acknowledgement
This repo is built on the excellent work [GraphCMR](https://github.com/nkolot/GraphCMR), [SPIN](https://github.com/nkolot/SPIN), [Pose2Mesh](https://github.com/hongsukchoi/Pose2Mesh_RELEASE), [HybrIK](https://github.com/Jeff-sjtu/HybrIK) and [CLIFF](https://github.com/haofanwang/CLIFF). Thanks for these great projects.
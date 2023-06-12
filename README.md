# Multiscale Object Contrastive Learning-Derived Few-Shot Object Detection in VHR Imagery

The code for “Multiscale Object Contrastive Learning–Derived Few-Shot Object Detection in VHR imagery”

[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/zh_CN/latest/)

This repo contains the implementation of our fewshot object detector, described in our TGRS 2022 paper, [Multiscale Object Contrastive Learning–Derived Few-Shot Object Detection in VHR imagery ](https://ieeexplore.ieee.org/document/9984671). MSOCL is built upon the codebase [FsDet v0.4](https://github.com/ucbdrive/few-shot-object-detection/tags), which released by an ICML 2020 paper [Frustratingly Simple Few-Shot Object Detection](https://arxiv.org/abs/2003.06957).

![image-20230606223141592](https://qindengda.oss-cn-beijing.aliyuncs.com/typora202306062231653.png)

If you find this repository useful for your publications, please consider citing our paper.

```
@ARTICLE{9984671,
  author={Chen, Jie and Qin, Dengda and Hou, Dongyang and Zhang, Jun and Deng, Min and Sun, Geng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Multiscale Object Contrastive Learning-Derived Few-Shot Object Detection in VHR Imagery}, 
  year={2022},
  volume={60},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2022.3229041}}
```



## Installation

FsDet is built on [Detectron2](https://github.com/facebookresearch/detectron2). You need to build detectron2 . You can follow the instructions below to install the dependencies and build `FsDet`. 

**Dependencies**

- Linux with Python >= 3.6
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.4
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation
- CUDA 9.2, 10.0, 10.1, 10.2, 11.0
- GCC >= 4.9

**Build FsDet**

1. You can also use `conda` to create a new environment.

```
conda create --n msocl python=3.8
conda activate msocl
```

1. Install PyTorch. You can choose the PyTorch and CUDA version according to your machine. Just make sure your PyTorch version matches the prebuilt Detectron2 version (next step). Example for PyTorch v1.7.1:

```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Currently, the codebase is compatible with [Detectron2 v0.4](https://github.com/facebookresearch/detectron2/releases/tag/v0.3). Example for PyTorch v1.7.1 and CUDA v11.0:

- Install Detectron2 v0.4

```
pip install detectron2==0.4 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```

- Install other requirements.

```
pip install -r requirements.txt
```



## Data preparation

- We evaluate our models on DIOR datasets:
  - [DIOR datasets](http://www.escience.cn/people/gongcheng/DIOR.html): We use the train/val sets of DIOR datasets for training and the test set of  DIOR datasets for evaluation. We randomly split the 20 object classes into 15 base classes and 5 novel classes, and we consider 4 splits like [P-CNN](https://ieeexplore.ieee.org/document/9435769). The splits can be found in [fsdet/data/builtin_meta.py](https://github.com/ucbdrive/few-shot-object-detection/blob/v0.3/fsdet/data/builtin_meta.py).
  
  

- Data Structure

```
├── datasets
│   ├── DIOR
│   │   ├── JPEGImages
│   │   ├── Annotations
│   │   ├── ImageSets
│   │   │   ├── Main
│   ├── diorsplit
```

If you would like to use your own custom dataset, see [CUSTOM.md](https://github.com/ucbdrive/few-shot-object-detection/blob/v0.4/docs/CUSTOM.md) for instructions. If you would like to contribute your custom dataset to our codebase, feel free to open a PR.



## Train & Inference

### Training

We follow the eaact training procedure of FsDet and we use **random initialization** for novel weights. For a full description of training procedure, see [here](https://github.com/ucbdrive/few-shot-object-detection/blob/master/docs/TRAIN_INST.md).

#### 1. Stage 1: Training base detector.

```
python -m tools.train_net \
        --config-file configs/DIOR/base-training/R101_FPN_base_training_split1.yml
```



#### 2. Random initialize  weights for novel classes.

```
python -m tools.ckpt_surgery \
        --src1 checkpoints/dior/faster_rcnn/faster_rcnn_R_101_FPN_base1/model_final.pth \
        --method randinit \
        --save-dir checkpoints/dior/faster_rcnn/faster_rcnn_R_101_FPN_all1
```

This step will create a `model_surgery.pth` from` model_final.pth`. 



#### 3. Stage 2: Fine-tune for novel data.

```
python -m tools.train_net \
        --config-file configs/PASCAL_VOC/split1/10shot_CL_IoU.yaml \
        --opts MODEL.WEIGHTS WEIGHTS_PATH
```

Where `WEIGHTS_PATH` points to the `model_surgery.pth` generated from the previous step. Or you can specify it in the configuration yaml. 



#### Evaluation

To evaluate the trained models, run

```angular2html
python tools.test_net \
        --config-file configs/PASCAL_VOC/split1/10shot_CL_IoU.yml \
        --eval-only
```

Or you can specify `TEST.EVAL_PERIOD` in the configuation yml to evaluate during training. We use  **PASCAL VOC  benchmark** for evaluation.


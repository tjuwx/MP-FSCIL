# MP-FSCIL: The Model Preconception Restrains Few-Shot Class Incremental Learning

PyTorch implementation of MP-FSCIL: The Model Preconception Restrains Few-Shot Class Incremental Learning

## Abstract
Few-Shot Class Incremental Learning (FSCIL) aims at incrementally learning new knowledge from few training examples without forgetting previous knowledge. Existing approaches tend to suffer from the model preconception issue that the model excessively focuses on the class-specific features of the base classes to the detriment of the novel class representations. In this work, we alleviate this issue by enhancing feature representation capabilities to improve the model transferability, and by increasing knowledge transmission across downstream tasks to improve the model adaptability to the coming tasks. Specifically, we expand the attention regions to capture global context information and then scatter intra-class samples to further extract the object-specific features. Furthermore, we implement the knowledge transmission across different tasks with a Transformational Adaptation (TA) strategy to continuously exploit new knowledge from downstream tasks. Extensive experimental results on \textit{mini}-ImageNet, CIFAR100, and CUB200 datasets demonstrate that the proposed approach outperforms the state-of-the-art approaches by a large margin. Code will be available at \url{https://github.com/esmhcism/MP-FSCIL}.


## Requirements
- [PyTorch >= version 1.1](https://pytorch.org)
- tqdm

## Datasets and pretrained models
We follow [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training.  
For CIFAR100, the dataset will be download automatically.  
For miniImagenet and CUB200, you can download from [here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN?usp=sharing). Please put the downloaded file under `data/` folder and unzip it:
    
    $ tar -xvf miniimagenet.tar
    $ tar -xvzf CUB_200_2011.tgz

## Training scripts
cifar100
    $python train.py -dataset cifar100 -pre_epochs 100 -pre_lr 0.1 -pre_batch_size 512 -num 1 -meta_episode 100 -meta_shot 3 -meta_way 20 -gpu 0,1,2,3 -temperature 16
    
mini_imagenet
    $python train.py -dataset mini_imagenet -pre_epochs 100 -pre_lr 0.1 -pre_batch_size 512 -num 2 -meta_episode 100 -meta_shot 3 -meta_way 20 -gpu 0,1,2,3 -temperature 16

cub200
    $python train.py -dataset cub200 -pre_epochs 100 -pre_lr 0.005 -pre_batch_size 512 -num 2 -meta_episode 100 -meta_shot 3 -meta_way 20 -gpu 0,1,2,3 -temperature 16


## Acknowledgment
Our project references the codes in the following repos.

- [fscil](https://github.com/xyutao/fscil)
- [DeepEMD](https://github.com/icoz69/DeepEMD)
- [CEC](https://github.com/icoz69/CEC-CVPR2021)
- [RFS](https://github.com/WangYueFt/rfs)

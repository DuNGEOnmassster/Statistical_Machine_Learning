# 运行前先按下面方法安装sklearn库
# pip install -U scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple/

import os.path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.dataloader_svm import Dataset, dataset_collate
from utils.trainer_svm import fit_one_epoch

from nets.model import Baseline


if __name__ == "__main__":
    Cuda = False
    # ------------------------------------------------------#
    #   pretrained_model_path        网络预训练权重文件路径
    # ------------------------------------------------------#
    pretrained_model_path  = ''
    # ------------------------------------------------------#
    #   input_shape     输入的shape大小
    # ------------------------------------------------------#
    input_shape = [28, 28]
    batch_size = 32

    num_workers = 0

    # ------------------------------------------------------#
    #   train_val_dataset_path   训练和测试文件路径
    # ------------------------------------------------------#
    train_val_dataset_path = 'dataset/NewDataset.mat'

    # ---------------------------------------#
    #   构建数据集加载器。
    # ---------------------------------------#
    train_dataset = Dataset(train_val_dataset_path, input_shape, is_train=True)
    val_dataset = Dataset(train_val_dataset_path, input_shape, is_train=False)

    shuffle = False

    train_gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=train_dataset.__len__(), num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)

    val_gen = DataLoader(val_dataset, shuffle=shuffle, batch_size=val_dataset.__len__(), num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)

    # import pdb; pdb.set_trace()

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    fit_one_epoch(train_gen, val_gen)
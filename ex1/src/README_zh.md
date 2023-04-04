# 这是NormanZ的`模式识别课程`第一次作业的官方代码实现，本次实验基于CNN和SVM完成了对200类别omniglot数据集的分类

## 快速开始⚡

### 配置环境

推荐您使用anaconda

```shell script
conda create -n py310 python=3.10 <--file requirements.txt>

conda activate py310
```

如果在GPU环境运行，建议您不带--file参数创建环境，在(py310)下根据 https://pytorch.org/ 的提示完成对应gpu和cuda version 的 PyTorch安装, 然后运行

```shell script
pip install -r requirement.txt
```
### 训练CNN

```shell script
CUDA_VISIBLE_DEVICES=<gpu_id e.g. "0"> train.py
```

训练CNN模型，提供了`from nets.model import`的三种模型结构`Baseline`,`Baseline2`,`Baseline3`，以及`from nets.resnet import`的五种ResNet结构`resnet18`,`resnet34`,`resnet50`,`resnet101`,`resnet152`可供使用

[dataset](./dataset/)中放置了本次使用的`omniglot`数据集，[checkpoints](./checkpoints/)中保存了预训练权重，[figure](./figure/)中保存了部分实验过程的loss与acc曲线

本次实验并没有选择argparse配置外传参数集，可以在主函数内配置超参数，默认为

- Cuda = True
- pretrained_model_path  = ''
- input_shape = [28, 28]
- batch_size = 128
- Init_Epoch = 0
- Epoch = 50
- random_seed = 3407
- Init_lr = 0.002
- save_dir = './checkpoints/'
- save_period = 10
- train_val_dataset_path = 'dataset/NewDataset.mat'

### 训练SVM

```shell script
python train_svm.py
```

SVM的`kernel`等参数的修改位于[trainer_svm](./utils/trainer_svm.py)，如果需要CV调参则将 `need_cv` 设置为 `True`，需要测试（通过`classification_report`得到结果）则将`need_test` 设置为 `True`

## Enjoy!😄
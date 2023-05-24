This is the official implementation of the first assignment of NormanZ's `Pattern Recognition Course`, which have completed the classification of the 200-category omniglot dataset based on CNN and SVM

## Quick Start⚡

### Setup

You need to prepare the virtual environment first, and we recommend you to use Anaconda

```shell script
conda create -n py310 python=3.10 <--file requirements.txt>

conda activate py310
```

If running in a GPU environment, it is recommended that you create the environment without the `--file parameter`. Complete the PyTorch installation corresponding to your gpu and cuda version according to the prompt of https://pytorch.org/, and then run

```shell script
pip install -r requirement.txt
```
### Train CNN

```shell script
CUDA_VISIBLE_DEVICES=<gpu_id e.g. "0"> train.py
```

To train the CNN model, we provide three model structures `Baseline`, `Baseline2`, `Baseline3` from `nets.model`, and five ResNet structures `resnet18`, `resnet34` from `nets.resnet`, `resnet50`, `resnet101`, `resnet152` are available


[dataset](./dataset/) stores the `omniglot` dataset used this experiment, [checkpoints](./checkpoints/) stores the pre-training weights, [figure](./figure/) saves selected `loss and acc curves` of training processes.

This experiment did not use `argparse` to configure the external parameter set, you can configure hyperparameters in the main function, the default is:

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

### Train SVM

```shell script
python train_svm.py
```

The modification of parameters such as `kernel` of SVM is located in [trainer_svm](./utils/trainer_svm.py). If CV tuning is required, set `need_cv` to `True`, and if testing(get the result through `classification_report`) is required, set `need_test` to `True`.

## Enjoy!😄
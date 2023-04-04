import os.path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.dataloader_cl import Dataset, dataset_collate
from utils.trainer import fit_one_epoch, draw_loss, draw_acc

from nets.model import Baseline2
# from nets.resnet import resnet50

train_losses = []
train_acc = []
train_counter = []
test_losses = []
test_acc = []

if __name__ == "__main__":
    Cuda = True
    # ------------------------------------------------------#
    #   pretrained_model_path        网络预训练权重文件路径
    # ------------------------------------------------------#
    # pretrained_model_path  = './logs/ep300-loss0.000.pth'
    # pretrained_model_path  = './checkpoints/baseline2_ep100.pth'
    pretrained_model_path  = ''
    # ------------------------------------------------------#
    #   input_shape     输入的shape大小
    # ------------------------------------------------------#
    input_shape = [28, 28]
    batch_size = 128
    Init_Epoch = 0
    Epoch = 50

    # ------------------------------------------------------#
    #   random seed     随机种子
    # ------------------------------------------------------#
    random_seed = 3407
    torch.manual_seed(random_seed)

    # ------------------------------------------------------#
    #   Init_lr     初始学习率
    # ------------------------------------------------------#
    Init_lr = 0.002
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 10
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = './checkpoints/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_workers = 0

    # ------------------------------------------------------#
    #   train_val_dataset_path   训练和测试文件路径
    # ------------------------------------------------------#
    train_val_dataset_path = 'dataset/NewDataset.mat'

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------------------------#
    #   创建模型
    # ------------------------------------------------------#
    model = Baseline2()
    # model = resnet50()

    if pretrained_model_path != '':
        print('Load weights {}.'.format(pretrained_model_path))
        # pretrained_dict = torch.load(pretrained_model_path, map_location= device)
        # model.load_state_dict(pretrained_dict)
        layers_False = ["fc.weight", "fc.bias"]
        pretrained_dict = torch.load(pretrained_model_path, map_location= device)
        # 删除有关分类类别的权重
        for k in list(pretrained_dict.keys()):
            if k in layers_False:
                del pretrained_dict[k]
        print(model.load_state_dict(pretrained_dict, strict=False))

    model_train = model.train()

    if Cuda:
        Generator_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        Generator_train = Generator_train.cuda()

    # opt_model = torch.optim.Adam(model.parameters(), lr=Init_lr)
    opt_model = torch.optim.AdamW(model.parameters(), lr=Init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    scheduler = CosineAnnealingLR(opt_model, T_max=5, eta_min=0)

    # ---------------------------------------#
    #   构建数据集加载器。
    # ---------------------------------------#
    train_dataset = Dataset(train_val_dataset_path, input_shape, epoch_length=Epoch, is_train=True, random_seed=random_seed)
    val_dataset = Dataset(train_val_dataset_path, input_shape, epoch_length=Epoch, is_train=False, random_seed=random_seed)

    shuffle = True

    train_gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)

    val_gen = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    total_epochs = []
    test_counter = [i * len(train_gen.dataset) for i in range(Epoch - Init_Epoch)]
    for epoch in range(Init_Epoch, Epoch):
        epoch_step = train_dataset.length // batch_size
        epoch_step_val = val_dataset.length // batch_size
        train_gen.dataset.epoch_now = epoch
        val_gen.dataset.epoch_now = epoch
        total_epochs.append(epoch)

        fit_one_epoch(model_train, model, opt_model, scheduler, epoch, epoch_step, epoch_step_val, train_gen, val_gen, Epoch, Cuda, save_period, save_dir, train_losses, train_acc, train_counter, test_losses, test_acc)
    
    draw_loss(train_counter, train_losses, test_counter, test_losses)
    draw_acc(total_epochs, train_acc, test_acc)
    
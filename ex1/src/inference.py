import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from utils.dataloader_cl import Dataset, dataset_collate
from nets.model import Baseline2

def show_examples(output, example_data):
    fig = plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.tight_layout()
        # 设置字体为楷体
        plt.rcParams['font.sans-serif'] = ['KaiTi']
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def get_testloader():
    train_val_dataset_path = 'dataset/NewDataset.mat'
    input_shape = [28, 28]
    random_seed = 3407
    batch_size = 128
    num_workers = 0

    test_dataset = Dataset(train_val_dataset_path, input_shape, epoch_length=1, is_train=False, random_seed=random_seed)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)

    return test_loader


def process():
    model = Baseline2()
    network_state_dict = torch.load('./checkpoints/baseline2_ep100.pth')
    model.load_state_dict(network_state_dict)

    test_loader = get_testloader()

    examples = enumerate(test_loader, start=100)
    batch_idx, (example_data, example_targets) = next(examples)
    with torch.no_grad():
        output = model(example_data)
    show_examples(output, example_data)

if __name__ == "__main__":
    process()
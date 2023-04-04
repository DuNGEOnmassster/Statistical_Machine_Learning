from random import sample, shuffle

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import scipy.io as scio
from utils.utils import preprocess_input


class Dataset(Dataset):
    def __init__(self, path, input_shape, is_train):
        super(Dataset, self).__init__()
        self.path               = path
        self.input_shape        = input_shape

        self.data               = scio.loadmat(path)
        self.is_train           = is_train

        if is_train:
            self.length         = np.shape(self.data['train'])[0] * 15
            self.data = self.data['train']
        else:
            self.length         = np.shape(self.data['test'])[0] * 5
            self.data = self.data['test']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        while(True):
            if self.is_train:
                index = index % self.length
                label = index // 15
                index = index % 15
            else:
                index = index % self.length
                label = index // 5
                index = index % 5
            if label == 121 or label == 122:
                continue
            else:
                break
        # print(label, index)
        image = self.data[label][index]
        # image = np.expand_dims(image, axis=2)
        image = preprocess_input(np.array(image, dtype=np.float32)).reshape(28*28)

        return image, label

    
# DataLoader中collate_fn使用
def dataset_collate(batch):
    images  = []
    Label = []
    for img, label in batch:
        images.append(img)
        Label.append(label)
            
    images  = images
    Label   = Label
    return images, Label

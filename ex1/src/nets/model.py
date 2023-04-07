import cv2
# import kornia
import numpy

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, if_maxpool=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.if_maxpool = if_maxpool

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = F.interpolate(identity, scale_factor=0.5)
        # print(f"x = {x.shape}")

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.maxpool(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if not self.downsample:
            out += identity

        out = self.relu(out)
        if self.if_maxpool:
            out = self.maxpool(out)

        # print(f"x view: {x.view(out.size(0), -1).shape}")

        # print(f"out = {out.shape}")

        return out

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 200)
        )

    def forward(self, x):  # size(x) == (B,1,28,28)
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # x = F.softmax(x, dim=1)
        return x

class Baseline2(nn.Module):
    def __init__(self):
        super(Baseline2, self).__init__()
        self.conv1 = BasicBlock(1, 16, downsample=True)
        self.conv2 = BasicBlock(16, 32, downsample=True)
        self.conv3 = BasicBlock(32, 64, downsample=True)
        self.classifier = nn.Linear(576, 200)
        # self.cls = BasicClassifier(576, 200)
    
    def forward(self, x):
        x = self.conv1(x)
        # print(f"conv1:{x.shape}")
        x = self.conv2(x)
        # print(f"conv2:{x.shape}")
        x = self.conv3(x)
        # print(f"conv3:{x.shape}")

        x = x.view(x.size(0), -1)
        # print(f"view x: {x.shape}")
        x = self.classifier(x)
        # x = self.cls(x)
        # print(f"classify: {x.shape}")
        # x = F.softmax(x, dim=1)
        # x = F.log_softmax(x, dim=1)
        # print(f"softmax: {x.shape}")
        return x


class BasicClassifier(nn.Module):
    def __init__(self, in_channel, out_channel, residual=True):
        super(BasicClassifier, self).__init__()
        self.cl1 = nn.Linear(in_channel, in_channel//2)
        self.cl2 = nn.Linear(in_channel//2, out_channel)
        self.cl3 = nn.Linear(in_channel, 16)
        self.cl4 = nn.Linear(16, in_channel//2)
        self.cl5 = nn.Linear(in_channel, out_channel)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward_normal(self, x):
        x = self.cl1(x)
        # print(f"cl1: {x.shape}")
        x = self.relu(x)
        # print(f"relu1: {x.shape}")
        x = self.cl2(x)
        # print(f"cl2: {x.shape}")
        return x

    def forward_residual(self, x):
        x1 = self.cl1(x)
        x1 = self.relu(x1)
        print(f"x1: {x1.shape}")
        x21 = self.cl3(x)
        x21 = self.relu(x21)
        print(f"x21: {x21.shape}")
        x22 = self.cl4(x21)
        x22 = self.relu(x22)
        print(f"x22: {x22.shape}")
        out = self.cl2(x1+x22)
        print(f"out: {out.shape}")
        return out

    def forward(self, x):
        if self.residual:
            x = self.forward_residual(x)
        else:
            x = self.forward_normal(x)
        return x
        
        


class Baseline3(nn.Module):
    def __init__(self):
        super(Baseline3, self).__init__()
        self.conv1 = BasicBlock(1, 16, downsample=True)
        self.conv2 = BasicBlock(16, 32, downsample=True)
        self.conv3 = BasicBlock(32, 32, downsample=False, if_maxpool=False)
        self.conv4 = BasicBlock(32, 64, downsample=True, if_maxpool=True)
        # self.conv5 = BasicBlock(128, 128, downsample=False, if_maxpool=False)
        self.dropout = nn.Dropout(0.5)
        self.classifier = BasicClassifier(576,200)

    def forward(self, x):
        x = self.conv1(x)
        # print(f"conv1:{x.shape}")
        x = self.conv2(x)
        # print(f"conv2:{x.shape}")
        x = self.conv3(x)
        # print(f"conv3:{x.shape}")
        x = self.conv4(x)
        # print(f"conv4:{x.shape}")
        # x = self.dropout(x)
        # print(f"dropout:{x.shape}")
        # x = self.conv5(x)
        # print(f"conv5:{x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"view x:{x.shape}")
        x = self.classifier(x)
        # print(f"cls x: {x.shape}")
        x = F.log_softmax(x, dim=1)
        return x



class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=200,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, 
                            stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        cond1 = x
        x = self.layer2(x)
        cond2 = x
        x = self.layer3(x)
        cond3 = x
        x = self.layer4(x)
        cond4 = x

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return cond1, x
    

def resnet34(num_classes=200, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

if __name__ == "__main__":
    model = Baseline3()
    input = torch.rand(16, 1, 28, 28)
    output = model(input)
    print(output.shape)
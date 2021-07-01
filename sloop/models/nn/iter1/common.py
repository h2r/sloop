import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONV1_PLAIN = 16
CONV1_KERNEL = 3
CONV2_PLAIN = 16
CONV2_KERNEL = 3
CONV3_PLAIN = 16
CONV3_KERNEL = 3
FOREF_L1 = 128
FOREF_L2 = 64
PR_L1 = 256
PR_L2 = 64
PR_L3 = 32

class MapImgCNN(nn.Module):
    def __init__(self, stride=1, map_dims=(21,21)):
        super(MapImgCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, CONV1_PLAIN,
                               kernel_size=CONV1_KERNEL,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(CONV1_PLAIN)
        self.max_pool1 = nn.MaxPool2d(CONV1_KERNEL, stride=stride, padding=1)
        
        self.conv2 = nn.Conv2d(CONV1_PLAIN, CONV2_PLAIN,
                               kernel_size=CONV2_KERNEL,
                               stride=stride, padding=1, bias=False)
        self.max_pool2 = nn.MaxPool2d(CONV2_KERNEL, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(CONV2_PLAIN)
        
        self.conv3 = nn.Conv2d(CONV2_PLAIN, CONV3_PLAIN,
                               kernel_size=CONV3_KERNEL,
                               stride=stride, padding=1, bias=False)
        self.max_pool3 = nn.MaxPool2d(CONV3_KERNEL, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(CONV3_PLAIN)
        
        self.input_size = map_dims[0]*map_dims[1]
        self.map_dims = map_dims

    def forward(self, x):
        x = x.view(-1, 1, self.map_dims[0], self.map_dims[1])
        out = self.max_pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.max_pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.max_pool3(F.relu(self.bn3(self.conv3(out))))
        out = out.view(out.shape[0], -1)
        return out


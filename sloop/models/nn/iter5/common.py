import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

CONV1_PLAIN = 16
CONV1_KERNEL = 3
CONV2_PLAIN = 16
CONV2_KERNEL = 3
CONV3_PLAIN = 16
CONV3_KERNEL = 3
HALF_FEAT_SIZE = 32
HALF_FEAT_SIZE2 = 32
FULL_FEAT_SIZE = 64
FULL_FEAT_SIZE2 = 64
L1 = 32
L2 = 32


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

class HalfModel(nn.Module):
    def __init__(self, map_dims=(21,21)):
        super(HalfModel, self).__init__()
        linear1 = nn.Linear(2, L1)
        linear2 = nn.Linear(L1, L2)
        self.layer_objloc = nn.Sequential(linear1,
                                          nn.ReLU(),
                                          linear2,
                                          nn.ReLU())
        self.layer_mapimg = MapImgCNN(map_dims=map_dims)
        self.layer_combine = nn.Sequential(nn.Linear(CONV3_PLAIN*map_dims[0]*map_dims[1]+L2, HALF_FEAT_SIZE),
                                           nn.ReLU(),
                                           nn.Linear(HALF_FEAT_SIZE, HALF_FEAT_SIZE2),
                                           nn.ReLU())  # tentative
        self.input_size = 2 + self.layer_mapimg.input_size

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x_objloc = x[:, :2]
        x_mapimg = x[:, 2:]

        out_objloc = self.layer_objloc(x_objloc)
        out_mapimg = self.layer_mapimg(x_mapimg)
        out = torch.cat([out_objloc, out_mapimg], 1)
        out = self.layer_combine(out)
        return out




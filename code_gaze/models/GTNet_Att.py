# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from caps import CapsuleLayer


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1) # 若卷积核大小为3x3, 那么就应该设定padding=1, 即填充1层边缘像素; 若卷积核大小为7x7, 那么就应该设定padding=3

        self.primary_capsules = CapsuleLayer(num_capsules=4, num_route_nodes=-1, in_channels=64, out_channels=4, kernel_size=3, stride=1)
        self.digit_capsules = CapsuleLayer(num_capsules=4, num_route_nodes=4 * 6 * 9, in_channels=4, out_channels=4)
        # num_capsules=8 数字胶囊的个数; num_route_nodes: 一个立方体的体积; in_channels=4 初始胶囊立方体个数; out_channels=4 数字胶囊的长度

        self.fc1 = nn.Linear(3584 + 0, 500) # 4x7x128 = 3600
        self.fc2 = nn.Linear(502, 2)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.01) # added by JZ Chen
        self.apply(initialize_weights)

    def forward(self, x, y):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv3(x), kernel_size=2, stride=2)
    # attention, added by JZ Chen
        z = self.conv4(x)
        z = F.sigmoid(z)
        z = z.repeat(1, 128, 1, 1)
        z = torch.mul(x, z)
        x = torch.add(z, x)
        # print(x.shape)
    # attention
    # CapsNet
    #     c = self.primary_capsules(x)
    #     c = self.digit_capsules(c)
    #     c = c.squeeze(0).transpose(0, 1)
    #     c = c.reshape(c.size(0), -1)
    #     x = x.reshape(x.size(0), -1)
    #     x = torch.cat([x, c], dim=-1)
        # print(x.shape)
    # CapsNet
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True)
        # print(x.shape)
        x = torch.cat([x, y], dim=1) # concated with pose
        x = self.fc2(x)
        return x
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

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)

        self.primary_capsules = CapsuleLayer(num_capsules=4, num_route_nodes=-1, in_channels=50, out_channels=4, kernel_size=5, stride=1)
        self.digit_capsules = CapsuleLayer(num_capsules=2, num_route_nodes=4 * 6 * 12, in_channels=4, out_channels=1)
        self.bn_x=nn.BatchNorm1d(3600)
        self.bn_c=nn.BatchNorm1d(2)
        self.fc1 = nn.Linear(3600 + 2, 500) # 6x12x50 = 3600
        self.fc2 = nn.Linear(502, 2)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def forward(self, x, y):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        # CapsNet
        c = self.primary_capsules(x)
        c = self.digit_capsules(c)
        c = c.squeeze(0).transpose(0, 1)
        c = c.reshape(c.size(0), -1)
        x = x.reshape(x.size(0), -1)
        # x=self.bn_x(x)
        # c=self.bn_c(c)
        x = torch.cat([x, c], dim=-1)
        # print(x.shape)
        # CapsNet
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True)
        x = torch.cat([x, y], dim=1) # concated with pose
        x = self.fc2(x)
        return x
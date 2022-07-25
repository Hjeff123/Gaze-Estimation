# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
# from caps import CapsuleLayer
import numpy as np

d = 2


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.constant_(module.bias, 0)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False))

    def forward(self, x):
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(x)
        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)
        y += self.shortcut(x)
        return y


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        depth = 8
        base_channels = 16
        input_shape = (1, 3, 448, 448)
        d2 = int(d * (d + 1) / 2)
        n_blocks_per_stage = (depth - 2) // 6
        assert n_blocks_per_stage * 6 + 2 == depth

        n_channels = [base_channels, base_channels * 2, base_channels * 4]

        self.conv = nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)

        self.stage1 = self._make_stage(
            n_channels[0],
            n_channels[0],
            n_blocks_per_stage,
            BasicBlock,
            stride=1)
        self.stage2 = self._make_stage(
            n_channels[0],
            n_channels[1],
            n_blocks_per_stage,
            BasicBlock,
            stride=2)
        self.stage3 = self._make_stage(
            n_channels[1],
            n_channels[2],
            n_blocks_per_stage,
            BasicBlock,
            stride=2)
        self.bn = nn.BatchNorm2d(n_channels[2])

        # self.primary_capsules = CapsuleLayer(num_capsules=4, num_route_nodes=-1, in_channels=64, out_channels=4, kernel_size=5, stride=1)
        # self.digit_capsules = CapsuleLayer(num_capsules=4, num_route_nodes=4 * 4 * 7, in_channels=4, out_channels=4)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)  # 若卷积核大小为3x3, 那么就应该设定padding=1, 即填充1层边缘像素; 若卷积核大小为7x7, 那么就应该设定padding=3
        # num_capsules=8 数字胶囊的个数; num_route_nodes: 一个立方体的体积; in_channels=4 初始胶囊立方体个数; out_channels=4 数字胶囊的长度
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv_att1 = nn.Conv2d(64, d, kernel_size=3, padding=1)
        self.conv_att2 = nn.Conv2d(d2, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 + 0, 64)  # num_capsules * out_channels
        self.fc2 = nn.Linear(64, 2)
        self.layer = nn.BatchNorm2d(3)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.apply(initialize_weights)

    def _initialize_weight(self):
        nn.init.normal_(self.conv4.weight, mean=0, std=0.01)  # added by JZ Chen
        self.apply(initialize_weights)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = 'block{}'.format(index + 1)
            if index == 0:
                stage.add_module(
                    block_name, block(in_channels, out_channels, stride=stride))
            else:
                stage.add_module(block_name, block(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.relu(self.bn(x), inplace=True)
        # x = F.adaptive_avg_pool2d(x, output_size=1)
        return x

    def forward(self, x, y):
        x = self.layer(x)
        x = self._forward_conv(x)  # (bs,64,112,112)
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv3(x), kernel_size=2, stride=2)
# bilinear pooling begin
        batch_size = x.shape[0]
        d2 = int(d * (d + 1) / 2)
        Y0 = self.conv_att1(x)
        Y1 = Y0.permute(0, 2, 3, 1).unsqueeze(3)  # (64,9,15,1,64)
        Y2 = Y0.permute(0, 2, 3, 1).unsqueeze(4)  # (64,9,15,64,1)
        Y = Y1.mul(Y2)  # torch.Size([64, 9, 15, 64, 64]) # matrix mul
        index = torch.triu(torch.ones((batch_size, 12, 12, d, d)) == 1)
        z = Y[index]
        z = z.reshape((batch_size, 12, 12, d2))
        z = z.permute(0, 3, 1, 2)
        A = self.conv_att2(z)  # torch.Size([64, 64, 9, 15])
# AiA
        G = self.avg(A)  # torch.Size([64, 64, 1, 1])
        A = torch.mul(A, G) + A  # torch.Size([64, 64, 9, 15])
# AiA
        D = F.softmax(A, dim=1)
        x = torch.mul(x, D) + x  # torch.Size([64, 64, 9, 15]) # element mul
# bilinear pooling endorch.cat([x, c], dim=1)
        x = F.adaptive_avg_pool2d(x, output_size=1)  # torch.Size([64, 64, 1, 1])
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True)  # torch.Size([64, 62])
        # x = torch.cat([x, y], dim=1)  # concated with pose   torch.Size([64, 64])
        x = self.fc2(x)  # torch.Size([64, 2])
        return x
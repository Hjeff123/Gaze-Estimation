# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

d = 2

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        d2 = int(d * (d + 1) / 2)

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) # padding = (kernel_size-1)/2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv_att1 = nn.Conv2d(64, d, kernel_size=3, stride=1, padding=1)
        self.conv_att2 = nn.Conv2d(d2, 64, kernel_size=3, stride=1, padding=1)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(12544, 512)  # 14x14x64 = 12544
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.BatchNorm2d = nn.BatchNorm2d(3)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1_1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv2_1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv3_1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv4_1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv5_1.weight, mean=0, std=0.01)
        self.apply(initialize_weights)

    def forward(self, x, y):
        x = self.BatchNorm2d(x)
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv4_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5_1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
# # bilinear pooling begin
        batch_size = x.shape[0]
        d2 = int(d * (d + 1) / 2)
        Y0 = self.conv_att1(x)
        Y1 = Y0.permute(0, 2, 3, 1).unsqueeze(3)  # (64,9,15,1,64)
        Y2 = Y0.permute(0, 2, 3, 1).unsqueeze(4)  # (64,9,15,64,1)
        Y = Y1.mul(Y2)  # torch.Size([64, 9, 15, 64, 64]) # matrix mul
        index = torch.triu(torch.ones((batch_size, 14, 14, d, d)) == 1)
        z = Y[index]
        z = z.reshape((batch_size, 14, 14, d2))
        z = z.permute(0, 3, 1, 2)
        A = self.conv_att2(z)  # torch.Size([64, 64, 9, 15])
# # AiA
        G=self.avg(A)     #torch.Size([64, 64, 1, 1])
        A=torch.mul(A, G) + A  #torch.Size([64, 64, 9, 15])
# # AiA
#         D = F.softmax(A, dim=1)
        D = F.sigmoid(A)
        x = torch.mul(x, D) + x  # torch.Size([64, 64, 9, 15]) # element mul
# # bilinear pooling end
        x = F.relu(self.fc1(x.view(x.size(0), -1)), inplace=True)
        x = F.relu(self.fc2(x))
        # x = torch.cat([x, y], dim=1)  # concated with pose
        x = self.fc3(x)
        return x, A, D
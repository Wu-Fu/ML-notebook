import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# SE层，用于引入注意力机制
class SE(nn.Module):
    pass


class BottleNeck(nn.Module):
    pass


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 因为残差块中存在着shortcut路径，而且使用了batchnorm，所以conv中bias的作用被抵消，所以将bias设置为False
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 当stride ！= 1 或 输入输出通道数不同时，加入一个1*1卷积块来作为shortcut旁路
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(X)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block_type, groups, nums_class):
        super(ResNet, self).__init__()
        self.block_type = block_type
        self.channels = 64
        self.conv1 = nn.Conv2d(3, out_channels=self.channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x = self._make_layer(64, groups[0], 1, 2)
        self.conv3_x = self._make_layer(128, groups[1], 2, 3)
        self.conv4_x = self._make_layer(256, groups[2], 2, 4)
        self.conv5_x = self._make_layer(512, groups[3], 2, 5)
        self.pool2 = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(512, nums_class) if block_type == 'BasicBlock' else nn.Linear(512 * 4, nums_class)

    def forward(self, X):
        out = F.relu(self.bn1(self.conv1(X)))
        out = self.pool1(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out

    # 批量生成残差块
    def _make_layer(self, channels, blocks, stride, index):
        # 除了第一层之外，conv层的步幅都是1
        list_strides = [stride] + [1] * blocks - 1
        conv_x = nn.Sequential()
        block_map = {'BasicBlock': ResidualBlock, 'BottleNeck': BottleNeck}
        block = block_map[self.block_type]
        for i in range(len(list_strides)):
            layer_name = str("block_%d_%d" % (index, i))
            conv_x.add_module(layer_name, block(self.channels, channels, list_strides[i]))
            self.channels = channels if self.block_type == 'BasicBlock' else channels * 4
        return conv_x

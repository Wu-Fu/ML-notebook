import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# SE层，用于引入注意力机制
class SE(nn.Module):
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
    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self):
        pass

import torch
import torch.nn.functional as F
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.relu = nn.LeakyReLU(negative_slope=0.04)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, padding_mode='replicate')
        self.bn1 = nn.Identity()#nn.BatchNorm2d(out_channels) #batchnorm currently disabled
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, padding_mode='replicate')
        self.bn2 = nn.Identity()#nn.BatchNorm2d(out_channels) #batchnorm currently disabled
 
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.04)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.04)
        return x
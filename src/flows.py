import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ConvBlock
from util import warp

class Damper(nn.Module):
    def __init__(self, const):
        super(Damper, self).__init__()
        self.constant = const

    def forward(self, x):
        return x*self.constant

class FlowAccumulator(nn.Module):
    def __init__(self, base_channels, device):
        super(FlowAccumulator, self).__init__()
        damping_factor = 0.01
        self.base_channels = base_channels
        self.device = device

        self.flow_block1 = nn.Sequential(ConvBlock(in_channels=self.base_channels*2, out_channels=self.base_channels),
                                         nn.Conv2d(in_channels=self.base_channels, out_channels=2, kernel_size=1, padding_mode='replicate'),
                                         Damper(damping_factor))
        
        self.flow_block2 = nn.Sequential(ConvBlock(in_channels=self.base_channels*6, out_channels=self.base_channels),
                                         nn.Conv2d(in_channels=self.base_channels, out_channels=2, kernel_size=1, padding_mode='replicate'),
                                         Damper(damping_factor))
        
        
        self.flow_block_shared = nn.Sequential(ConvBlock(in_channels=self.base_channels*14, out_channels=self.base_channels),
                                               nn.Conv2d(in_channels=self.base_channels, out_channels=2, kernel_size=1, padding_mode='replicate'),
                                               Damper(damping_factor))

    def _bi_directional_flow_block(self, f1, f2, accumulated_flow1, accumulated_flow2, conv_block):
        warped_f2 = warp(f2, accumulated_flow1)
        warped_f1 = warp(f1, accumulated_flow2)
        r1 = conv_block(torch.concat((f1, warped_f2), dim=1))
        r2 = conv_block(torch.concat((f2, warped_f1), dim=1))
        accumulated_flow1 += r1
        accumulated_flow2 += r2
        return (accumulated_flow1, accumulated_flow2)
    
    def forward(self, feature_pyramid1, feature_pyramid2):
        B, _, H, W = feature_pyramid1[0].size()
        accumulated_flow1 = torch.zeros((B, 2, H, W)).to(self.device)
        accumulated_flow2 = torch.zeros((B, 2, H, W)).to(self.device)

        flows_1 = []
        flows_2 = []
        conv_blocks = [self.flow_block_shared, self.flow_block_shared, self.flow_block_shared, self.flow_block2, self.flow_block1]
        for f1, f2, conv_block in zip(feature_pyramid1, feature_pyramid2, conv_blocks):
            accumulated_flow1, accumulated_flow2 = self._bi_directional_flow_block(f1, f2, accumulated_flow1, accumulated_flow2, conv_block)
            flows_1.append(accumulated_flow1)
            flows_2.append(accumulated_flow2)
            accumulated_flow1 = F.interpolate(accumulated_flow1, scale_factor=2, mode='bilinear', align_corners=True)
            accumulated_flow2 = F.interpolate(accumulated_flow2, scale_factor=2, mode='bilinear', align_corners=True)
        
        return (flows_1, flows_2)
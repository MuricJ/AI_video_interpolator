import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import ConvBlock

class EncoderFILM(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super(EncoderFILM, self).__init__()
        
        self.encoder11 = ConvBlock(in_channels, base_channels) 
        self.encoder12 = ConvBlock(in_channels, base_channels)
        self.encoder13 = ConvBlock(in_channels, base_channels) 
        self.encoder14 = ConvBlock(in_channels, base_channels) 
        self.encoder15 = ConvBlock(in_channels, base_channels)

        self.encoder2 = ConvBlock(base_channels, base_channels * 2)
        self.encoder3 = ConvBlock(base_channels * 2, base_channels * 4)

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=0.5, mode="bilinear", align_corners=True)
        x3 = F.interpolate(x1, scale_factor=0.25, mode="bilinear", align_corners=True)
        x4 = F.interpolate(x1, scale_factor=0.125, mode="bilinear", align_corners=True)
        x5 = F.interpolate(x1, scale_factor=0.0625, mode="bilinear", align_corners=True)

        enc11 = self.encoder11(x1)   #B
        enc12 = self.encoder12(x2)   #B
        enc13 = self.encoder13(x3)   #B
        enc14 = self.encoder14(x4)   #B
        enc15 = self.encoder15(x5)   #B

        enc21 = self.encoder2(F.max_pool2d(enc11, kernel_size=2))  #2B
        enc22 = self.encoder2(F.max_pool2d(enc12, kernel_size=2))  #2B
        enc23 = self.encoder2(F.max_pool2d(enc13, kernel_size=2))   #2B
        enc24 = self.encoder2(F.max_pool2d(enc14, kernel_size=2))   #2B

        enc31 = self.encoder3(F.max_pool2d(enc21, kernel_size=2))  #4B
        enc32 = self.encoder3(F.max_pool2d(enc22, kernel_size=2))  #4B
        enc33 = self.encoder3(F.max_pool2d(enc23, kernel_size=2))   #4B
 
        feat1 = enc11 #B
        feat2 = torch.concat((enc21, enc12), dim=1) #3B
        feat3 = torch.concat((enc31, enc22, enc13), dim=1) #7B
        feat4 = torch.concat((enc32, enc23, enc14), dim=1) #7B
        feat5 = torch.concat((enc33, enc24, enc15), dim=1) #7B

        return (feat5, feat4, feat3, feat2, feat1)
    

class Decoder(nn.Module):
    def __init__(self, base_channels=16, out_channels=3):
        super(Decoder, self).__init__()
        
        self.decoder_with_flow = nn.ModuleList([
            ConvBlock(in_channels=14 * base_channels+4, out_channels=14 * base_channels),
            ConvBlock(in_channels=28 * base_channels+4, out_channels=14 * base_channels),
            ConvBlock(in_channels=28 * base_channels+4, out_channels=6 * base_channels),
            ConvBlock(in_channels=12 * base_channels+4, out_channels=2 * base_channels),
            nn.Conv2d(in_channels=4 * base_channels+4, out_channels=out_channels, kernel_size=1, padding_mode='replicate')
        ])

        self.decoder_with_feature_skip = nn.ModuleList([
                ConvBlock(in_channels=28 * base_channels+4, out_channels=28 * base_channels),
                ConvBlock(in_channels=56 * base_channels+4, out_channels=28 * base_channels),
                ConvBlock(in_channels=56 * base_channels+4, out_channels=12 * base_channels),
                ConvBlock(in_channels=24 * base_channels+4, out_channels=4 * base_channels),
                nn.Conv2d(in_channels=8 * base_channels+4, out_channels=out_channels, kernel_size=1, padding_mode='replicate')
        ])

        self.decoder_single_pyramid = nn.ModuleList([
            ConvBlock(in_channels=7 * base_channels, out_channels=7 * base_channels),
            ConvBlock(in_channels=14 * base_channels, out_channels=7 * base_channels),
            ConvBlock(in_channels=14 * base_channels, out_channels=3 * base_channels),
            ConvBlock(in_channels=6 * base_channels, out_channels=1 * base_channels),
            nn.Conv2d(in_channels=2 * base_channels, out_channels=out_channels, kernel_size=1, padding_mode='replicate')
        ])
        
        self.decoder = self.decoder_with_feature_skip

    def forward(self, feats):
        feat5, feat4, feat3, feat2, feat1 = feats

        out = self.decoder[0](feat5)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

        out = self.decoder[1](torch.concat((out, feat4), dim=1))
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

        out = self.decoder[2](torch.concat((out, feat3), dim=1))
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

        out = self.decoder[3](torch.concat((out, feat2), dim=1))
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)

        out = self.decoder[4](torch.concat((out, feat1), dim=1))

        return out
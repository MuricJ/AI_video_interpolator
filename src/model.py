import torch.nn as nn
from encoder_decoder import EncoderFILM, Decoder
from blocks import ResBlock
from flows import FlowAccumulator
from util import warp_pyramids, concatenate_pyramids, avg_pyramids, warp, mul_pyramids
from visual import visualize_flow_layers, show_images2

class FrameInterpolator(nn.Module):
    def __init__(self, device):
        super(FrameInterpolator, self).__init__()
        self.base_channels = 24
        self.encoder = EncoderFILM(base_channels=self.base_channels, in_channels=3)
        self.decoder = Decoder(base_channels=self.base_channels, out_channels=3)
        self.flow_accumulator = FlowAccumulator(base_channels=self.base_channels, device=device)
        self.refiner = ResBlock(in_channels=3)

    def forward(self, frame1, frame2):
        feature_pyramid1 = self.encoder(frame1)
        feature_pyramid2 = self.encoder(frame2)

        flows_1, flows_2 = self.flow_accumulator(feature_pyramid1, feature_pyramid2)
        warped_feature_pyramid1 = warp_pyramids(feature_pyramid1, flows_1, -0.5)
        warped_feature_pyramid2 = warp_pyramids(feature_pyramid2, flows_2, -0.5)

        con_pyramids = concatenate_pyramids((warped_feature_pyramid1,
                                             warped_feature_pyramid2,
                                             feature_pyramid1,
                                             feature_pyramid2, 
                                             mul_pyramids(flows_1, -0.5), 
                                             mul_pyramids(flows_2, -0.5)))

        out = self.decoder(con_pyramids)

        #visualize_flow_layers(flows_1 + flows_2)
        #show_images2([warp(frame1, flows_2[-1])[0],frame2[0],frame1[0],warp(frame2, flows_1[-1])[0]])

        return (out, flows_1, flows_2, feature_pyramid1, feature_pyramid2)



        
        







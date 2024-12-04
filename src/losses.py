from util import gram_matrix, warp
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class MobileNetV3PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(MobileNetV3PerceptualLoss, self).__init__()
        
        mobilenet_v3 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2).features
        
        self.layers = [2,3,4] 
        self.max_layer = max(self.layers) 
        self.feature_extractor = nn.ModuleList(mobilenet_v3[:self.max_layer + 1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.device = device
        self.feature_extractor.to(self.device)


    def transform(self, img):
        img = transforms.functional.resize(img, [224, 224])
        img = transforms.functional.convert_image_dtype(img, torch.float)
        img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img
    
    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        
        input_features = self.transform(input)
        target_features = self.transform(target)
        style_loss = 0.0
        perceptual_loss = 0.0
        for i, layer in enumerate(self.feature_extractor):
            input_features = layer(input_features)
            target_features = layer(target_features)
            if i in self.layers:
                style_loss += F.mse_loss(gram_matrix(input_features),gram_matrix(target_features))/len(self.layers)
                perceptual_loss += F.l1_loss(input_features, target_features)/len(self.layers)
        
        return (perceptual_loss, style_loss)


# perceptual loss device is set when loss functoin is first called
# to reset device, set percLoss.mv3 = None
def percLoss(pred_frame, true_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2):
    if not hasattr(percLoss, "mv3") or percLoss.mv3 == None:
        percLoss.mv3 = MobileNetV3PerceptualLoss(pred_frame.device)
    perceptual_loss, style_loss = percLoss.mv3(pred_frame, true_frame)
    L1_loss = F.l1_loss(pred_frame, true_frame)
    L1_w = 1.0
    perc_w = 0.003
    style_w = 15.0

    L1_loss, perceptual_loss, style_loss = L1_w*L1_loss, perc_w*perceptual_loss, style_w*style_loss
    print(f"\rL1: {L1_loss.item():.12f} | Perceptual: {perceptual_loss.item():.12f} | Style: {style_loss.item():.12f}", end="")
    return L1_loss + perceptual_loss + style_loss

def l1_photometric_loss(flow_pyramid_1, flow_pyramid_2, feature_pyramid1, feature_pyramid2):
    total_loss = 0.0
    for flow_1, flow_2, feat1, feat2 in zip(flow_pyramid_1, flow_pyramid_2, feature_pyramid1, feature_pyramid2):
        warped_feat2 = warp(feat2, flow_1)
        warped_feat1 = warp(feat1, flow_2)
        loss_1 = F.l1_loss(warped_feat2, feat1)
        loss_2 = F.l1_loss(warped_feat1, feat2)
        total_loss += loss_1 + loss_2
    return total_loss / len(flow_pyramid_1)

def L1_image_L1_photometric(pred_frame, true_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2):
    image_w = 1.0
    photometric_w = 0.03
    imageL1 = image_w*F.l1_loss(pred_frame, true_frame)
    photometricL1 = photometric_w*l1_photometric_loss(flows_1, flows_2, feature_pyramid1, feature_pyramid2)
    print(imageL1, photometricL1)
    return(imageL1+photometricL1)

def L1_only(pred_frame, true_frame, flows_1, flows_2, feature_pyramid1, feature_pyramid2):
    return F.l1_loss(pred_frame, true_frame)

import torch
import torch.nn.functional as F

DENORM_MEAN = 0.5
DENORM_STD = 0.5

def gram_matrix(features):
    B, C, H, W = features.size()
    features = features.view(B, C, H * W)
    gram = torch.bmm(features, features.transpose(1, 2))
    gram = gram / (C * H * W)
    return gram

def warp(img, flow):
    B, C, H, W = img.size()
    grid_x, grid_y = torch.meshgrid(torch.arange(0, W), torch.arange(0, H), indexing='xy')
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).float()
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1).to(img.device)
    grid_flow = grid + flow.permute(0, 2, 3, 1) 
    return F.grid_sample(img, grid_flow, mode='bilinear', padding_mode='border', align_corners=True)

def PSNR(x, target):
    flat_x = torch.flatten(x, start_dim=1)
    flat_target = torch.flatten(target, start_dim=1)
    return -10*torch.log10(torch.mean(torch.square(flat_target-flat_x)))

def denormalize(tensor):
    mean = torch.tensor(DENORM_MEAN).reshape(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(DENORM_STD).reshape(1, -1, 1, 1).to(tensor.device)
    return tensor * std + mean

def warp_pyramids(feature_pyramid, flows, factor):
    feature_pyramid_warped = []
    for f, flow in zip(feature_pyramid, flows):
        feature_pyramid_warped.append(warp(f, factor*flow))
    return feature_pyramid_warped
    
def concatenate_pyramids(pyramids, dim=1):
    concatenated_pyramid = []
    for p in zip(*pyramids):
        concatenated_pyramid.append(torch.concat(p, dim=dim))
    return concatenated_pyramid

def avg_pyramids(pyramid1, pyramid2):
    avg_pyramid = []
    for p1, p2 in zip(pyramid1, pyramid2):
        avg_pyramid.append((p1 + p2)/2)
    return avg_pyramid

def mul_pyramids(feature_pyramid, factor):
    feature_pyramid_mul = []
    for f in feature_pyramid:
        feature_pyramid_mul.append(f*factor)
    return feature_pyramid_mul
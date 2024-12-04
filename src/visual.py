import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import torch
from util import denormalize

def flow_to_image(flow):
    flow_u = flow[..., 0]
    flow_v = flow[..., 1]

    H, W = flow_u.shape

    magnitude = np.sqrt(flow_u**2 + flow_v**2)
    angle = np.arctan2(flow_v, flow_u)

    hue = (angle + np.pi) / (2 * np.pi)

    mag_norm = np.clip(magnitude, 0, 1)

    hsv_image = np.stack((hue, np.ones_like(hue), mag_norm), axis=-1)
    rgb_image = hsv_to_rgb(hsv_image)*6

    high_magnitude_mask = magnitude > 1

    rgb_image[high_magnitude_mask] = [1, 1, 1]

    return (rgb_image * 255).astype(np.uint8)

def visualize_flow_layers(flows):
    plt.figure(figsize=(20, 8))

    for i, flow in enumerate(flows[:5]):
        flow_np = flow[0].permute(1, 2, 0).cpu().detach().numpy() 
        flow_image = flow_to_image(flow_np)

        plt.subplot(2, 5, i + 1)
        plt.imshow(flow_image)
        plt.title(f"Flow Layer {i + 1}")
        plt.axis('off')

    for i, flow in enumerate(flows[5:10]):
        flow_np = flow[0].permute(1, 2, 0).cpu().detach().numpy()
        flow_image = flow_to_image(flow_np)

        plt.subplot(2, 5, i + 6)
        plt.imshow(flow_image)
        plt.title(f"Flow Layer {i + 6}")
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=True)

def show_images2(tensor_images, titles=None, ncols=5, figsize=(15, 10)):
    n_images = len(tensor_images)
    nrows = (n_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, image_tensor in enumerate(tensor_images):
        image = image_tensor
        image = denormalize(image).detach().cpu().numpy()
        image = image.squeeze(0)
        image = image.transpose(1,2,0)
        axes[i].imshow(image)
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis("off")
    
    for i in range(n_images, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show(block=True)
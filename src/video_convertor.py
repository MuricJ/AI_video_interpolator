from moviepy import VideoFileClip
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
import numpy as np
import gc
import tempfile
import shutil
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from util import denormalize

def video_to_frames(video_path, buffer_size=100):
    clip = VideoFileClip(video_path)
    frames = []
    for i, frame in enumerate(clip.iter_frames(fps=clip.fps, dtype="uint8")):
        frames.append(Image.fromarray(frame))
        if len(frames) == buffer_size:
            yield frames, clip.fps
            frames = []
    if frames:
        yield frames, clip.fps
    clip.close()

def frames_to_video(frames, output_path, fps=24):
    frame_arrays = [frame.convert("RGB") for frame in frames]
    frame_arrays = [np.array(frame) for frame in frame_arrays]
    clip = ImageSequenceClip(frame_arrays, fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False)

def decompose_frame(frame, patch_size=224):
    width, height = frame.size
    padded_width = (width + patch_size - 1) // patch_size * patch_size
    padded_height = (height + patch_size - 1) // patch_size * patch_size
    padded_frame = Image.new("RGB", (padded_width, padded_height))
    padded_frame.paste(frame, (0, 0))
    patches = []
    for i in range(0, padded_height, patch_size):
        for j in range(0, padded_width, patch_size):
            patch = padded_frame.crop((j, i, j + patch_size, i + patch_size))
            patches.append(patch)
    return patches

def reassemble_frame(patches, original_resolution, patch_size=224):
    original_width, original_height = original_resolution
    num_patches_x = (original_width + patch_size - 1) // patch_size
    num_patches_y = (original_height + patch_size - 1) // patch_size
    reassembled_frame = Image.new("RGB", (num_patches_x * patch_size, num_patches_y * patch_size))
    idx = 0
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            reassembled_frame.paste(patches[idx], (j * patch_size, i * patch_size))
            idx += 1
    return reassembled_frame.crop((0, 0, original_width, original_height))

def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu()
    if tensor.ndim == 2:
        array = tensor.numpy()
        return Image.fromarray((array * 255).astype(np.uint8))
    elif tensor.ndim == 3:
        if tensor.shape[0] in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        return Image.fromarray((array * 255).astype(np.uint8))
    else:
        raise ValueError("Unsupported tensor shape for image conversion.")

def videoconv_whole(model, model_name, video_path, device, mytransform, speed=0.5, buffer_size=100):
    print(f"Processing video: {video_path}")
    output_path = f"./OUT_videos/{model_name}_{os.path.basename(video_path)}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    writer = None
    prev_frame = None
    model.eval()
    for framebuffer, fps in video_to_frames(video_path, buffer_size):
        for i, frame in enumerate(framebuffer):
            print(f"Processing frame in buffer {i}/{buffer_size}\r")
            if writer is None:
                outsize = tuple(mytransform(frame)[0].size())
                writer = FFMPEG_VideoWriter(output_path, size=outsize, fps=2*speed*fps, codec="libx264")
            if prev_frame is not None:
                frame1 = mytransform(prev_frame).to(device).unsqueeze(0)
                frame2 = mytransform(frame).to(device).unsqueeze(0)

                with torch.no_grad():
                    pred_frame, *_ = model(frame1, frame2)
                pred_frame = denormalize(pred_frame).squeeze(0)

                interpolated_frame = tensor_to_pil(pred_frame)
                orig_frame = tensor_to_pil(denormalize(mytransform(prev_frame)).squeeze(0))
                writer.write_frame(np.array(orig_frame.convert("RGB")))
                writer.write_frame(np.array(interpolated_frame.convert("RGB")))
                
                gc.collect()
            
            prev_frame = frame
            
        if prev_frame is not None: 
            writer.write_frame(np.array(tensor_to_pil(denormalize(mytransform(prev_frame)).squeeze(0)).convert("RGB")))

    writer.close()
    gc.collect()
    print(f"Video processing complete. Output saved to: {output_path}")
    

def generate_video_samples(model, device, transform, path="./video_samples"):
    model_name = "L1Perc"
    os.makedirs("./OUT_videos", exist_ok=True)

    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Processing file: {file_path}")
            videoconv_whole(model, model_name, file_path, device, transform, speed=1.0, buffer_size=10)


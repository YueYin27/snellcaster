from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np
import torch
from PIL import Image
import os
from pathlib import Path

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

# Base directory containing all generated image folders
base_dir = "results/scenes/artroom_771/generated"

# Find all folders and load images
images = []
folder_names = []

for folder_name in sorted(os.listdir(base_dir)):
    folder_path = os.path.join(base_dir, folder_name)
    if os.path.isdir(folder_path):
        image_path = os.path.join(folder_path, "artroom_771_main.png")
        if os.path.exists(image_path):
            # Load image
            image_pil = Image.open(image_path)
            image = np.array(image_pil).astype(np.float32) / 255.0  # Convert to float [0, 1] range
            images.append(image)
            folder_names.append(folder_name)
            print(f"Loaded: {folder_name}")

# Stack all images into a batch
images = np.stack(images, axis=0)
print(f"\nTotal images loaded: {len(images)}")

# Same prompt for all images
prompt = "An art classroom with a rectangular wooden worktable near the camera, with a glass ball on top, surrounded by easels, color-splattered stools, sketches pinned on the walls, jars of brushes on shelves, and warm daylight pouring through wide windows."

# Calculate CLIP score for each image
print("\nCLIP scores for each image:")
print("-" * 80)
for i, (folder_name, image) in enumerate(zip(folder_names, images)):
    single_image = np.expand_dims(image, axis=0)  # Add batch dimension
    clip_score_value = calculate_clip_score(single_image, [prompt])
    print(f"{folder_name}: {clip_score_value}")

# Calculate overall CLIP score for all images
prompts = [prompt] * len(images)
overall_clip_score = calculate_clip_score(images, prompts)
print("-" * 80)
print(f"\nOverall CLIP score: {overall_clip_score}")

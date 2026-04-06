#!/usr/bin/env python3
"""
Script to combine masks and apply them to ground truth images.

Input: A folder containing:
    - *_fgmask.png
    - *_mask.png
    - *_gt.png

Process:
    1. Invert *_mask.png
    2. Add inverted mask to *_fgmask.png
    3. Use combined mask to mask *_gt.png
    4. Save as *_maskedgt.png
"""

import os
import sys
import glob
import numpy as np
from PIL import Image


def find_files(folder_path):
    """Find the three required image files in the folder."""
    fgmask_files = glob.glob(os.path.join(folder_path, "*_fgmask.png"))
    mask_files = glob.glob(os.path.join(folder_path, "*_mask.png"))
    gt_files = glob.glob(os.path.join(folder_path, "*_gt.png"))
    
    if not fgmask_files:
        raise FileNotFoundError(f"No *_fgmask.png file found in {folder_path}")
    if not mask_files:
        raise FileNotFoundError(f"No *_mask.png file found in {folder_path}")
    if not gt_files:
        raise FileNotFoundError(f"No *_gt.png file found in {folder_path}")
    
    # Get the base name from the first file to match all three
    base_name = os.path.basename(fgmask_files[0]).replace("_fgmask.png", "")
    
    fgmask_path = os.path.join(folder_path, f"{base_name}_fgmask.png")
    mask_path = os.path.join(folder_path, f"{base_name}_mask.png")
    gt_path = os.path.join(folder_path, f"{base_name}_gt.png")
    
    # Verify all files exist
    for path, name in [(fgmask_path, "_fgmask.png"), (mask_path, "_mask.png"), (gt_path, "_gt.png")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    return fgmask_path, mask_path, gt_path, base_name


def process_masks(folder_path):
    """Process masks and apply to ground truth image."""
    # Find the required files
    fgmask_path, mask_path, gt_path, base_name = find_files(folder_path)
    
    # Load images
    print(f"Loading images from {folder_path}...")
    fgmask = Image.open(fgmask_path)
    mask = Image.open(mask_path)
    gt = Image.open(gt_path)
    
    # Convert to numpy arrays for processing
    fgmask_array = np.array(fgmask)
    mask_array = np.array(mask)
    gt_array = np.array(gt)
    gt_array = gt_array[:, :, :3]

    # Process the foreground mask to be 1 channel and binary (0 or 255)
    if len(mask_array.shape) == 3:
        # If already 3 channels, convert to 1 channel (take first channel)
        mask_array = mask_array[:, :, 0]
    # Threshold: if value > 0, set to 255, else 0
    mask_array = np.where(mask_array > 0, 255, 0).astype(np.uint8)
    print(f"mask_array unique values: {np.unique(mask_array)}")
    print(f"mask_array shape: {mask_array.shape}")

    # Normalize masks to 0-1 range if they're not already
    if fgmask_array.max() > 1:
        fgmask_array = fgmask_array.astype(np.float32) / 255.0
    if mask_array.max() > 1:
        mask_array = mask_array.astype(np.float32) / 255.0

    # Handle grayscale vs RGB masks
    if len(fgmask_array.shape) == 2:
        fgmask_array = fgmask_array[:, :, np.newaxis]
    if len(mask_array.shape) == 2:
        mask_array = mask_array[:, :, np.newaxis]
    
    
    
    # Invert the mask (1 - mask)
    print("Inverting mask...")
    inverted_mask = 1.0 - fgmask_array
    
    # Add inverted mask to fgmask (combine masks)
    print("Combining masks...")
    combined_mask = mask_array + inverted_mask
    # Clamp to [0, 1]
    combined_mask = np.clip(combined_mask, 0.0, 1.0)

    print(f"mask_array unique values: {np.unique(mask_array)}")
    print(f"fgmask_array unique values: {np.unique(fgmask_array)}")

    # Save combined mask as a PNG image
    # Convert to uint8 and squeeze singleton dimensions for grayscale
    # combined_mask_2d = (combined_mask * 255.0).astype(np.uint8)
    # if len(combined_mask_2d.shape) == 3:
    #     combined_mask_2d = combined_mask_2d[:, :, 0]  # Take first channel if 3D
    # combined_mask_image = Image.fromarray(combined_mask_2d, mode='L')
    # combined_mask_path = os.path.join(folder_path, f"{base_name}_combined_mask.png")
    # combined_mask_image.save(combined_mask_path)
    # print(f"Saved combined mask to: {combined_mask_path}")
    
    # Expand mask to match GT image channels if needed
    if len(gt_array.shape) == 3:
        if combined_mask.shape[2] == 1:
            combined_mask = np.repeat(combined_mask, gt_array.shape[2], axis=2)
        elif combined_mask.shape[2] != gt_array.shape[2]:
            # Use only the first channel of the mask for all channels
            combined_mask = np.repeat(combined_mask[:, :, 0:1], gt_array.shape[2], axis=2)
    
    # Apply mask to ground truth image
    print("Applying mask to ground truth image...")
    if gt_array.max() > 1:
        gt_array = gt_array.astype(np.float32) / 255.0
        masked_gt = gt_array * combined_mask
        masked_gt = (masked_gt * 255.0).astype(np.uint8)
    else:
        masked_gt = (gt_array * combined_mask).astype(np.uint8)
    
    # Convert back to PIL Image and save
    if len(masked_gt.shape) == 3:
        output_image = Image.fromarray(masked_gt)
    else:
        output_image = Image.fromarray(masked_gt, mode='L')
    
    output_path = os.path.join(folder_path, f"{base_name}_maskedgt.png")
    output_image.save(output_path)
    print(f"Saved masked ground truth image to: {output_path}")
    
    return output_path


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python mask_and_apply.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    try:
        process_masks(folder_path)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


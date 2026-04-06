"""
Compute the LPIPS between shadow images and original images.
Processes all shadow images in scene folders and selects the top 5 with lowest LPIPS.
Lower LPIPS indicates better quality/less perceptual difference from the original.
"""

import argparse
import numpy as np
from PIL import Image
import os
import shutil
from pathlib import Path
import torch
import lpips


def masked_psnr(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Compute masked PSNR between two images.
    
    Args:
        img1: First image tensor, normalized to [0, 1], shape [C, H, W] or [1, C, H, W]
        img2: Second image tensor, normalized to [0, 1], shape [C, H, W] or [1, C, H, W]
        mask: Binary mask tensor, shape [1, H, W] or [1, 1, H, W], values in {0, 1}
    
    Returns:
        PSNR value in dB. Higher is better.
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape.")

    mask = mask.expand_as(img1)  # [1, 3, H, W] or [3, H, W]
    valid = mask == 1.0
    diff = (img1 - img2)[valid]
    if diff.numel() == 0:
        raise ValueError("Mask selects no valid pixels.")

    mse = torch.mean(diff ** 2)
    if torch.isclose(mse, torch.tensor(0.0, device=mse.device)):
        return float("inf")
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.item())


def masked_lpips(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor, loss_fn) -> float:
    """
    Compute masked LPIPS between two images using spatial mode.
    
    Args:
        img1: First image tensor, normalized to [0, 1], shape [1, C, H, W]
        img2: Second image tensor, normalized to [0, 1], shape [1, C, H, W]
        mask: Binary mask tensor, shape [1, 1, H, W], values in {0, 1}
        loss_fn: Pre-loaded LPIPS model
    
    Returns:
        LPIPS distance value. Lower is better (less perceptual difference).
    """
    mask_rgb = mask.expand_as(img1)

    # LPIPS expects inputs in [-1, 1]
    img1_tensor = img1 * 2.0 - 1.0
    img2_tensor = img2 * 2.0 - 1.0

    with torch.no_grad():
        distance_map = loss_fn(img2_tensor, img1_tensor)
        valid = mask == 1.0
        if not valid.any():
            raise ValueError("Mask selects no valid pixels at LPIPS resolution.")

        masked_distance = distance_map[valid]
        distance = masked_distance.mean()
    
    return float(distance.item())


def compute_masked_psnr(shadow_img_path, main_img_path, mask_img_path):
    """
    Compute the PSNR (Peak Signal-to-Noise Ratio) between two images within a masked region.
    Returns the PSNR value within the mask. Higher PSNR means better quality/less difference.
    """
    # Load images
    shadow_img = Image.open(shadow_img_path).convert('RGB')
    main_img = Image.open(main_img_path).convert('RGB')
    mask_img = Image.open(mask_img_path).convert('L')  # Load as grayscale
    
    # Ensure same dimensions
    if shadow_img.size != main_img.size:
        main_img = main_img.resize(shadow_img.size, Image.LANCZOS)
    
    if mask_img.size != shadow_img.size:
        mask_img = mask_img.resize(shadow_img.size, Image.LANCZOS)
    
    # Convert to numpy arrays and normalize to [0, 1]
    shadow_arr = np.array(shadow_img, dtype=np.float32) / 255.0
    main_arr = np.array(main_img, dtype=np.float32) / 255.0
    mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
    
    # Convert to torch tensors [H, W, C] -> [C, H, W]
    shadow_tensor = torch.from_numpy(shadow_arr).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    main_tensor = torch.from_numpy(main_arr).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Binarize mask (threshold at 0.5)
    mask_tensor = (mask_tensor > 0.5).float()
    
    # Compute PSNR
    psnr = masked_psnr(shadow_tensor, main_tensor, mask_tensor)
    
    return psnr


def compute_masked_lpips(shadow_img_path, main_img_path, mask_img_path, loss_fn, device):
    """
    Compute the LPIPS (Learned Perceptual Image Patch Similarity) between two images within a masked region.
    Returns the LPIPS value within the mask. Lower LPIPS means better quality/less perceptual difference.
    """
    # Load images
    shadow_img = Image.open(shadow_img_path).convert('RGB')
    main_img = Image.open(main_img_path).convert('RGB')
    mask_img = Image.open(mask_img_path).convert('L')  # Load as grayscale
    
    # Ensure same dimensions
    if shadow_img.size != main_img.size:
        main_img = main_img.resize(shadow_img.size, Image.LANCZOS)
    
    if mask_img.size != shadow_img.size:
        mask_img = mask_img.resize(shadow_img.size, Image.LANCZOS)
    
    # Convert to numpy arrays and normalize to [0, 1]
    shadow_arr = np.array(shadow_img, dtype=np.float32) / 255.0
    main_arr = np.array(main_img, dtype=np.float32) / 255.0
    mask_arr = np.array(mask_img, dtype=np.float32) / 255.0
    
    # Convert to torch tensors [H, W, C] -> [C, H, W]
    shadow_tensor = torch.from_numpy(shadow_arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, C, H, W]
    main_tensor = torch.from_numpy(main_arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, C, H, W]
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    
    # Binarize mask (threshold at 0.5)
    mask_tensor = (mask_tensor > 0.5).float()
    
    # Compute LPIPS
    lpips_value = masked_lpips(shadow_tensor, main_tensor, mask_tensor, loss_fn)
    
    return lpips_value


def parse_args():
    parser = argparse.ArgumentParser(description="Compute masked LPIPS for scene shadow images")
    parser.add_argument("--out_dir", default=os.getcwd(), help="Directory containing scene folders")
    parser.add_argument("--top_n", type=int, default=1, help="Number of best images to save per object")
    parser.add_argument("--metric", choices=["lpips", "psnr"], default="lpips", help="Metric to use: 'lpips' (lower is better) or 'psnr' (higher is better)")
    return parser.parse_args()


def main():
    """
    Process all scene folders, compute masked LPIPS for shadow images,
    and save the top N images with lowest LPIPS.
    """
    args = parse_args()
    output_dir = args.out_dir
    top_n = args.top_n

    if not os.path.exists(output_dir):
        print(f"Error: output_dir '{output_dir}' does not exist!")
        return

    print(f"Using output_dir={output_dir}, top_n={top_n}")

    metric = args.metric

    loss_fn = None
    device = "cpu"
    if metric == "lpips":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading LPIPS model on {device}...")
        loss_fn = lpips.LPIPS(net="vgg", spatial=True).to(device)
        print("LPIPS model loaded successfully!")
    else:
        print("Using PSNR metric (no LPIPS model loaded)")

    # For each scene, iterate object folders (like add_shadows.py) and compute LPIPS for main_shadow_* images
    # Use the same iteration pattern as add_shadows.py (with commented alternatives)
    processed_count = 0
    for scene_name in sorted(os.listdir(output_dir)):
        # for scene_name in sorted(os.listdir(output_dir), reverse=True):
            # if not os.path.isdir(os.path.join(output_dir, scene_name)):
            #     continue
            # if not scene_name.startswith("liv"):
            #     continue
        print("\n" + "="*56)
        print(f"Processing scene: {scene_name}")
        print("" + "="*56 + "\n")

        scene_path = os.path.join(output_dir, scene_name)

        main_img_path = os.path.join(scene_path, "base_image.jpg")
        if not os.path.exists(main_img_path):
            print(f"Warning: base_image.jpg not found in {scene_path}, skipping scene")
            continue

        # iterate all subfolders in the scene (objects)
        for obj_name in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
            obj_path = os.path.join(scene_path, obj_name)
            if not os.path.isdir(obj_path):
                continue

            mask_img_path = os.path.join(obj_path, "mask_fg.jpg")

            # support multiple possible shadow folder names (main_shadows, main_shadow, shadows)
            shadow_folder_candidates = ["main_shadows", "main_shadow", "shadows"]
            shadows_dir = None
            for cand in shadow_folder_candidates:
                cand_path = os.path.join(obj_path, cand)
                if os.path.exists(cand_path) and os.path.isdir(cand_path):
                    shadows_dir = cand_path
                    break

            if not os.path.exists(mask_img_path):
                print(f"  Skipping object '{obj_name}': mask not found at {mask_img_path}")
                continue

            if shadows_dir is None:
                print(f"  Skipping object '{obj_name}': shadows folder not found (checked {shadow_folder_candidates})")
                continue
            else:
                print(f"  Using shadows folder: {shadows_dir}")

            shadow_images = [f for f in sorted(os.listdir(shadows_dir))
                             if f.startswith("main_shadow_") and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not shadow_images:
                print(f"  No shadow images found for object '{obj_name}' in {shadows_dir}")
                continue

            print(f"  Found {len(shadow_images)} shadow images for object '{obj_name}' - computing LPIPS...")

            lpips_results = []
            for shadow_name in shadow_images:
                shadow_path = os.path.join(shadows_dir, shadow_name)
                try:
                    if metric == "lpips":
                        score = compute_masked_lpips(shadow_path, main_img_path, mask_img_path, loss_fn, device)
                        print(f"    {shadow_name}: LPIPS = {score:.4f}")
                    else:
                        score = compute_masked_psnr(shadow_path, main_img_path, mask_img_path)
                        print(f"    {shadow_name}: PSNR = {score:.4f}")
                    lpips_results.append((shadow_name, score, shadow_path))
                except Exception as e:
                    print(f"    Error processing {shadow_name}: {e}")

            if not lpips_results:
                print(f"  No valid LPIPS values for object '{obj_name}', skipping")
                continue

            # Sort: LPIPS -> ascending (lower better); PSNR -> descending (higher better)
            if metric == "lpips":
                lpips_results.sort(key=lambda x: x[1])
            else:
                lpips_results.sort(key=lambda x: x[1], reverse=True)
            top_images = lpips_results[:top_n]

            print(f"  Top {top_n} images for '{obj_name}':")
            for rank, (img_name, lp_val, _) in enumerate(top_images, 1):
                label = "LPIPS" if metric == "lpips" else "PSNR"
                print(f"    {rank}. {img_name} ({label}: {lp_val:.4f})")

            # Copy top N images into the object folder
            for rank, (img_name, lp_val, src_path) in enumerate(top_images, 1):
                base_name, ext = os.path.splitext(img_name)
                metric_tag = "lpips" if metric == "lpips" else "psnr"
                new_name = f"best_{rank:02d}_{base_name}_{metric_tag}{lp_val:.4f}{ext}"
                dst_path = os.path.join(obj_path, new_name)
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"    Copied: {new_name}")
                except Exception as e:
                    print(f"    Error copying {img_name} -> {dst_path}: {e}")

            # Also copy the best (rank 1) as main_shadow.jpg at the object folder level
            try:
                best_src = top_images[0][2]
                main_dst = os.path.join(obj_path, "main_shadow.jpg")
                shutil.copy2(best_src, main_dst)
                print(f"    Wrote best shadow to: {main_dst}")
            except Exception as e:
                print(f"    Error copying best shadow to main_shadow.jpg: {e}")

        print(f"Done with scene {scene_name}.")
        processed_count += 1

    print(f"\n{'#'*80}")
    print(f"# ALL DONE! Processed {processed_count} scene folders.")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()


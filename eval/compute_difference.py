import argparse
import os
import glob
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute difference between ground truth and generated images in masked region."
    )
    parser.add_argument("input_folder", type=str, help="Path to folder containing *_gt.png, *_main.png, and *_mask.png images.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output directory for difference images and histograms. Default: input_folder/differences",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (e.g. 'cpu', 'cuda', 'cuda:0'). Default: cuda.",
    )
    return parser.parse_args()


def ensure_file_exists(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def load_rgb_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: str) -> np.ndarray:
    mask_img = Image.open(path).convert("L")
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return mask


def to_torch_image(image: np.ndarray, device: str) -> torch.Tensor:
    # [H, W, C] -> [1, C, H, W]
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device=device)
    return tensor


def to_torch_mask(mask: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device=device)
    binary = (tensor >= 0.5).float()
    if binary.sum() == 0:
        raise ValueError("Mask contains no foreground pixels after thresholding.")
    return binary


def compute_difference(
    ground_truth_path: str, generated_path: str, mask_path: str, device: str, output_dir: str, prefix: str = ""
) -> None:
    """
    Compute difference between GT and generated image in masked region, save difference image and histogram.
    
    Args:
        ground_truth_path: Path to ground truth image
        generated_path: Path to generated image
        mask_path: Path to mask image
        device: Torch device
        output_dir: Directory to save outputs
        prefix: Prefix for output filenames
    """
    ensure_file_exists(ground_truth_path, "Ground truth image")
    ensure_file_exists(generated_path, "Generated image")
    ensure_file_exists(mask_path, "Mask image")

    gt_np = load_rgb_image(ground_truth_path)
    gen_np = load_rgb_image(generated_path)
    mask_np = load_mask(mask_path)

    if mask_np.shape != gt_np.shape[:2] or mask_np.shape != gen_np.shape[:2]:
        raise ValueError("Mask dimensions must match the height and width of the images.")

    gt = to_torch_image(gt_np, device=device)
    gen = to_torch_image(gen_np, device=device)
    mask = to_torch_mask(mask_np, device=device)

    # Compute difference: GT - generated
    diff = gt - gen  # [1, 3, H, W]

    # Apply mask to difference
    mask_rgb = mask.expand_as(diff)  # [1, 3, H, W]
    masked_diff = diff * mask_rgb

    # Save the masked difference image
    # Normalize difference to [0, 1] for visualization
    # Shift from [-1, 1] range to [0, 1] range
    diff_normalized = (masked_diff + 1.0) / 2.0
    diff_normalized = torch.clamp(diff_normalized, 0.0, 1.0)
    
    prefix_str = f"{prefix}_" if prefix else ""
    diff_image_path = os.path.join(output_dir, f"{prefix_str}difference.png")
    torchvision.utils.save_image(diff_normalized, diff_image_path)
    
    # Also save absolute difference
    abs_diff = torch.abs(masked_diff)
    abs_diff_normalized = torch.clamp(abs_diff, 0.0, 1.0)
    abs_diff_image_path = os.path.join(output_dir, f"{prefix_str}difference_abs.png")
    torchvision.utils.save_image(abs_diff_normalized, abs_diff_image_path)

    # Extract masked difference values for histogram
    valid = mask_rgb == 1.0
    if valid.sum() == 0:
        raise ValueError("Mask selects no valid pixels for difference computation.")
    
    diff_values = diff[valid].cpu().numpy()  # [N] flattened difference values
    abs_diff_values = torch.abs(diff[valid]).cpu().numpy()  # [N] absolute difference values

    # Compute histogram of difference values
    # Use bins from -1 to 1 for signed difference, 0 to 1 for absolute difference
    n_bins = 256
    
    # Histogram for signed difference
    hist_signed, bin_edges_signed = np.histogram(diff_values, bins=n_bins, range=(-1.0, 1.0))
    bin_centers_signed = (bin_edges_signed[:-1] + bin_edges_signed[1:]) / 2.0
    
    # Histogram for absolute difference
    hist_abs, bin_edges_abs = np.histogram(abs_diff_values, bins=n_bins, range=(0.0, 1.0))
    bin_centers_abs = (bin_edges_abs[:-1] + bin_edges_abs[1:]) / 2.0

    # Save histogram plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Signed difference histogram
    axes[0].bar(bin_centers_signed, hist_signed, width=(bin_edges_signed[1] - bin_edges_signed[0]), alpha=0.7)
    axes[0].set_xlabel('Difference Value (GT - Generated)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Histogram of Signed Difference (Masked Region)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=1, label='Zero')
    axes[0].legend()
    
    # Absolute difference histogram
    axes[1].bar(bin_centers_abs, hist_abs, width=(bin_edges_abs[1] - bin_edges_abs[0]), alpha=0.7, color='orange')
    axes[1].set_xlabel('Absolute Difference Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Histogram of Absolute Difference (Masked Region)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    histogram_path = os.path.join(output_dir, f"{prefix_str}difference_histogram.png")
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Print statistics
    print(f"  Difference statistics (masked region):")
    print(f"    Mean: {diff_values.mean():.6f}")
    print(f"    Std: {diff_values.std():.6f}")
    print(f"    Min: {diff_values.min():.6f}")
    print(f"    Max: {diff_values.max():.6f}")
    print(f"    Mean absolute: {abs_diff_values.mean():.6f}")
    print(f"    RMSE: {np.sqrt(np.mean(diff_values**2)):.6f}")


def find_image_groups(folder_path: str) -> List[Dict[str, str]]:
    """Find all image groups in the folder matching *_gt.png, *_main.png, *_mask.png patterns."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Input folder does not exist: {folder_path}")
    
    # Find all matching files
    gt_files = glob.glob(os.path.join(folder_path, "*_gt.png"))
    main_files = glob.glob(os.path.join(folder_path, "*_main.png"))
    mask_files = glob.glob(os.path.join(folder_path, "*_mask.png"))
    
    # Create dictionaries mapping prefix to file path
    gt_dict = {}
    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        prefix = basename[:-7]  # Remove "_gt.png"
        gt_dict[prefix] = gt_file
    
    main_dict = {}
    for main_file in main_files:
        basename = os.path.basename(main_file)
        prefix = basename[:-9]  # Remove "_main.png"
        main_dict[prefix] = main_file
    
    mask_dict = {}
    for mask_file in mask_files:
        basename = os.path.basename(mask_file)
        prefix = basename[:-9]  # Remove "_mask.png"
        mask_dict[prefix] = mask_file
    
    # Find common prefixes that have all three files
    all_prefixes = set(gt_dict.keys()) & set(main_dict.keys()) & set(mask_dict.keys())
    
    if not all_prefixes:
        raise ValueError("No complete image groups found. Each group needs *_gt.png, *_main.png, and *_mask.png files.")
    
    # Create list of image groups
    groups = []
    for prefix in sorted(all_prefixes):
        groups.append({
            "prefix": prefix,
            "gt": gt_dict[prefix],
            "main": main_dict[prefix],
            "mask": mask_dict[prefix]
        })
    
    return groups


def main() -> None:
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_folder, "differences")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all image groups in the folder
    print(f"Scanning folder: {args.input_folder}")
    image_groups = find_image_groups(args.input_folder)
    print(f"Found {len(image_groups)} image group(s) to process.")
    print(f"Output directory: {args.output_dir}")
    
    # Process each group
    for i, group in enumerate(image_groups, 1):
        prefix = group["prefix"]
        print(f"\nProcessing [{i}/{len(image_groups)}]: {prefix}")
        
        try:
            compute_difference(
                group["gt"], group["main"], group["mask"], args.device,
                output_dir=args.output_dir, prefix=prefix
            )
        except Exception as e:
            print(f"  Error processing {prefix}: {str(e)}")
    
    print(f"\nCompleted! Processed {len(image_groups)} image group(s).")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


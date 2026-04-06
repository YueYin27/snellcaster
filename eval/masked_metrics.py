import argparse
import os
import glob
import csv
from typing import Tuple, List, Dict

import lpips
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models import VGG16_Weights
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np
import torch
from PIL import Image
import os
from pathlib import Path
import ImageReward as RM




# Define scene names and their corresponding prompts
scene_prompts = {
    "living": "A cozy living room with a polished wooden coffee table close to the camera, with a solid transparent glass sphere on top, surrounded by a beige sofa with colorful pillows, a patterned rug, green plants, bookshelves, framed wall art, and sunlight filtering through sheer curtains.",
    "dining": "A bright dining room with a wooden dining table placed in the center, with a solid transparent glass sphere on top, surrounded by upholstered chairs, a pendant lamp above, fruit bowls and paintings in the background, and daylight streaming through tall windows with patterned curtains.",
    "office": "A minimalist home office with a smooth wooden desk positioned closer to the camera, with a solid transparent glass sphere on top, surrounded by a black office chair, bookshelves with plants, framed posters, a side table with a computer monitor, and a large window with blinds letting in soft daylight.",
    "kitchen": "A modern kitchen with a large marble island in the center, with a solid transparent glass sphere on top, surrounded by wooden cabinetry, bar stools, hanging lights, colorful utensils, and reflections from stainless-steel appliances under bright morning light.",
    "artroom": "An art classroom with a rectangular wooden worktable near the camera, with a solid transparent glass sphere on top, surrounded by easels, color-splattered stools, sketches pinned on the walls, jars of brushes on shelves, and warm daylight pouring through wide windows.",
    "cafe": "A minimalist café interior with a square wooden table in the near foreground, with a solid transparent glass sphere on top, surrounded by metal-framed chairs, green plants, hanging lights, a counter with pastries and cups in the background, and soft sunlight illuminating the colorful tiled floor.",
    "landscape": "A beautiful landscape with a river and mountains, viewed from a camera positioned directly in front of a stone table and chairs in the immediate foreground, filling the lower frame and very close to the lens, a solid transparent glass sphere on the table, on the left of some colorful food on the table.",
    "karaoke": "A karaoke room with colorful lights, TV on the wall displaying music videos, a coffee table with a solid transparent glass sphere on top, in front of the TV, and a sofa around the coffee table.",
    "cave": "A rocky cave interior lit by a bright campfire, warm flickering light casting dramatic shadows on the walls, a solid transparent glass sphere on the ground, with camping gear scattered around—tents, sleeping bags, backpacks, lanterns, cooking pots, and a folding chair—smoke and embers in the air, cozy but high contrast.",
    "desert": "A high-noon desert scene with blinding sunlight and hard shadows, heat haze over sand and rocks, a solid transparent glass sphere on the ground, and camping gear in the foreground—tent, backpacks, and a small stove.",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute masked PSNR and LPIPS between ground truth and generated images."
    )
    parser.add_argument("input_folder", type=str, help="Path to folder containing *_gt.png, *_main.png, and *_mask.png images.")
    parser.add_argument(
        "--output_csv",
        type=str,
        default="metrics_results_noisy.csv",
        help="Path to output CSV file. Default: metrics_results.csv",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run LPIPS on (e.g. 'cpu', 'cuda', 'cuda:0'). Default: cuda.",
    )
    return parser.parse_args()

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)


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


def masked_psnr(gt: torch.Tensor, gen: torch.Tensor, mask: torch.Tensor) -> float:
    if gt.shape != gen.shape:
        raise ValueError("Ground truth and generated images must have the same shape.")
    # mse = torch.mean((gt - gen) ** 2)
    # print("unmasked PSNR", 10.0 * torch.log10(1.0 / mse))

    mask = mask.expand_as(gt)  # [1, 3, H, W]
    valid = mask == 1.0
    diff = (gt - gen)[valid]
    if diff.numel() == 0:
        raise ValueError("Mask selects no valid pixels.")

    mse = torch.mean(diff ** 2)
    if torch.isclose(mse, torch.tensor(0.0, device=mse.device)):
        return float("inf")
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.item())


def masked_lpips(gt: torch.Tensor, gen: torch.Tensor, mask: torch.Tensor, device: str, output_dir: str, prefix: str = "", spatial: bool = False) -> float:
    if spatial:
        loss_fn = lpips.LPIPS(net="vgg", spatial=True).to(device)
    else:
        loss_fn = lpips.LPIPS(net="vgg").to(device)
    mask_rgb = mask.expand_as(gt)

    masked_gt = gt * mask_rgb
    masked_gen = gen * mask_rgb

    # Save the masked images
    masked_gt_path = os.path.join(output_dir, f"{prefix}_masked_gt.png" if prefix else "masked_gt.png")
    masked_gen_path = os.path.join(output_dir, f"{prefix}_masked_gen.png" if prefix else "masked_gen.png")
    torchvision.utils.save_image(masked_gt, masked_gt_path)
    torchvision.utils.save_image(masked_gen, masked_gen_path)

    # LPIPS expects inputs in [-1, 1]
    gt_tensor = masked_gt * 2.0 - 1.0
    gen_tensor = masked_gen * 2.0 - 1.0

    with torch.no_grad():
        if spatial:
            distance_map = loss_fn(gen_tensor, gt_tensor)
            lpips_map_path = os.path.join(output_dir, f"{prefix}_lpips_map.png" if prefix else "lpips_map.png")
            torchvision.utils.save_image(distance_map, lpips_map_path)
            valid = mask == 1.0
            if not valid.any():
                raise ValueError("Mask selects no valid pixels at LPIPS resolution.")

            masked_distance = distance_map[valid]
            distance = masked_distance.mean()
        else:
            # Compute bounding box covering the masked region
            mask_2d = mask.squeeze(0).squeeze(0)
            coords = torch.nonzero(mask_2d, as_tuple=False)
            if coords.numel() == 0:
                raise ValueError("Mask selects no valid pixels for LPIPS computation.")

            ymin = int(coords[:, 0].min().item())
            ymax = int(coords[:, 0].max().item()) + 1
            xmin = int(coords[:, 1].min().item())
            xmax = int(coords[:, 1].max().item()) + 1

            gt_crop = gt_tensor[:, :, ymin:ymax, xmin:xmax]
            gen_crop = gen_tensor[:, :, ymin:ymax, xmin:xmax]

            # Save the cropped images, denormalize the images
            gt_crop = (gt_crop + 1.0) / 2.0
            gen_crop = (gen_crop + 1.0) / 2.0
            gt_crop_path = os.path.join(output_dir, f"{prefix}_gt_crop.png" if prefix else "gt_crop.png")
            gen_crop_path = os.path.join(output_dir, f"{prefix}_gen_crop.png" if prefix else "gen_crop.png")
            torchvision.utils.save_image(gt_crop, gt_crop_path)
            torchvision.utils.save_image(gen_crop, gen_crop_path)

            if gt_crop.numel() == 0 or gen_crop.numel() == 0:
                raise ValueError("Cropped tensors are empty; check mask contents.")

            distance = loss_fn(gen_crop, gt_crop)
    return float(distance.item())


def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB image to grayscale luminance using standard weights.
    
    Args:
        rgb: RGB image tensor [1, 3, H, W] or [3, H, W]
    
    Returns:
        Grayscale luminance tensor [1, 1, H, W] or [1, H, W]
    """
    # Standard RGB to grayscale conversion weights
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
    if rgb.dim() == 4:  # [1, 3, H, W]
        weights = weights.view(1, 3, 1, 1)
        luminance = (rgb * weights).sum(dim=1, keepdim=True)
    else:  # [3, H, W]
        weights = weights.view(3, 1, 1)
        luminance = (rgb * weights).sum(dim=0, keepdim=True)
    return luminance


def histogram_equalization(image_values: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """
    Apply histogram equalization to image values.
    Transforms the image so that its histogram is approximately uniform.
    
    Args:
        image_values: Image values [N]
        n_bins: Number of bins for histogram
    
    Returns:
        Equalized image values [N]
    """
    # Normalize to [0, 1] for histogram computation
    img_min, img_max = image_values.min().item(), image_values.max().item()
    img_norm = (image_values - img_min) / (img_max - img_min + 1e-8)
    
    # Compute histogram
    hist = torch.histc(img_norm, bins=n_bins, min=0.0, max=1.0)
    
    # Normalize histogram to get PDF
    pdf = hist / (hist.sum() + 1e-8)
    
    # Compute CDF
    cdf = torch.cumsum(pdf, dim=0)
    
    # Map each pixel value using CDF to get uniform distribution
    bin_indices = torch.clamp((img_norm * (n_bins - 1)).long(), 0, n_bins - 1)
    equalized_norm = cdf[bin_indices]
    
    # Denormalize back to original range
    equalized = equalized_norm * (img_max - img_min) + img_min
    
    return equalized


def histogram_matching(source: torch.Tensor, template: torch.Tensor, n_bins: int = 256) -> torch.Tensor:
    """
    Match histogram of source to template using histogram matching.
    
    Args:
        source: Source image values [N]
        template: Template image values [M]
        n_bins: Number of bins for histogram
    
    Returns:
        Matched source values [N]
    """
    # Compute histograms and CDFs
    source_min, source_max = source.min().item(), source.max().item()
    template_min, template_max = template.min().item(), template.max().item()
    
    # Normalize to [0, 1] for histogram computation
    source_norm = (source - source_min) / (source_max - source_min + 1e-8)
    template_norm = (template - template_min) / (template_max - template_min + 1e-8)
    
    # Compute histograms
    source_hist = torch.histc(source_norm, bins=n_bins, min=0.0, max=1.0)
    template_hist = torch.histc(template_norm, bins=n_bins, min=0.0, max=1.0)
    
    # Normalize histograms to get PDFs
    source_pdf = source_hist / (source_hist.sum() + 1e-8)
    template_pdf = template_hist / (template_hist.sum() + 1e-8)
    
    # Compute CDFs
    source_cdf = torch.cumsum(source_pdf, dim=0)
    template_cdf = torch.cumsum(template_pdf, dim=0)
    
    # Create mapping: for each source bin, find corresponding template bin
    # using inverse CDF mapping
    bin_centers = torch.linspace(0.0, 1.0, n_bins, device=source.device)
    
    # For each source value, find its bin and map to template
    source_bin_indices = torch.clamp((source_norm * (n_bins - 1)).long(), 0, n_bins - 1)
    source_cdf_values = source_cdf[source_bin_indices]
    
    # Vectorized matching: for each source CDF value, find closest template CDF
    # Expand dimensions for broadcasting: [N, 1] vs [1, n_bins]
    source_cdf_expanded = source_cdf_values.unsqueeze(1)  # [N, 1]
    template_cdf_expanded = template_cdf.unsqueeze(0)  # [1, n_bins]
    
    # Find closest template bin for each source value
    diff = torch.abs(template_cdf_expanded - source_cdf_expanded)  # [N, n_bins]
    matched_bins = torch.argmin(diff, dim=1)  # [N]
    matched_values = bin_centers[matched_bins]
    
    # Denormalize back to original range
    matched = matched_values * (source_max - source_min) + source_min
    
    return matched


def apply_grayscale_histogram_matching(gt_gray: torch.Tensor, gen_gray: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply histogram matching to grayscale images within the masked region.
    Match the histogram of generated grayscale image to ground truth grayscale image.
    
    Args:
        gt_gray: Ground truth grayscale image tensor [1, 1, H, W]
        gen_gray: Generated grayscale image tensor [1, 1, H, W]
        mask: Binary mask tensor [1, 1, H, W]
    
    Returns:
        Matched GT and generated grayscale images (GT unchanged, gen histogram-matched)
    """
    mask_expanded = mask.expand_as(gt_gray)  # [1, 1, H, W]
    valid = mask_expanded == 1.0
    if valid.sum() == 0:
        raise ValueError("Mask selects no valid pixels for histogram matching.")
    
    # Extract masked values
    gt_masked = gt_gray[valid]  # [N]
    gen_masked = gen_gray[valid]  # [N]
    
    # Apply histogram matching: match gen to gt
    gen_matched_values = histogram_matching(gen_masked, gt_masked)
    
    # Reconstruct full grayscale image with matched values
    gen_matched = gen_gray.clone()
    gen_matched[valid] = gen_matched_values
    
    # Clamp values to valid range [0, 1]
    gen_matched = torch.clamp(gen_matched, 0.0, 1.0)
    
    # GT remains unchanged
    gt_matched = gt_gray.clone()
    
    return gt_matched, gen_matched


def compute_metrics(
    ground_truth_path: str, generated_path: str, mask_path: str, device: str, output_dir: str, prefix: str = ""
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Compute metrics including grayscale PSNR with and without histogram matching, CLIP score, and ImageReward score.
    
    Returns:
        Tuple of (psnr_rgb, lpips_spatial, lpips_non_spatial, psnr_gray_no_match, psnr_gray_matched, clip_score, image_reward_score, mae_gray_matched)
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

    # Convert to grayscale
    gt_gray = rgb_to_luminance(gt)  # [1, 1, H, W]
    gen_gray = rgb_to_luminance(gen)  # [1, 1, H, W]
    
    # Apply histogram equalization to both grayscale images
    mask_expanded = mask.expand_as(gt_gray)  # [1, 1, H, W]
    valid = mask_expanded == 1.0
    if valid.sum() == 0:
        raise ValueError("Mask selects no valid pixels for histogram equalization.")
    
    # Extract masked values
    gt_masked = gt_gray[valid]  # [N]
    gen_masked = gen_gray[valid]  # [N]
    
    # Apply histogram equalization to both images
    # gt_equalized_values = histogram_equalization(gt_masked)
    # gen_equalized_values = histogram_equalization(gen_masked)
    
    # Reconstruct equalized grayscale images
    # gt_gray_equalized = gt_gray.clone()
    # gen_gray_equalized = gen_gray.clone()
    # gt_gray_equalized[valid] = gt_equalized_values
    # gen_gray_equalized[valid] = gen_equalized_values
    # gt_gray_equalized = torch.clamp(gt_gray_equalized, 0.0, 1.0)
    # gen_gray_equalized = torch.clamp(gen_gray_equalized, 0.0, 1.0)
    
    # Compute grayscale PSNR on equalized images (unmatched)
    # psnr_gray_no_match = masked_psnr(gt_gray_equalized, gen_gray_equalized, mask)
    psnr_gray_no_match = masked_psnr(gt_gray, gen_gray, mask)
    
    # Apply histogram matching to equalized grayscale images
    gt_gray_matched, gen_gray_matched = apply_grayscale_histogram_matching(gt_gray, gen_gray, mask)
    
    # Compute grayscale PSNR with histogram matching
    psnr_gray_matched = masked_psnr(gt_gray_matched, gen_gray_matched, mask)
    
    # Compute MAE (Mean Absolute Error) in masked region for matched grayscale images
    mask_expanded_matched = mask.expand_as(gt_gray_matched)  # [1, 1, H, W]
    valid_matched = mask_expanded_matched == 1.0
    if valid_matched.sum() == 0:
        raise ValueError("Mask selects no valid pixels for MAE computation.")
    diff_matched = (gt_gray_matched - gen_gray_matched)[valid_matched]
    mae_gray_matched = torch.mean(torch.abs(diff_matched))
    mae_gray_matched = float(mae_gray_matched.item())
    
    # Save all intermediate grayscale images
    prefix_str = f"{prefix}_" if prefix else ""
    gt_gray_path = os.path.join(output_dir, f"{prefix_str}gt_gray.png")
    gen_gray_path = os.path.join(output_dir, f"{prefix_str}gen_gray.png")
    # gt_gray_equalized_path = os.path.join(output_dir, f"{prefix_str}gt_gray_equalized.png")
    # gen_gray_equalized_path = os.path.join(output_dir, f"{prefix_str}gen_gray_equalized.png")
    gt_gray_matched_path = os.path.join(output_dir, f"{prefix_str}gt_gray_matched.png")
    gen_gray_matched_path = os.path.join(output_dir, f"{prefix_str}gen_gray_matched.png")
    
    # Save the grayscale images
    torchvision.utils.save_image(gt_gray, gt_gray_path)
    torchvision.utils.save_image(gen_gray, gen_gray_path)
    # torchvision.utils.save_image(gt_gray_equalized, gt_gray_equalized_path)
    # torchvision.utils.save_image(gen_gray_equalized, gen_gray_equalized_path)
    torchvision.utils.save_image(gt_gray_matched, gt_gray_matched_path)
    torchvision.utils.save_image(gen_gray_matched, gen_gray_matched_path)

    # Compute RGB PSNR (original, no normalization)
    psnr_rgb = masked_psnr(gt, gen, mask)
    
    # LPIPS still uses original images
    lpips_spatial = masked_lpips(gt, gen, mask, device=device, output_dir=output_dir, prefix=prefix, spatial=True)
    lpips_non_spatial = masked_lpips(gt, gen, mask, device=device, output_dir=output_dir, prefix=prefix, spatial=False)
    
    # Compute CLIP score and ImageReward score using generated image and scene prompt
    # Extract scene type from prefix (first word before '_')
    scene_type = prefix.split('_')[0] if prefix else ""
    print(f"Scene type: {scene_type}")
    if scene_type in scene_prompts:
        prompt = scene_prompts[scene_type]
        # Prepare generated image for CLIP score: [H, W, C] numpy array in [0, 1]
        gen_for_clip = gen_np  # Already in [0, 1] range, shape [H, W, C]
        # Add batch dimension: [1, H, W, C]
        gen_for_clip_batch = np.expand_dims(gen_for_clip, axis=0)
        clip_score_val = calculate_clip_score(gen_for_clip_batch, [prompt])
        ir_model = RM.load("ImageReward-v1.0")
        # Compute ImageReward score using the same prompt and generated image path
        with torch.no_grad():
            image_reward_score = ir_model.score(prompt, generated_path)
        image_reward_score = round(float(image_reward_score), 4)
    else:
        print(f"Warning: Scene type '{scene_type}' not found in scene_prompts. Setting CLIP score and ImageReward score to 0.0")
        clip_score_val = 0.0
        image_reward_score = 0.0
    
    return psnr_rgb, lpips_spatial, lpips_non_spatial, psnr_gray_no_match, psnr_gray_matched, clip_score_val, image_reward_score, mae_gray_matched


def find_image_groups(folder_path: str) -> List[Dict[str, str]]:
    """Find all image groups in the folder matching *_gt.png, *_flux_fill.png, *_mask.png patterns."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Input folder does not exist: {folder_path}")
    
    # Find all matching files
    gt_files = glob.glob(os.path.join(folder_path, "*_gt.png"))
    # main_files = glob.glob(os.path.join(folder_path, "*_main.png"))
    # Search for *_ab.png files in generated_no_pano subdirectories
    main_files = glob.glob(os.path.join(folder_path, "generated_noisy", "*", "*_main.png"))
    # main_files = glob.glob(os.path.join(folder_path, "*_flux.png"))
    # main_files = glob.glob(os.path.join(folder_path, "*_new.png"))
    # main_files = glob.glob(os.path.join(folder_path, "*_flux_fill.png"))
    # main_files = glob.glob(os.path.join(folder_path, "*_main_wo_lpw.png"))
    # main_files = glob.glob(os.path.join(folder_path, "*_main_wo_lpw_shadow.png"))
    # main_files = glob.glob(os.path.join(folder_path, "*_main_wo_tt.png"))
    # main_files = glob.glob(os.path.join(folder_path, "*_main_original.png"))
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
        # prefix = basename[:-9]  # Remove "_main.png"
        prefix = basename[:-9]  # Remove "_ab.png"
        # prefix = basename[:-8]  # Remove "_new.png"
        # prefix = basename[:-14]  # Remove "_flux_fill.png"
        # prefix = basename[:-16]  # Remove "_main_wo_dpa.png"
        # prefix = basename[:-15]  # Remove "_main_wo_tt.png"
        # prefix = basename[:-18]  # Remove "_main_original.png"
        main_dict[prefix] = main_file
    
    mask_dict = {}
    for mask_file in mask_files:
        basename = os.path.basename(mask_file)
        prefix = basename[:-9]  # Remove "_mask.png"
        mask_dict[prefix] = mask_file

    print(main_dict)
    print(mask_dict)
    print(gt_dict)
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
    
    # Set CSV output path to input folder if not explicitly provided
    if args.output_csv == "metrics_results.csv":
        args.output_csv = os.path.join(args.input_folder, "metrics_results.csv")
    elif not os.path.isabs(args.output_csv):
        # If relative path, make it relative to input folder
        args.output_csv = os.path.join(args.input_folder, args.output_csv)
    
    # Ensure output directory exists for intermediate results
    intermediate_dir = os.path.join(args.input_folder, "_intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Find all image groups in the folder
    print(f"Scanning folder: {args.input_folder}")
    image_groups = find_image_groups(args.input_folder)
    print(f"Found {len(image_groups)} image group(s) to process.")
    print(f"Intermediate results will be saved to: {intermediate_dir}")
    
    # Process each group and collect results
    results = []
    for i, group in enumerate(image_groups, 1):
        prefix = group["prefix"]
        print(f"\nProcessing [{i}/{len(image_groups)}]: {prefix}")
        
        # try:
        psnr_rgb, lpips_spatial, lpips_non_spatial, psnr_gray_no_match, psnr_gray_matched, clip_score_val, image_reward_score, mae_gray_matched = compute_metrics(
            group["gt"], group["main"], group["mask"], args.device, 
            output_dir=intermediate_dir, prefix=prefix
        )
        
        results.append({
            "prefix": prefix,
            "psnr_rgb": psnr_rgb if np.isfinite(psnr_rgb) else float("inf"),
            "psnr_gray_no_match": psnr_gray_no_match if np.isfinite(psnr_gray_no_match) else float("inf"),
            "psnr_gray_matched": psnr_gray_matched if np.isfinite(psnr_gray_matched) else float("inf"),
            "lpips_spatial": lpips_spatial,
            "lpips_non_spatial": lpips_non_spatial,
            "clip_score": clip_score_val,
            "image_reward_score": image_reward_score,
            "mae_gray_matched": mae_gray_matched
        })
        
        psnr_rgb_str = f"{psnr_rgb:.4f} dB" if np.isfinite(psnr_rgb) else "inf dB"
        psnr_gray_no_match_str = f"{psnr_gray_no_match:.4f} dB" if np.isfinite(psnr_gray_no_match) else "inf dB"
        psnr_gray_matched_str = f"{psnr_gray_matched:.4f} dB" if np.isfinite(psnr_gray_matched) else "inf dB"
        print(f"  PSNR (RGB): {psnr_rgb_str}, PSNR (Gray, no match): {psnr_gray_no_match_str}, PSNR (Gray, matched): {psnr_gray_matched_str}")
        print(f"  LPIPS (spatial): {lpips_spatial:.6f}, LPIPS (non-spatial): {lpips_non_spatial:.6f}")
        print(f"  CLIP score: {clip_score_val:.4f}, ImageReward score: {image_reward_score:.4f}")
        print(f"  MAE (Gray, matched): {mae_gray_matched:.6f}")
            
        # except Exception as e:
        #     print(f"  Error processing {prefix}: {str(e)}")
        #     results.append({
        #         "prefix": prefix,
        #         "psnr": None,
        #         "lpips_spatial": None,
        #         "lpips_non_spatial": None,
        #         "error": str(e)
        #     })
    
    # Save results to CSV
    print(f"\nSaving results to: {args.output_csv}")
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ["prefix", "psnr_rgb", "psnr_gray_no_match", "psnr_gray_matched", "lpips_spatial", "lpips_non_spatial", "clip_score", "image_reward_score", "mae_gray_matched"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            def format_psnr(val):
                if val is None:
                    return "N/A"
                elif np.isinf(val):
                    return "inf"
                else:
                    return f"{val:.4f}"
            
            psnr_rgb_str = format_psnr(result.get("psnr_rgb"))
            psnr_gray_no_match_str = format_psnr(result.get("psnr_gray_no_match"))
            psnr_gray_matched_str = format_psnr(result.get("psnr_gray_matched"))
            
            lpips_spatial_val = result.get("lpips_spatial")
            lpips_spatial_str = f"{lpips_spatial_val:.6f}" if lpips_spatial_val is not None else "N/A"
            
            lpips_non_spatial_val = result.get("lpips_non_spatial")
            lpips_non_spatial_str = f"{lpips_non_spatial_val:.6f}" if lpips_non_spatial_val is not None else "N/A"
            
            clip_score_val = result.get("clip_score")
            clip_score_str = f"{clip_score_val:.4f}" if clip_score_val is not None else "N/A"
            
            image_reward_score_val = result.get("image_reward_score")
            image_reward_score_str = f"{image_reward_score_val:.4f}" if image_reward_score_val is not None else "N/A"
            
            mae_gray_matched_val = result.get("mae_gray_matched")
            mae_gray_matched_str = f"{mae_gray_matched_val:.6f}" if mae_gray_matched_val is not None else "N/A"
            
            row = {
                "prefix": result["prefix"],
                "psnr_rgb": psnr_rgb_str,
                "psnr_gray_no_match": psnr_gray_no_match_str,
                "psnr_gray_matched": psnr_gray_matched_str,
                "lpips_spatial": lpips_spatial_str,
                "lpips_non_spatial": lpips_non_spatial_str,
                "clip_score": clip_score_str,
                "image_reward_score": image_reward_score_str,
                "mae_gray_matched": mae_gray_matched_str
            }
            writer.writerow(row)
    
    print(f"Completed! Processed {len(results)} image group(s).")


if __name__ == "__main__":
    main()



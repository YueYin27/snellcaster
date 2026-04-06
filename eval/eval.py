"""
Example:
    python eval/eval.py results/generated --generated_image main_shadow.jpg
"""

import argparse
import os
import zipfile
from typing import Tuple, List, Dict
from xml.sax.saxutils import escape

import lpips
import numpy as np
import torch
from PIL import Image
from torchmetrics.functional.multimodal import clip_score
from functools import partial
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
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to generated folder containing scene/object subfolders.",
    )
    parser.add_argument(
        "--generated_image",
        type=str,
        default="main.jpg",
        help="Generated image filename inside each object folder. Default: main.jpg",
    )
    parser.add_argument(
        "--output_xlsx",
        type=str,
        default=None,
        help="Path to output XLSX file. Default: <generated_image_stem>.xlsx in input folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device to run LPIPS on (e.g. 'cpu', 'cuda', 'cuda:0'). Default: cuda.",
    )
    return parser.parse_args()


def _excel_col_name(index: int) -> str:
    """Convert 1-based column index to Excel column name (A, B, ..., AA, AB, ...)."""
    name = []
    while index > 0:
        index, rem = divmod(index - 1, 26)
        name.append(chr(ord("A") + rem))
    return "".join(reversed(name))


def _xlsx_sheet_xml(headers: List[str], rows: List[List[str]]) -> str:
    """Create worksheet XML using inline strings."""
    all_rows = [headers] + rows
    max_col = max((len(r) for r in all_rows), default=1)
    max_row = max(len(all_rows), 1)
    dim = f"A1:{_excel_col_name(max_col)}{max_row}"

    row_xml_parts = []
    for row_idx, row in enumerate(all_rows, start=1):
        cell_xml_parts = []
        for col_idx, value in enumerate(row, start=1):
            cell_ref = f"{_excel_col_name(col_idx)}{row_idx}"
            value_escaped = escape("" if value is None else str(value))
            cell_xml_parts.append(
                f'<c r="{cell_ref}" t="inlineStr"><is><t>{value_escaped}</t></is></c>'
            )
        row_xml_parts.append(f'<row r="{row_idx}">{"".join(cell_xml_parts)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dim}"/>'
        "<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews>"
        "<sheetFormatPr defaultRowHeight=\"15\"/>"
        f'<sheetData>{"".join(row_xml_parts)}</sheetData>'
        "</worksheet>"
    )


def write_results_xlsx(output_path: str, headers: List[str], rows: List[List[str]]) -> None:
    """Write a minimal XLSX file with one worksheet named Metrics."""
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        "</Relationships>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Metrics" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:creator>eval_new.py</dc:creator>"
        "<cp:lastModifiedBy>eval_new.py</cp:lastModifiedBy>"
        "</cp:coreProperties>"
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Python</Application>"
        "</Properties>"
    )
    sheet_xml = _xlsx_sheet_xml(headers, rows)

    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)

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


def masked_lpips(
    gt: torch.Tensor,
    gen: torch.Tensor,
    mask: torch.Tensor,
    loss_fn,
    spatial: bool = False,
) -> float:
    mask_rgb = mask.expand_as(gt)

    masked_gt = gt * mask_rgb
    masked_gen = gen * mask_rgb

    # LPIPS expects inputs in [-1, 1]
    gt_tensor = masked_gt * 2.0 - 1.0
    gen_tensor = masked_gen * 2.0 - 1.0

    with torch.no_grad():
        if spatial:
            distance_map = loss_fn(gen_tensor, gt_tensor)
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
    ground_truth_path: str,
    generated_path: str,
    mask_path: str,
    device: str,
    lpips_spatial_model,
    lpips_non_spatial_model,
    ir_model,
    prefix: str = "",
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
    
    # Compute RGB PSNR (original, no normalization)
    psnr_rgb = masked_psnr(gt, gen, mask)
    
    # LPIPS still uses original images
    lpips_spatial = masked_lpips(
        gt, gen, mask, loss_fn=lpips_spatial_model, spatial=True
    )
    lpips_non_spatial = masked_lpips(
        gt, gen, mask, loss_fn=lpips_non_spatial_model, spatial=False
    )
    
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
        # Compute ImageReward score using the same prompt and generated image path
        with torch.no_grad():
            image_reward_score = ir_model.score(prompt, generated_path)
        image_reward_score = round(float(image_reward_score), 4)
    else:
        print(f"Warning: Scene type '{scene_type}' not found in scene_prompts. Setting CLIP score and ImageReward score to 0.0")
        clip_score_val = 0.0
        image_reward_score = 0.0
    
    return psnr_rgb, lpips_spatial, lpips_non_spatial, psnr_gray_no_match, psnr_gray_matched, clip_score_val, image_reward_score, mae_gray_matched


def find_image_groups(folder_path: str, generated_image_name: str) -> List[Dict[str, str]]:
    """Find all valid scene/object groups with blender.jpg, generated image, and blender_mask.jpg."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Input folder does not exist: {folder_path}")

    groups = []
    for scene_name in sorted(os.listdir(folder_path)):
        scene_path = os.path.join(folder_path, scene_name)
        if not os.path.isdir(scene_path):
            continue

        for object_name in sorted(os.listdir(scene_path)):
            object_path = os.path.join(scene_path, object_name)
            if not os.path.isdir(object_path):
                continue

            gt_path = os.path.join(object_path, "blender.jpg")
            gen_path = os.path.join(object_path, generated_image_name)
            mask_path = os.path.join(object_path, "blender_mask.jpg")
            if not (os.path.isfile(gt_path) and os.path.isfile(gen_path) and os.path.isfile(mask_path)):
                continue

            groups.append(
                {
                    "prefix": f"{scene_name}_{object_name}",
                    "gt": gt_path,
                    "main": gen_path,
                    "mask": mask_path,
                }
            )

    if not groups:
        raise ValueError(
            f"No complete scene/object groups found with blender.jpg, {generated_image_name}, and blender_mask.jpg."
        )

    return groups


def main() -> None:
    args = parse_args()

    generated_stem = Path(args.generated_image).stem
    if args.output_xlsx is None:
        args.output_xlsx = os.path.join(args.input_folder, f"{generated_stem}.xlsx")
    elif not os.path.isabs(args.output_xlsx):
        args.output_xlsx = os.path.join(args.input_folder, args.output_xlsx)
    
    # Find all image groups in the folder
    print(f"Scanning folder: {args.input_folder}")
    image_groups = find_image_groups(args.input_folder, args.generated_image)
    print(f"Found {len(image_groups)} image group(s) to process.")

    print("Loading LPIPS and ImageReward models once...")
    lpips_spatial_model = lpips.LPIPS(net="vgg", spatial=True).to(args.device)
    lpips_non_spatial_model = lpips.LPIPS(net="vgg").to(args.device)
    ir_model = RM.load("ImageReward-v1.0")
    print("Model loading complete.")
    
    # Process each group and collect results
    results = []
    for i, group in enumerate(image_groups, 1):
        prefix = group["prefix"]
        print(f"\nProcessing [{i}/{len(image_groups)}]: {prefix}")
        
        # try:
        psnr_rgb, lpips_spatial, lpips_non_spatial, psnr_gray_no_match, psnr_gray_matched, clip_score_val, image_reward_score, mae_gray_matched = compute_metrics(
            group["gt"],
            group["main"],
            group["mask"],
            args.device,
            lpips_spatial_model,
            lpips_non_spatial_model,
            ir_model,
            prefix=prefix
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
    
    # Save results to XLSX
    print(f"\nSaving results to: {args.output_xlsx}")
    headers = [
        "prefix",
        "psnr_rgb",
        "psnr_gray_no_match",
        "psnr_gray_matched",
        "lpips_spatial",
        "lpips_non_spatial",
        "clip_score",
        "image_reward_score",
        "mae_gray_matched",
    ]

    def format_psnr(val):
        if val is None:
            return "N/A"
        if np.isinf(val):
            return "inf"
        return f"{val:.4f}"

    rows = []
    for result in results:
        lpips_spatial_val = result.get("lpips_spatial")
        lpips_non_spatial_val = result.get("lpips_non_spatial")
        clip_score_val = result.get("clip_score")
        image_reward_score_val = result.get("image_reward_score")
        mae_gray_matched_val = result.get("mae_gray_matched")

        row = [
            result["prefix"],
            format_psnr(result.get("psnr_rgb")),
            format_psnr(result.get("psnr_gray_no_match")),
            format_psnr(result.get("psnr_gray_matched")),
            f"{lpips_spatial_val:.6f}" if lpips_spatial_val is not None else "N/A",
            f"{lpips_non_spatial_val:.6f}" if lpips_non_spatial_val is not None else "N/A",
            f"{clip_score_val:.4f}" if clip_score_val is not None else "N/A",
            f"{image_reward_score_val:.4f}" if image_reward_score_val is not None else "N/A",
            f"{mae_gray_matched_val:.6f}" if mae_gray_matched_val is not None else "N/A",
        ]
        rows.append(row)

    write_results_xlsx(args.output_xlsx, headers, rows)

    print(f"Completed! Processed {len(results)} image group(s).")


if __name__ == "__main__":
    main()



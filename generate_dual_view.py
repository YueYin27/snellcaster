"""
Dual Generation Method for Transparent Object Generation

This pipeline implements a synchronized dual-generation approach that creates:
1. A main image from the primary camera perspective
2. A panoramic image from the center of the transparent object

The panoramic view captures structures that would otherwise be invisible from the main view,
including behind-camera structures, ceiling elements, and occluded regions that are critical
for realistic reflection and refraction effects.
"""

import os
import torch
import torch.nn.functional as F
import time
import argparse
import random
import numpy as np
from PIL import Image
from pipeline_snellcaster_flux_dual_view import DualSnellcasterPipeline_Flux
from denoising_callbacks_dual_view import dual_tweedie_callback
from snell_flow_match_euler_discrete_scheduler import SnellFlowMatchEulerDiscreteScheduler
from utils.warping import build_laplacian_pyramid, laplacian_pyramid_warping

def load_preprocessing_assets(args):
    """
    Load all preprocessing assets (images and UV maps) once before the denoising loop.
    Also preprocesses main_clean and fg_mask by resizing, warping, and building pyramids.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        tuple: (main_clean, fg_mask, fresnel_map, uv_maps, self_warped_main_clean_image, self_warped_main_clean_laplacian_pyramid, warped_main_clean_image, warped_main_clean_laplacian_pyramid)
    """
    # Load and preprocess images
    main_clean = None
    fg_mask = None
    fresnel_map = None
    
    if args.main_clean_path is not None:
        # Load main clean image and convert to [-1, 1]
        main_clean_img = Image.open(args.main_clean_path).convert("RGB")
        main_clean_np = np.array(main_clean_img).astype(np.float32) / 255.0  # [0, 1]
        main_clean = torch.from_numpy(main_clean_np).permute(2, 0, 1).contiguous()  # [C, H, W]
        main_clean = main_clean * 2.0 - 1.0  # [-1, 1]
        main_clean = main_clean.to(device="cuda", dtype=torch.float16)
        print(f"Loaded main_clean image: {args.main_clean_path}")

    if args.fg_mask_path is not None:
        # Load foreground mask as greyscale and convert to [0, 1]
        fg_mask_img = Image.open(args.fg_mask_path).convert("L")
        fg_mask_np = np.array(fg_mask_img).astype(np.float32) / 255.0  # [0, 1]
        fg_mask = torch.from_numpy(fg_mask_np)  # [H, W]
        fg_mask = fg_mask.to(device="cuda", dtype=torch.float16)
        print(f"Loaded fg_mask image: {args.fg_mask_path}")

    fresnel_map_path = os.path.join(args.warpings_dir, "fresnel_reflection_ratio.png")
    if os.path.exists(fresnel_map_path):
        # Load Fresnel mask as greyscale and convert to [0, 1]
        fresnel_map_img = Image.open(fresnel_map_path).convert("L")
        fresnel_map_np = np.array(fresnel_map_img).astype(np.float32) / 255.0  # [0, 1]
        fresnel_map = torch.from_numpy(fresnel_map_np)  # [H, W]
        fresnel_map = fresnel_map.to(device="cuda", dtype=torch.float16)
        print(f"Loaded fresnel_map image: {fresnel_map_path}")

    # Load UV maps once (to avoid repeated disk I/O during denoising)
    print("Loading UV maps...")
    pano_to_main_uv_path = os.path.join(args.warpings_dir, "pano_to_main_uv.npz")
    pano_to_main_uv_path_reflection = os.path.join(args.warpings_dir, "pano_to_main_uv_reflection.npz")
    main_to_pano_uv_path = os.path.join(args.warpings_dir, "main_to_pano_uv_with_fg.npz")
    main_to_pano_uv_path_without_fg = os.path.join(args.warpings_dir, "main_to_pano_uv_without_fg.npz")
    self_uv_map_path = os.path.join(args.warpings_dir, "self_uv_map.npz")

    uv_maps = {}
    for name, path in [
        ("pano_to_main", pano_to_main_uv_path),
        ("pano_to_main_reflection", pano_to_main_uv_path_reflection),
        ("main_to_pano", main_to_pano_uv_path),
        ("main_to_pano_without_fg", main_to_pano_uv_path_without_fg),
        ("self", self_uv_map_path),
    ]:
        if os.path.exists(path):
            uv_npz = np.load(path)
            uv_maps[name] = uv_npz["uv"].astype(np.float32)
            print(f"  Loaded {name}: {uv_maps[name].shape}")
        else:
            uv_maps[name] = None
            print(f"  Warning: {name} not found at {path}")
    
    # Preprocess main_clean and fg_mask: resize, warp, and build pyramids
    self_warped_main_clean_image = None
    self_warped_main_clean_laplacian_pyramid = None
    warped_main_clean_image = None
    warped_main_clean_laplacian_pyramid = None
    
    if main_clean is not None and fg_mask is not None:
        print("Preprocessing main_clean and fg_mask...")
        # Clone to avoid modifying originals
        main_clean_img = main_clean.clone()  # Already in [-1, 1] range
        fg_mask_img = fg_mask.clone()    # Already in [0, 1] range

        # Resize to match target main image dimensions
        main_h, main_w = args.main_height, args.main_width
        if main_clean_img.shape[1] != main_h or main_clean_img.shape[2] != main_w:
            main_clean_img = F.interpolate(main_clean_img.unsqueeze(0), size=(main_h, main_w), mode='bilinear', align_corners=False).squeeze(0)
        if fg_mask_img.shape[0] != main_h or fg_mask_img.shape[1] != main_w:
            fg_mask_img = F.interpolate(fg_mask_img.unsqueeze(0).unsqueeze(0), size=(main_h, main_w), mode='nearest').squeeze(0).squeeze(0)
        
        # Get self-warped main clean image
        self_uv = uv_maps.get("self")
        if self_uv is not None:
            # laplacian_pyramid_warping expects numpy array and converts internally
            self_warped_main_clean_image = laplacian_pyramid_warping(
                image=main_clean_img,
                uv_map=self_uv,
                levels=args.levels
            )
            # Ensure tensors are on the same device and dtype before torch.where (match main_clean_img)
            if main_clean_img.device != self_warped_main_clean_image.device or main_clean_img.dtype != self_warped_main_clean_image.dtype:
                self_warped_main_clean_image = self_warped_main_clean_image.to(device=main_clean_img.device, dtype=main_clean_img.dtype)
            fg_mask_img = fg_mask_img.to(device=main_clean_img.device, dtype=main_clean_img.dtype)
            fg_mask_nan = torch.where(fg_mask_img == 0, 1.0, torch.nan)
            main_clean_sphere_free = main_clean_img * fg_mask_nan
            self_warped_main_clean_image = torch.where(torch.isnan(main_clean_sphere_free), self_warped_main_clean_image, main_clean_sphere_free)
            
            # Build Laplacian pyramid for the clean image
            self_warped_main_clean_laplacian_pyramid = build_laplacian_pyramid(self_warped_main_clean_image, levels=1)
        else:
            print("  Warning: self_uv not found, skipping main_clean self-warping")
        
        # Get pano-warped main clean image (warped to panorama view)
        main_to_pano_uv_without_fg = uv_maps.get("main_to_pano_without_fg")
        if main_to_pano_uv_without_fg is not None:
            warped_main_clean_image = laplacian_pyramid_warping(
                image=main_clean_img,
                uv_map=main_to_pano_uv_without_fg,
                levels=1
            )
            # Build Laplacian pyramid for the pano-warped clean image
            warped_main_clean_laplacian_pyramid = build_laplacian_pyramid(warped_main_clean_image, levels=1)
        else:
            print("  Warning: main_to_pano_uv_without_fg not found, skipping main_clean pano-warping")
    
    return main_clean, fg_mask, fresnel_map, uv_maps, self_warped_main_clean_image, self_warped_main_clean_laplacian_pyramid, warped_main_clean_image, warped_main_clean_laplacian_pyramid

def parse_args():
    parser = argparse.ArgumentParser(description='Generate dual images (main + panorama) with Snellcaster pipeline')
    parser.add_argument('--image_name', type=str, default='living_room', 
                       help='Base name for generated images (default: living_room)')
    parser.add_argument('--main_seed', type=int, default=42, 
                       help='Seed for main image (default: 42)')
    parser.add_argument('--pano_seed', type=int, default=42,
                       help='Seed for panorama image (default: 42)')
    parser.add_argument('--main_height', type=int, default=720,
                       help='Height for main image (default: 720)')
    parser.add_argument('--main_width', type=int, default=1280,
                       help='Width for main image (default: 1280)')
    parser.add_argument('--pano_height', type=int, default=1024,
                       help='Height for panorama image (default: 1024)')
    parser.add_argument('--pano_width', type=int, default=2048,
                       help='Width for panorama image (default: 2048)')
    parser.add_argument('--num_steps', type=int, default=20,
                       help='Number of inference steps (default: 20)')
    parser.add_argument('--main_guidance_scale', type=float, default=3.5,
                       help='Guidance scale for main image (default: 7.0)')
    parser.add_argument('--pano_guidance_scale', type=float, default=3.5,
                       help='Guidance scale for panorama image (default: 3.5)')
    parser.add_argument('--main_prompt', type=str, 
                       default='A living room with a sofa with colorful patterned cushions, and beside the sofa stands a bookshelf filled with books. Right in front of the sofa is a clean wooden coffee table with a big glass sphere on top, refracting the background and cast shadows and caustics on the table. The entire scene is in sharp focus, including the background and all furniture.',
                       help='Prompt for the main image')
    parser.add_argument('--pano_prompt', type=str,
                       default='A high-resolution equirectangular 360-degree panorama of a modern living room full of furniture with smooth walls and a ceiling, captured from a camera placed on a clean wooden coffee table. Seamless, continuous surfaces with uniform paint and non-repetitive textures; High quality, high detail, high resolution.',
                       help='Prompt for the panorama image (designed to match main image)')
    parser.add_argument('--negative_prompt', type=str,
                       default='blurry background, shallow depth of field, soft focus, low detail background, washed out colors, inconsistent colors, mismatched wall textures, conflicting room designs, different lighting conditions, grid artifacts, checkerboard pattern, mosaic pattern, pixelation, blocky texture',
                       help='Negative prompt for both images')
    parser.add_argument('--model_name', type=str, default='black-forest-labs/FLUX.1-dev',
                       help='Model name to use for generation (default: black-forest-labs/FLUX.1-dev)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Detail-preserving averaging parameter (0.0 = normal averaging, 1.0 = value-weighted averaging, default: 0.5)')
    parser.add_argument('--levels', type=int, default=1,
                       help='Number of pyramid levels for Laplacian blending (default: 1)')
    parser.add_argument('--blend_step_ratio', type=float, default=1.0,
                       help='Fraction of steps during which to apply blending (default: 0.8, meaning blend only in first 80%% of steps)')
    parser.add_argument('--time_travel_repeats', type=int, default=2,
                       help='Number of sub-steps (repeats) to perform during time travel (default: 2)')
    parser.add_argument('--time_travel_start_ratio', type=float, default=0.2,
                       help='Start ratio (0.0-1.0) of the denoising process when time travel begins (default: 0.2)')
    parser.add_argument('--time_travel_end_ratio', type=float, default=0.8,
                       help='End ratio (0.0-1.0) of the denoising process when time travel ends (default: 0.8)')
    parser.add_argument('--warpings_dir', type=str, default='results/warpings/',
                       help='Directory containing warping NPZ files (default: results/warpings/)')
    parser.add_argument('--output_dir', type=str, default='results/dual_views',
                       help='Base output directory for generated images (default: results/dual_views)')
    parser.add_argument('--save_dir', type=str, default='results/tweedie_estimates',
                       help='Directory to save tweedie estimates (default: results/tweedie_estimates)')
    parser.add_argument('--main_clean_path', type=str, default=None,
                       help='Path to main clean image (default: None)')
    parser.add_argument('--fg_mask_path', type=str, default=None,
                       help='Path to foreground mask (default: None)')
    parser.add_argument('--intermediate_vis', action='store_true',
                       help='Enable intermediate visualization (default: False)')
    return parser.parse_args()

# Create separate schedulers for main and panorama images
main_scheduler = SnellFlowMatchEulerDiscreteScheduler(stochastic_sampling=True)
pano_scheduler = SnellFlowMatchEulerDiscreteScheduler(stochastic_sampling=True)

# Parse command line arguments
args = parse_args()

start_time = time.time()

pipe = DualSnellcasterPipeline_Flux.from_pretrained(
    args.model_name,
    torch_dtype=torch.float16,
    scheduler=main_scheduler,  # Default scheduler
    main_scheduler=main_scheduler,
    pano_scheduler=pano_scheduler,
    device_map="balanced"
)

seed = args.pano_seed

# Ensure no leftovers from previous runs
for attr in [
    'main_tweedie_estimates', 'main_tweedie_images',
    'pano_tweedie_estimates', 'pano_tweedie_images'
]:
    if hasattr(pipe, attr):
        delattr(pipe, attr)

# Generators: fixed main seed to match geometry, pano varies
generator_main = torch.Generator(device="cuda").manual_seed(args.main_seed)
generator_pano = torch.Generator(device="cuda").manual_seed(seed)

# Output directory encoding parameters
param_str = (
    f"seed{seed}_a{args.alpha:.2f}_L{args.levels}_"
    f"tt{args.time_travel_start_ratio:.2f}-{args.time_travel_end_ratio:.2f}x{args.time_travel_repeats}_steps{args.num_steps}"
)
base_output_dir = args.output_dir
os.makedirs(base_output_dir, exist_ok=True)
# Save param_str alongside the outputs for bookkeeping
try:
    with open(os.path.join(base_output_dir, "param_str.txt"), "w") as pf:
        pf.write(param_str)
except Exception as e:
    print(f"Warning: failed to write param_str: {e}")

# Load all preprocessing assets (images and UV maps) once
main_clean, fg_mask, fresnel_map, uv_maps, self_warped_main_clean_image, self_warped_main_clean_laplacian_pyramid, warped_main_clean_image, warped_main_clean_laplacian_pyramid = load_preprocessing_assets(args)

# Build callback
denoising_callback = lambda pipeline, step, timestep, main_h, main_w, pano_h, pano_w, main_tweedie, pano_tweedie: dual_tweedie_callback(
    pipeline, step, timestep, main_h, main_w, pano_h, pano_w, base_output_dir, main_tweedie, pano_tweedie,
    save_dir=os.path.join(base_output_dir, "tweedie_estimates"),
    store_estimates=False,
    save_single_immediately=False,
    save_grid_at_end=False,
    total_steps=args.num_steps,
    warpings_dir=args.warpings_dir,
    alpha=args.alpha,
    levels=args.levels,
    blend_step_ratio=args.blend_step_ratio,
    main_clean=main_clean,
    fg_mask=fg_mask,
    fresnel_map=fresnel_map,
    uv_maps=uv_maps,
    self_warped_main_clean_image=self_warped_main_clean_image,
    self_warped_main_clean_laplacian_pyramid=self_warped_main_clean_laplacian_pyramid,
    warped_main_clean_image=warped_main_clean_image,
    warped_main_clean_laplacian_pyramid=warped_main_clean_laplacian_pyramid,
    intermediate_vis=args.intermediate_vis
)

# Run generation
result = pipe(
    main_prompt=args.main_prompt,
    pano_prompt=args.pano_prompt,
    negative_prompt=args.negative_prompt,
    main_height=args.main_height,
    main_width=args.main_width,
    pano_height=args.pano_height,
    pano_width=args.pano_width,
    main_guidance_scale=args.main_guidance_scale,
    pano_guidance_scale=args.pano_guidance_scale,
    num_inference_steps=args.num_steps,
    generator_main=generator_main,
    generator_pano=generator_pano,
    callback_on_denoising=denoising_callback,
    time_travel_repeats=args.time_travel_repeats,
    time_travel_start_ratio=args.time_travel_start_ratio,
    time_travel_end_ratio=args.time_travel_end_ratio,
)

# Save final images
main_image, pano_image = result.images
main_filename = f"{base_output_dir}/main.jpg"
pano_filename = f"{base_output_dir}/pano.jpg"
main_image[0].save(main_filename)
pano_image[0].save(pano_filename)
print(
    f"Generated: seed={seed}, b={args.blend_step_ratio}, L={args.levels}, TT={args.time_travel_start_ratio}-{args.time_travel_end_ratio} -> {main_filename}, {pano_filename}"
)

# End timing and calculate duration
end_time = time.time()
generation_time = end_time - start_time

print(f"Dual generation completed in {generation_time:.2f} seconds (seed: {seed})")

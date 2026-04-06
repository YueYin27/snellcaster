"""
Dual-image denoising callbacks for Snellcaster pipeline.
This module provides callbacks that can process two latents simultaneously.
"""

import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from typing import List, Optional
from torchvision import transforms
from utils.blending import laplacian_pyramid_blending, reconstruction
from utils.images import *
from utils.images import tensor_to_pil, create_pyramid_visualization
from utils.warping import build_gaussian_pyramid, build_laplacian_pyramid, laplacian_pyramid_warping


def decoder(pipeline, latents, height, width):
    """
    Decode latent tensor to pixel space image.
    
    Args:
        pipeline: The pipeline instance
        latent: Latent tensor to decode
        height: The original image height
        width: The original image width
    
    Returns:
        List of PIL images
    """
    with torch.no_grad():
        latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type="pil")
        
        return image


def encoder(pipeline, image, height, width):
    """
    Encode PIL image to latent tensor.
    
    Args:
        pipeline: The pipeline instance
        image: PIL image or list containing single PIL image
        height: The original image height
        width: The original image width
    
    Returns:
        Encoded latent tensor in packed format
    """
    with torch.no_grad():
        # Ensure image is a list
        if not isinstance(image, list):
            image = [image]
        
        # Preprocess image to tensor format (reverse of postprocess in decoder)
        processed_image = pipeline.image_processor.preprocess(image, height=height, width=width)
        
        # Move to device if needed
        if hasattr(pipeline, 'device'):
            processed_image = processed_image.to(pipeline.device)
        
        # Convert to same dtype as VAE model
        vae_dtype = next(pipeline.vae.parameters()).dtype
        processed_image = processed_image.to(dtype=vae_dtype)
        
        # Encode to latent (reverse of decode in decoder)
        latent = pipeline.vae.encode(processed_image, return_dict=False)[0]
        
        # Extract the latent tensor if it's a DiagonalGaussianDistribution
        if hasattr(latent, 'sample'):
            latent = latent.sample()
        elif hasattr(latent, 'mean'):
            latent = latent.mean()
        elif hasattr(latent, 'latent_dist'):
            latent = latent.latent_dist.sample()
        
        # Denormalize latent (reverse of normalization in decoder)
        # Decoder: (latents / scaling_factor) + shift_factor
        # Encoder: (latents - shift_factor) * scaling_factor
        latent = (latent - pipeline.vae.config.shift_factor) * pipeline.vae.config.scaling_factor
        
        # Pack latent (reverse of unpack in decoder)
        batch_size = latent.shape[0]
        num_channels_latents = latent.shape[1]
        latent_height = latent.shape[2]
        latent_width = latent.shape[3]
        packed_latent = pipeline._pack_latents(latent, batch_size, num_channels_latents, latent_height, latent_width)
        
        return packed_latent


def dual_tweedie_callback(
    pipeline, 
    step: int, 
    timestep: int, 
    main_height: int,
    main_width: int,
    pano_height: int,
    pano_width: int,
    base_output_dir: str,
    main_tweedie_latent=None,
    pano_tweedie_latent=None,
    save_dir: str = "tweedie_estimates",
    main_image_name: str = "main",
    pano_image_name: str = "pano",
    store_estimates: bool = True,
    total_steps: int = None,
    save_step: int = 1,
    save_single_immediately: bool = True,
    save_grid_at_end: bool = True,
    warpings_dir: str = 'results/warpings/',
    alpha: float = 0.5,
    levels: int = 1,
    blend_step_ratio: float = 1.0,
    main_clean: torch.Tensor = None,
    fg_mask: torch.Tensor = None,
    fresnel_map: torch.Tensor = None,
    uv_maps: dict = None,
    self_warped_main_clean_image: torch.Tensor = None,
    self_warped_main_clean_laplacian_pyramid: list = None,
    warped_main_clean_image: torch.Tensor = None,
    warped_main_clean_laplacian_pyramid: list = None,
    intermediate_vis: bool = False
):
    """
    Dual-image Tweedie estimate callback that processes both main and panorama latents together.
    
    This callback:
    1. Gets tweedie estimates for both latents
    2. Decodes both to image space
    3. Gets pixel correspondences using the npz file
    4. Aggregates them together
    5. Returns updated latents for both images
    
    Args:
        pipeline: The pipeline instance
        step: Current denoising step
        timestep: Current timestep value
        main_height: The main image height
        main_width: The main image width
        pano_height: The panorama image height
        pano_width: The panorama image width
        main_tweedie_latent: Tweedie estimate for main image (if available)
        pano_tweedie_latent: Tweedie estimate for panorama (if available)
        save_dir: Directory to save images
        main_image_name: Base name for main images
        pano_image_name: Base name for panorama images
        store_estimates: Whether to store estimates for grid creation
        total_steps: Total number of denoising steps
        save_step: Save every N steps
        save_single_immediately: Whether to save individual images immediately
        save_grid_at_end: Whether to save grid at the end
        warpings_dir: Directory containing warping NPZ files (default: 'results/warpings/')
        alpha: Detail-preserving averaging parameter (0.0 = normal averaging, 1.0 = value-weighted averaging)
        levels: Number of pyramid levels for Laplacian blending (default: 5)
        blend_step_ratio: Fraction of steps during which to apply blending (default: 0.8, meaning blend only in first 80%% of steps)
        main_clean: Preprocessed main clean image tensor in [-1, 1] range (default: None)
        fg_mask: Preprocessed foreground mask tensor in [0, 1] range (default: None)
        fresnel_map: Preprocessed Fresnel mask tensor in [0, 1] range (default: None)
        uv_maps: Dictionary of preloaded UV maps (default: None, will load from warpings_dir)
        self_warped_main_clean_image: Preprocessed self-warped main clean image (default: None)
        self_warped_main_clean_laplacian_pyramid: Preprocessed Laplacian pyramid for self-warped main clean (default: None)
        warped_main_clean_image: Preprocessed pano-warped main clean image (default: None)
        warped_main_clean_laplacian_pyramid: Preprocessed Laplacian pyramid for pano-warped main clean (default: None)
        intermediate_vis: Whether to create intermediate visualizations (default: False)
        base_output_dir: Base output directory for saving intermediate results (lpb composites). Required.
    
    Returns:
        Tuple of (updated_main_latent, updated_pano_latent)
    """
    # Use preloaded UV maps if provided, otherwise construct paths (backward compatibility)
    if uv_maps is not None:
        # Use preloaded UV maps (efficient)
        pano_to_main_uv = uv_maps.get("pano_to_main")
        pano_to_main_uv_reflection = uv_maps.get("pano_to_main_reflection")
        main_to_pano_uv = uv_maps.get("main_to_pano")
        main_to_pano_uv_without_fg = uv_maps.get("main_to_pano_without_fg")
        self_uv = uv_maps.get("self")
    else:
        # Fallback to paths (backward compatibility, less efficient)
        pano_to_main_uv = os.path.join(warpings_dir, "pano_to_main_uv.npz")
        pano_to_main_uv_reflection = os.path.join(warpings_dir, "pano_to_main_uv_reflection.npz")
        main_to_pano_uv = os.path.join(warpings_dir, "main_to_pano_uv_with_fg.npz")
        main_to_pano_uv_without_fg = os.path.join(warpings_dir, "main_to_pano_uv_without_fg.npz")
        self_uv = os.path.join(warpings_dir, "self_uv_map.npz")
    
    # Determine if we should blend at this step. We avoid blending on the final step
    # so the final output remains the natural denoised result from the model.
    is_final_step = (total_steps is not None) and (step >= total_steps - 1)
    should_blend = (not is_final_step) and (step < blend_step_ratio * total_steps)

    if should_blend:
        # 1. Get updated tweedie images
        # Get tweedie latents and decode to images
        main_tweedie_latent = main_tweedie_latent.to(dtype=torch.float16)
        pano_tweedie_latent = pano_tweedie_latent.to(dtype=torch.float16)
        
        main_tweedie_image = decoder(pipeline, main_tweedie_latent, main_height, main_width)[0]
        pano_tweedie_image = decoder(pipeline, pano_tweedie_latent, pano_height, pano_width)[0]

        # # Add a stronger blur to the panorama tweedie image for first few steps
        # if step == 0:
        #     pano_tweedie_image = pano_tweedie_image.filter(ImageFilter.GaussianBlur(radius=5.0))
        
        # Apply laplacian pyramid blending for both images in pixel space
        updated_main_tweedie_image, updated_pano_tweedie_image = lp_warping_blending(
            main_tweedie_image,
            pano_tweedie_image,
            base_output_dir,
            pano_to_main_uv,
            pano_to_main_uv_reflection,
            main_to_pano_uv,
            levels,
            alpha,
            step,
            total_steps,
            fresnel_map=fresnel_map,
            self_warped_main_clean_image=self_warped_main_clean_image,
            self_warped_main_clean_laplacian_pyramid=self_warped_main_clean_laplacian_pyramid,
            warped_main_clean_image=warped_main_clean_image,
            warped_main_clean_laplacian_pyramid=warped_main_clean_laplacian_pyramid,
            intermediate_vis=intermediate_vis
    )
        
        # Encode back to latent space
        updated_main_tweedie_latent = encoder(pipeline, updated_main_tweedie_image, main_height, main_width)
        updated_pano_tweedie_latent = encoder(pipeline, updated_pano_tweedie_image, pano_height, pano_width)
    
        # # 2. Get updated residuals in latent space
        # # Compute residuals for both images
        # main_residual = main_tweedie_latent - encoder(pipeline, main_tweedie_image, main_height, main_width)
        # pano_residual = pano_tweedie_latent - encoder(pipeline, pano_tweedie_image, pano_height, pano_width)

        # # Decode residuals to image space
        # main_residual_image = decoder(pipeline, main_residual, main_height, main_width)[0]
        # pano_residual_image = decoder(pipeline, pano_residual, pano_height, pano_width)[0]

        # # Apply equivalent transformations in pixel space for residuals using detail-preserving averaging
        # updated_main_residual_image, updated_pano_residual_image = lp_warping_blending(
        #     main_residual_image,
        #     pano_residual_image,
        #     pano_to_main_uv_path="results/warpings/pano_to_main_uv.npz",
        #     main_to_pano_uv_path="results/warpings/main_to_pano_uv.npz",
        #     levels=5,
        #     alpha=alpha,
        #     step=step,
        #     pano_to_main_mask_path="results/warpings/mask_pano_main.png",
        #     main_to_pano_mask_path="results/warpings/mask_main_pano.png",
        # )

        # # Encode back to latent space
        # updated_main_residual_latent = encoder(pipeline, updated_main_residual_image, main_height, main_width)
        # updated_pano_residual_latent = encoder(pipeline, updated_pano_residual_image, pano_height, pano_width)

        # # Apply residual corrections to preserve original information
        # updated_main_tweedie_latent += updated_main_residual_latent
        # updated_pano_tweedie_latent += updated_pano_residual_latent
        
        # Save images before blending
        if save_single_immediately:
            save_single_tweedie_image(
                main_tweedie_image, save_dir, step, timestep, 
                f"{main_image_name}"
            )
            save_single_tweedie_image(
                pano_tweedie_image, save_dir, step, timestep, 
                f"{pano_image_name}"
            )
        
        # Store for later use if requested
        if store_estimates:
            # Store main image estimates
            if not hasattr(pipeline, 'main_tweedie_estimates'):
                pipeline.main_tweedie_estimates = []
            pipeline.main_tweedie_estimates.append((step, timestep, main_tweedie_latent))
            
            if not hasattr(pipeline, 'main_tweedie_images'):
                pipeline.main_tweedie_images = []
            pipeline.main_tweedie_images.append((step, timestep, main_tweedie_image))
            
            # Store panorama estimates
            if not hasattr(pipeline, 'pano_tweedie_estimates'):
                pipeline.pano_tweedie_estimates = []
            pipeline.pano_tweedie_estimates.append((step, timestep, pano_tweedie_latent))
            
            if not hasattr(pipeline, 'pano_tweedie_images'):
                pipeline.pano_tweedie_images = []
            pipeline.pano_tweedie_images.append((step, timestep, pano_tweedie_image))
        
        # Check if this is the final step and create grids
        if save_grid_at_end and total_steps is not None and step == total_steps - 1:
            if hasattr(pipeline, 'main_tweedie_images') and pipeline.main_tweedie_images:
                save_tweedie_images_grid(
                    pipeline.main_tweedie_images, save_dir, total_steps, 
                    main_image_name, save_step
                )
            if hasattr(pipeline, 'pano_tweedie_images') and pipeline.pano_tweedie_images:
                save_tweedie_images_grid(
                    pipeline.pano_tweedie_images, save_dir, total_steps, 
                    pano_image_name, save_step
            )
        
        return updated_main_tweedie_latent, updated_pano_tweedie_latent
    
    else:
        # No blending: still decode and optionally save the natural Tweedie images
        main_tweedie_latent = main_tweedie_latent.to(dtype=torch.float16)
        pano_tweedie_latent = pano_tweedie_latent.to(dtype=torch.float16)

        main_tweedie_image = decoder(pipeline, main_tweedie_latent, main_height, main_width)[0]
        pano_tweedie_image = decoder(pipeline, pano_tweedie_latent, pano_height, pano_width)[0]

        if save_single_immediately:
            save_single_tweedie_image(
                main_tweedie_image, save_dir, step, timestep,
                f"{main_image_name}"
            )
            save_single_tweedie_image(
                pano_tweedie_image, save_dir, step, timestep,
                f"{pano_image_name}"
            )

        if store_estimates:
            if not hasattr(pipeline, 'main_tweedie_estimates'):
                pipeline.main_tweedie_estimates = []
            pipeline.main_tweedie_estimates.append((step, timestep, main_tweedie_latent))

            if not hasattr(pipeline, 'main_tweedie_images'):
                pipeline.main_tweedie_images = []
            pipeline.main_tweedie_images.append((step, timestep, main_tweedie_image))

            if not hasattr(pipeline, 'pano_tweedie_estimates'):
                pipeline.pano_tweedie_estimates = []
            pipeline.pano_tweedie_estimates.append((step, timestep, pano_tweedie_latent))

            if not hasattr(pipeline, 'pano_tweedie_images'):
                pipeline.pano_tweedie_images = []
            pipeline.pano_tweedie_images.append((step, timestep, pano_tweedie_image))

        # If this is the final step, save grids even without blending
        if save_grid_at_end and total_steps is not None and step == total_steps - 1:
            if hasattr(pipeline, 'main_tweedie_images') and pipeline.main_tweedie_images:
                save_tweedie_images_grid(
                    pipeline.main_tweedie_images, save_dir, total_steps,
                    main_image_name, save_step
                )
            if hasattr(pipeline, 'pano_tweedie_images') and pipeline.pano_tweedie_images:
                save_tweedie_images_grid(
                    pipeline.pano_tweedie_images, save_dir, total_steps,
                    pano_image_name, save_step
                )

        return main_tweedie_latent, pano_tweedie_latent


def create_intermediate_visualization(
    main_tensor,
    pano_tensor,
    main_gaussian_pyramid,
    main_laplacian_pyramid,
    warped_pano_image,
    warped_pano_laplacian_pyramid,
    warped_main_image,
    warped_main_laplacian_pyramid,
    reconstructed_image_main,
    reconstructed_image_pano,
    skip_main_clean_blending,
    skip_pano_blending,
    self_warped_main_clean_image,
    self_warped_main_clean_laplacian_pyramid,
    warped_main_clean_image,
    warped_main_clean_laplacian_pyramid,
    levels,
    base_output_dir,
    step
):
    """
    Create visualization images for debugging and analysis.
    
    Args:
        main_tensor: Main image tensor [C, H, W] in [-1, 1] range
        pano_tensor: Panorama image tensor [C, H, W] in [-1, 1] range
        main_gaussian_pyramid: Gaussian pyramid for main image
        main_laplacian_pyramid: Laplacian pyramid for main image
        warped_pano_image: Warped panorama image tensor
        warped_pano_laplacian_pyramid: Laplacian pyramid for warped panorama
        warped_main_image: Warped main image tensor
        warped_main_laplacian_pyramid: Laplacian pyramid for warped main
        reconstructed_image_main: Reconstructed main image tensor
        reconstructed_image_pano: Reconstructed panorama image tensor
        skip_main_clean_blending: Whether main clean blending was skipped
        skip_pano_blending: Whether pano blending was skipped
        self_warped_main_clean_image: Self-warped main clean image tensor
        self_warped_main_clean_laplacian_pyramid: Laplacian pyramid for self-warped main clean
        warped_main_clean_image: Pano-warped main clean image tensor
        warped_main_clean_laplacian_pyramid: Laplacian pyramid for pano-warped main clean
        levels: Number of pyramid levels
        base_output_dir: Base output directory for saving
        step: Current denoising step
    """
    # Build visualization PILs needed for the layout
    main_gauss_pil = create_pyramid_visualization(main_gaussian_pyramid, "Main G", normalize=True)
    main_lap_pil = create_pyramid_visualization(main_laplacian_pyramid, "Main L", normalize=True)
    warped_pano_pil = tensor_to_pil(warped_pano_image)
    warped_lap_pil = create_pyramid_visualization(warped_pano_laplacian_pyramid, "Warped Pano L", normalize=True)
    
    # Self-warped main clean visuals (only if not skipping main clean blending)
    if not skip_main_clean_blending and self_warped_main_clean_image is not None:
        self_warped_main_clean_pil = tensor_to_pil(self_warped_main_clean_image)
        self_warped_main_clean_lap_pil = create_pyramid_visualization(self_warped_main_clean_laplacian_pyramid, "Self-Warped Main Clean L", normalize=True)
    else:
        self_warped_main_clean_pil = None
        self_warped_main_clean_lap_pil = None
    
    # Warped main visuals (always created)
    warped_main_pil = tensor_to_pil(warped_main_image)
    warped_main_gaussian_pyramid = build_gaussian_pyramid(warped_main_image, levels=levels)
    warped_main_gauss_pil = create_pyramid_visualization(warped_main_gaussian_pyramid, "Warped Main G", normalize=True)
    warped_main_lap_pil = create_pyramid_visualization(warped_main_laplacian_pyramid, "Warped Main L", normalize=True)
    
    # Panorama visuals (always created)
    panorama_gaussian_pyramid = build_gaussian_pyramid(pano_tensor, levels=levels)
    panorama_gauss_pil = create_pyramid_visualization(panorama_gaussian_pyramid, "Pano G", normalize=True)
    panorama_laplacian_pyramid = build_laplacian_pyramid(pano_tensor, levels=levels)
    panorama_lap_pil = create_pyramid_visualization(panorama_laplacian_pyramid, "Pano L", normalize=True)
    
    # Warped main clean visuals (only if pano blending is not skipped)
    if not skip_pano_blending and warped_main_clean_image is not None:
        warped_main_clean_pil = tensor_to_pil(warped_main_clean_image)
        warped_main_clean_gaussian_pyramid = build_gaussian_pyramid(warped_main_clean_image, levels=levels)
        warped_main_clean_gauss_pil = create_pyramid_visualization(warped_main_clean_gaussian_pyramid, "Warped Main Clean G", normalize=True)
        warped_main_clean_lap_pil = create_pyramid_visualization(warped_main_clean_laplacian_pyramid, "Warped Main Clean L", normalize=True)
    else:
        warped_main_clean_pil = None
        warped_main_clean_gauss_pil = None
        warped_main_clean_lap_pil = None

    # Ensure both tensors are on the same device
    device = pano_tensor.device
    reconstructed_image_pano = reconstructed_image_pano.to(device)
  
    # Build Gaussian pyramid for reconstructed pano (blended result)
    blended_gaussian_pyramid_pano = build_gaussian_pyramid(reconstructed_image_pano, levels=levels)
    blended_gauss_pil_pano = create_pyramid_visualization(blended_gaussian_pyramid_pano, "Blended Pano G", normalize=True)
    
    # Convert reconstructed images to PIL for visualization (tensors are still in [-1, 1] range)
    reconstructed_main_pil = tensor_to_pil(reconstructed_image_main)
    reconstructed_pano_pil = tensor_to_pil(reconstructed_image_pano)
    
    # Visualization images are created but not saved (can be extended to save if needed)
    # For now, this function just creates the visualizations without saving them
    pass


def lp_warping_blending(
    main_image,
    pano_image,
    base_output_dir: str,
    pano_to_main_uv,
    pano_to_main_uv_reflection,
    main_to_pano_uv,
    levels,
    alpha,
    step,
    total_steps,
    fresnel_map: torch.Tensor = None,
    self_warped_main_clean_image: torch.Tensor = None,
    self_warped_main_clean_laplacian_pyramid: list = None,
    warped_main_clean_image: torch.Tensor = None,
    warped_main_clean_laplacian_pyramid: list = None,
    intermediate_vis: bool = False
):
    """
    1. Builds Gaussian and Laplacian pyramids from main image
    2. Warps the panorama image to the main view and builds its Laplacian pyramid
    3. Blends the two Laplacian pyramids in the main view
    4. Reconstructs the final blended image from the blended Laplacian pyramid
    5. Warps the reconstructed image back to the panorama view
    6. Returns the reconstructed image and the warped back panorama image
    
    Args:
        main_image: Main image (tensor or PIL Image)
        pano_image: Panorama image (tensor or PIL Image)
        base_output_dir: Base output directory for saving intermediate results (required)
        pano_to_main_uv: Pano-to-main UV mapping (numpy array or path string)
        pano_to_main_uv_reflection: Pano-to-main UV mapping for reflection (numpy array or path string)
        main_to_pano_uv: Main-to-pano UV mapping (numpy array or path string)
        levels: Number of pyramid levels
        alpha: Blending parameter
        step: Current denoising step for saving
        total_steps: Total number of denoising steps
        fresnel_map: Preprocessed Fresnel mask tensor in [0, 1] range (default: None)
        self_warped_main_clean_image: Preprocessed self-warped main clean image (default: None)
        self_warped_main_clean_laplacian_pyramid: Preprocessed Laplacian pyramid for self-warped main clean (default: None)
        warped_main_clean_image: Preprocessed pano-warped main clean image (default: None)
        warped_main_clean_laplacian_pyramid: Preprocessed Laplacian pyramid for pano-warped main clean (default: None)
        intermediate_vis: Whether to create intermediate visualizations (default: False)
    
    Returns:
        Tuple of (reconstructed_image, warped_back_to_pano)
    """
    # # Create output directory with step
    # out_dir = os.path.join(base_output_dir, "lpb")
    # os.makedirs(out_dir, exist_ok=True)

    # Step 1: Build pyramids from main image
    # Convert PIL Image to tensor format [C, H, W]
    if hasattr(main_image, 'dim'):  # Already a tensor
        if main_image.dim() == 4:  # [B, C, H, W]
            main_tensor = main_image.squeeze(0)  # Remove batch dimension
        else:
            main_tensor = main_image  # Already [C, H, W]
    else:  # PIL Image
        main_np = np.array(main_image).astype(np.float32) / 255.0
        main_tensor = torch.from_numpy(main_np).permute(2, 0, 1).contiguous()  # [C, H, W]
    
    # Normalize main_tensor to [-1, 1] and build pyramids
    main_tensor = main_tensor * 2.0 - 1.0
    main_laplacian_pyramid = build_laplacian_pyramid(main_tensor, levels=1)
    
    # Step 2: Warp panorama and build pyramids
    # Convert PIL Image to tensor format [C, H, W]
    if hasattr(pano_image, 'dim'):  # Already a tensor
        if pano_image.dim() == 4:  # [B, C, H, W]
            pano_tensor = pano_image.squeeze(0)  # Remove batch dimension
        else:
            pano_tensor = pano_image  # Already [C, H, W]
    else:  # PIL Image
        pano_np = np.array(pano_image).astype(np.float32) / 255.0
        pano_tensor = torch.from_numpy(pano_np).permute(2, 0, 1).contiguous()  # [C, H, W]
    
    # Normalize pano_tensor to [-1, 1] and build Laplacian pyramid
    pano_tensor = pano_tensor * 2.0 - 1.0
    warped_pano_image = laplacian_pyramid_warping(image=pano_tensor, uv_map=pano_to_main_uv, levels=levels)
    warped_pano_laplacian_pyramid = build_laplacian_pyramid(warped_pano_image, levels=1)

    # Check if we should skip main clean blending (after 40% of steps)
    skip_main_clean_blending = (total_steps is not None) and (step > 1.0 * total_steps)

    # Step 3: Blend Laplacian pyramids and reconstruct the image
    if skip_main_clean_blending:
        # After xx% of steps, only blend main and warped pano (exclude clean)
        blended_laplacian_pyramid_main_rfr = laplacian_pyramid_blending(
            [main_laplacian_pyramid, warped_pano_laplacian_pyramid],
            alpha=alpha,
        )
    else:
        # First xx% of steps, blend all three (clean, main, warped pano)
        blended_laplacian_pyramid_main_rfr = laplacian_pyramid_blending(
            [self_warped_main_clean_laplacian_pyramid, main_laplacian_pyramid, warped_pano_laplacian_pyramid],
            alpha=alpha,
        )
    reconstructed_image_main_rfr = reconstruction(blended_laplacian_pyramid_main_rfr)

    # Step 4: Apply Fresnel masks and blend the two Laplacian pyramids
    # Use preprocessed fresnel mask tensor (already in [0, 1] range)
    device = pano_tensor.device
    fresnel_map_img = fresnel_map.clone().to(device)
    fresnel_map_rfl = torch.where(fresnel_map_img > 0, fresnel_map_img, torch.tensor(0.0, device=device))
    fresnel_map_rfr = torch.where(fresnel_map_img > 0, (1 - fresnel_map_img), torch.tensor(1.0, device=device))

    # 1) Warp the panorama image to the main view for reflection and build its Laplacian pyramid
    warped_pano_image_rfl = laplacian_pyramid_warping(image=pano_tensor, uv_map=pano_to_main_uv_reflection, levels=levels)

    # 2) Convert from [-1, 1] to [0, 1] and apply Fresnel mask
    warped_pano_image_rfl = (warped_pano_image_rfl.to(device) + 1) / 2
    reconstructed_image_main_rfr = (reconstructed_image_main_rfr.to(device) + 1) / 2

    # 3) Convert from sRGB to linear RGB space for physically correct Fresnel application
    warped_pano_image_rfl_linear = srgb_to_linear(warped_pano_image_rfl)
    reconstructed_image_main_rfr_linear = srgb_to_linear(reconstructed_image_main_rfr)
    
    # 4) Apply Fresnel masks in linear space and add the two images while ignore NaNs
    warped_pano_image_rfl_linear = fresnel_map_rfl * warped_pano_image_rfl_linear
    reconstructed_image_main_rfr_linear = fresnel_map_rfr * reconstructed_image_main_rfr_linear
    reconstructed_image_main = torch.where(torch.isnan(warped_pano_image_rfl_linear), reconstructed_image_main_rfr_linear, warped_pano_image_rfl_linear + reconstructed_image_main_rfr_linear)
    reconstructed_image_main = linear_to_srgb(reconstructed_image_main)  # Convert back from linear RGB to sRGB
    reconstructed_image_main = reconstructed_image_main * 2 - 1  # Convert back from [0, 1] to [-1, 1]

    # Check if we should skip pano blending (after xx% of steps)
    skip_pano_blending = (total_steps is not None) and (step > 1.0 * total_steps)
    
    if skip_pano_blending:
        # After xx% of steps, only blend warped main (no warped clean) with pano
        # Step 6: Get warped main image only (skip clean)
        warped_main_image = laplacian_pyramid_warping(
            image=main_tensor,
            uv_map=main_to_pano_uv,
            levels=levels
        )
        
        # Set clean-related placeholders to None (not used in this phase)
        warped_main_clean_image = None
        warped_main_clean_laplacian_pyramid = None
        
        # Step 7: Build Laplacian pyramids from warped main image and original panorama image
        warped_main_laplacian_pyramid = build_laplacian_pyramid(warped_main_image, levels=levels)
        panorama_laplacian_pyramid = build_laplacian_pyramid(pano_tensor, levels=levels)
        
        # Step 8: Blend only two Laplacian pyramids (warped main + pano, no clean) and reconstruct the image
        blended_laplacian_pyramid_pano = laplacian_pyramid_blending(
            [warped_main_laplacian_pyramid, panorama_laplacian_pyramid],
            alpha=alpha,
        )
        reconstructed_image_pano = reconstruction(blended_laplacian_pyramid_pano)
    else:
        # Step 5: Get warped main image (warped main clean image already preprocessed)
        warped_main_image = laplacian_pyramid_warping(image=main_tensor, uv_map=main_to_pano_uv, levels=1)
        
        # Step 6: Build Laplacian pyramids from warped main image and original panorama image
        warped_main_laplacian_pyramid = build_laplacian_pyramid(warped_main_image, levels=1)
        panorama_laplacian_pyramid = build_laplacian_pyramid(pano_tensor, levels=1)

        # # print the first 10 pixels of the first level of the three Laplacian pyramids
        # print("shape of warped_main_clean_laplacian_pyramid:", warped_main_clean_laplacian_pyramid[0].shape)
        # print("warped_main_clean_laplacian_pyramid:", warped_main_clean_laplacian_pyramid[0][:, 5, 10])
        # print("warped_main_laplacian_pyramid:", warped_main_laplacian_pyramid[0][:, 5, 10])
        # print("panorama_laplacian_pyramid:", panorama_laplacian_pyramid[0][:, 5, 10])
        
        # Step 7: Blend three Laplacian pyramids and reconstruct the image
        blended_laplacian_pyramid_pano = laplacian_pyramid_blending(
            [warped_main_clean_laplacian_pyramid, warped_main_laplacian_pyramid, panorama_laplacian_pyramid],
            alpha=alpha,
        )

        # print the first 10 pixels of the first level of the blended Laplacian pyramid
        # print("blended_laplacian_pyramid_pano:", blended_laplacian_pyramid_pano[0][:, 5, 10])

        reconstructed_image_pano = reconstruction(blended_laplacian_pyramid_pano)
        # print("reconstructed_image_pano:", reconstructed_image_pano[:, 5, 10])

        # exit(0)
    
    # Convert to PIL for return
    reconstructed_main = tensor_to_pil(reconstructed_image_main)
    reconstructed_pano = tensor_to_pil(reconstructed_image_pano)

    # Create intermediate visualizations if requested
    if intermediate_vis:
        create_intermediate_visualization(
            main_tensor=main_tensor,
            pano_tensor=pano_tensor,
            main_gaussian_pyramid=main_gaussian_pyramid,
            main_laplacian_pyramid=main_laplacian_pyramid,
            warped_pano_image=warped_pano_image,
            warped_pano_laplacian_pyramid=warped_pano_laplacian_pyramid,
            warped_main_image=warped_main_image,
            warped_main_laplacian_pyramid=warped_main_laplacian_pyramid,
            reconstructed_image_main=reconstructed_image_main,
            reconstructed_image_pano=reconstructed_image_pano,
            skip_main_clean_blending=skip_main_clean_blending,
            skip_pano_blending=skip_pano_blending,
            self_warped_main_clean_image=self_warped_main_clean_image,
            self_warped_main_clean_laplacian_pyramid=self_warped_main_clean_laplacian_pyramid,
            warped_main_clean_image=warped_main_clean_image,
            warped_main_clean_laplacian_pyramid=warped_main_clean_laplacian_pyramid,
            levels=levels,
            base_output_dir=base_output_dir,
            step=step
        )

    return reconstructed_main, reconstructed_pano

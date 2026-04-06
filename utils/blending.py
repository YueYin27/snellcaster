import os
import numpy as np
import torch
from PIL import Image
from typing import List


def detail_preserving_average(values: List[torch.Tensor], alpha: float = 0.5) -> torch.Tensor:
    """
    Detail-Preserving Averaging: normal average + value-weighted average, preserving original NaN handling semantics.

    Args:
        values: List of Laplacian pyramid levels (torch tensors, values can be outside [-1, 1]).
        alpha: Interpolation parameter between normal averaging (0) and value-weighted averaging (1).

    Returns:
        Blended Laplacian pyramid level.
    """
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError("values must be a non-empty list of tensors")

    # Ensure same device/dtype and shapes
    device = values[0].device
    dtype = values[0].dtype
    for v in values:
        if v.device != device:
            raise ValueError("All tensors in values must be on the same device")
        if v.dtype != dtype:
            raise ValueError("All tensors in values must have the same dtype")
        if v.shape != values[0].shape:
            raise ValueError("All tensors in values must have the same shape")

    # Stack tensors: [N, ...]
    V = torch.stack(values, dim=0)

    # Finite/NaN masks (preserve original semantics elementwise)
    finite_mask = torch.isfinite(V)
    nan_mask = ~finite_mask

    # Normal average using nanmean (ignores NaN values)
    avg_val = torch.nanmean(V, dim=0)

    # Value-weighted average: (sum_i |v_i| * v_i) / (sum_i |v_i|)
    abs_V = torch.abs(V)
    num = torch.where(finite_mask, abs_V * V, torch.zeros_like(V))
    denom = torch.where(finite_mask, abs_V, torch.zeros_like(V))
    num_sum = torch.sum(num, dim=0)
    denom_sum = torch.sum(denom, dim=0)
    vavg = torch.full_like(num_sum, float('nan'))
    nonzero_denom = (denom_sum != 0)
    vavg[nonzero_denom] = num_sum[nonzero_denom] / denom_sum[nonzero_denom]

    # When vavg is NaN (all finite values are 0.0), use avg_val directly
    # Otherwise use the interpolation formula
    blended = torch.where(torch.isfinite(vavg), 
                          avg_val + alpha * (vavg - avg_val),
                          avg_val)

    # Start with blended result
    result = blended

#     # Apply requested NaN/0 rules per element (generalized to N inputs):
#     # 1) If at least one is NaN and all finite values are 0 -> result 0
#     any_nan = torch.any(nan_mask, dim=0)
#     finite_abs_sum = torch.sum(torch.where(finite_mask, abs_V, torch.zeros_like(V)), dim=0)
#     all_finite_zero = (finite_abs_sum == 0)
#     result = torch.where(any_nan & all_finite_zero, torch.zeros_like(result), result)

#     # 2) If exactly one is number (finite) and all others NaN -> result is that number
#     num_finite = torch.sum(finite_mask.to(torch.int32), dim=0)
#     single_finite_mask = (num_finite == 1)
#     # Sum of finite values equals the single finite value when exactly one is finite
#     single_value = torch.sum(torch.where(finite_mask, V, torch.zeros_like(V)), dim=0)
#     result = torch.where(single_finite_mask, single_value, result)

#     # 3) If all are NaN -> result is NaN
#     any_finite = torch.any(finite_mask, dim=0)
#     all_nan = ~any_finite
#     result = torch.where(all_nan, torch.full_like(result, float('nan')), result)

    return result


def weighted_blending(base_img, ray_correspondence_path: str, color_correspondence_path: str = "results/point_color_pairs.npz", mask_path: str = None, kernel_size: int = 3, fg_mask: str = None, alpha: float = 0.5) -> Image.Image:
	"""
	Args:
		base_img: Base image as a file path (str), a PIL Image, or a NumPy array (H, W, 3).
		ray_correspondence_path: Path to NPZ file containing 'src' and 'dst' pixel pair arrays of shape (N, 2).
		color_correspondence_path: Path to NPZ file containing 'dst' (N,2) and 'rgba' (N,4) for mesh colors.
		dst_weight: Weight for the destination pixel (0.0 = only source, 1.0 = only destination, 0.5 = equal blend).
		mask_path: Optional path to mask image. White pixels in mask indicate areas to process.
		kernel_size: Odd kernel size for Gaussian neighbor fill (default 3).
		fg_mask: Optional path to foreground mask image. White pixels in fg_mask indicate foreground areas.
		alpha: Detail-preserving averaging parameter (0.0 = normal averaging, 1.0 = value-weighted averaging).

	Returns:
		PIL Image with blended/filled pixels applied using detail-preserving averaging.
	"""
	# Normalize base image to ndarray
	if isinstance(base_img, str):
		img_obj = Image.open(base_img).convert('RGB')
		base_np = np.array(img_obj)
	elif isinstance(base_img, Image.Image):
		base_np = np.array(base_img.convert('RGB'))
	elif isinstance(base_img, np.ndarray):
		base_np = base_img
	else:
		raise TypeError("base_img must be a file path, PIL Image, or numpy ndarray")

	out_np = base_np.copy()

	# Load pairs
	data = np.load(ray_correspondence_path)
	src_pixels = data.get('src')
	dst_pixels = data.get('dst')
	if src_pixels is None or dst_pixels is None:
		raise ValueError("NPZ file must contain 'src' and 'dst' arrays")

	# Load dst->rgba mesh color pairs if available
	color_map = {}
	try:
		if color_correspondence_path is not None and os.path.exists(color_correspondence_path):
			cdata = np.load(color_correspondence_path)
			dst_cols = cdata.get('dst')
			rgba_cols = cdata.get('rgba')
			if dst_cols is not None and rgba_cols is not None and len(dst_cols) == len(rgba_cols):
				for (dx, dy), col in zip(dst_cols, rgba_cols):
					color_map[(int(dx), int(dy))] = np.array(col, dtype=np.uint8)
	except Exception:
		# Silently ignore color file issues; fall back to blending only
		pass

	# Load mask if provided
	mask_np = None
	if mask_path is not None:
		mask_img = Image.open(mask_path).convert('L')  # Convert to grayscale
		mask_np = np.array(mask_img)
		# Ensure mask has same dimensions as base image
		if mask_np.shape[:2] != base_np.shape[:2]:
			raise ValueError("Mask dimensions must match base image dimensions")

	# Load fg_mask if provided
	fg_mask_np = None
	if fg_mask is not None:
		fg_mask_img = Image.open(fg_mask).convert('L')  # Convert to grayscale
		fg_mask_np = np.array(fg_mask_img)
		# Ensure fg_mask has same dimensions as base image
		if fg_mask_np.shape[:2] != base_np.shape[:2]:
			raise ValueError("fg_mask dimensions must match base image dimensions")

	# Build mapping from src -> dst for O(1) lookup
	src_to_dst = {}
	if len(src_pixels) == len(dst_pixels) and len(src_pixels) > 0:
		for (sx, sy), (dx, dy) in zip(src_pixels, dst_pixels):
			src_to_dst[(int(sx), int(sy))] = (int(dx), int(dy))

	H, W = out_np.shape[0], out_np.shape[1]

	# Vectorized path without mask: only apply blending to known pairs
	if mask_np is None:
		if src_to_dst:
			src_arr = np.array(list(src_to_dst.keys()), dtype=np.int32)
			dst_arr = np.array(list(src_to_dst.values()), dtype=np.int32)
			# Filter to bounds
			inb_src = (src_arr[:,1] >= 0) & (src_arr[:,1] < H) & (src_arr[:,0] >= 0) & (src_arr[:,0] < W)
			inb_dst = (dst_arr[:,1] >= 0) & (dst_arr[:,1] < H) & (dst_arr[:,0] >= 0) & (dst_arr[:,0] < W)
			keep = inb_src & inb_dst
			src_xy = src_arr[keep]
			dst_xy = dst_arr[keep]
			if len(src_xy) > 0:
				# Gather colors from original image
				src_colors = base_np[src_xy[:,1], src_xy[:,0]].astype(np.float32)
				dst_colors = base_np[dst_xy[:,1], dst_xy[:,0]].astype(np.float32)
				
				# Normalize to [-1, 1] range for detail-preserving averaging
				src_norm = (src_colors / 127.5) - 1.0
				dst_norm = (dst_colors / 127.5) - 1.0
				
				# Convert to torch tensors for detail-preserving averaging
				src_tensor = torch.from_numpy(src_norm)
				dst_tensor = torch.from_numpy(dst_norm)
				
				# Apply detail-preserving averaging
				blended_tensor = detail_preserving_average([src_tensor, dst_tensor], alpha)
				
				# Convert back to numpy and [0, 255] range
				blended_norm = blended_tensor.numpy()
				blended_norm = np.nan_to_num(blended_norm, nan=0.0)  # Handle NaN before casting
				blended = ((blended_norm + 1.0) * 127.5).astype(np.uint8)
				out_np[src_xy[:,1], src_xy[:,0]] = blended
		return Image.fromarray(out_np)

	# Mask path: vectorize both blending and Gaussian fill
	mask_bool = (mask_np > 0)

	# Known pair mask (within bounds)
	known_mask = np.zeros((H, W), dtype=bool)
	if src_to_dst:
		src_arr = np.array(list(src_to_dst.keys()), dtype=np.int32)
		dst_arr = np.array(list(src_to_dst.values()), dtype=np.int32)
		inb_src = (src_arr[:,1] >= 0) & (src_arr[:,1] < H) & (src_arr[:,0] >= 0) & (src_arr[:,0] < W)
		inb_dst = (dst_arr[:,1] >= 0) & (dst_arr[:,1] < H) & (dst_arr[:,0] >= 0) & (dst_arr[:,0] < W)
		keep = inb_src & inb_dst
		src_xy = src_arr[keep]
		dst_xy = dst_arr[keep]
		if len(src_xy) > 0:
			known_mask[src_xy[:,1], src_xy[:,0]] = True
			# Apply blending for known pairs with fg_mask check
			src_colors = base_np[src_xy[:,1], src_xy[:,0]].astype(np.float32)
			dst_colors = base_np[dst_xy[:,1], dst_xy[:,0]].astype(np.float32)
			
			# Normalize to [-1, 1] range for detail-preserving averaging
			src_norm = (src_colors / 127.5) - 1.0
			dst_norm = (dst_colors / 127.5) - 1.0
			
			# Convert to torch tensors for detail-preserving averaging
			src_tensor = torch.from_numpy(src_norm)
			dst_tensor = torch.from_numpy(dst_norm)
			
			# Apply detail-preserving averaging
			blended_tensor = detail_preserving_average([src_tensor, dst_tensor], alpha)
			
			# Convert back to numpy and [0, 255] range
			blended_norm = blended_tensor.numpy()
			blended_norm = np.nan_to_num(blended_norm, nan=0.0)  # Handle NaN before casting
			blended = ((blended_norm + 1.0) * 127.5).astype(np.uint8)
			
			# Check fg_mask for dst pixels - only blend if dst pixel is NOT in fg_mask
			if fg_mask_np is not None:
				# Check which dst pixels are NOT in fg_mask (fg_mask == 0)
				dst_not_in_fg = fg_mask_np[dst_xy[:,1], dst_xy[:,0]] == 0
				# Apply blending to pixels where dst is not in fg_mask
				valid_indices = dst_not_in_fg
				if np.any(valid_indices):
					out_np[src_xy[valid_indices,1], src_xy[valid_indices,0]] = blended[valid_indices]
				# For dst pixels in fg_mask, assign mesh color (from color_npz_path) to corresponding src pixel
				dst_in_fg = ~dst_not_in_fg
				if np.any(dst_in_fg) and color_map:
					# Iterate in-bounds subset
					for (sx, sy), (dx, dy) in zip(src_xy[dst_in_fg], dst_xy[dst_in_fg]):
						key = (int(dx), int(dy))
						col = color_map.get(key)
						if col is None:
							continue
						# Use RGB channels; ignore alpha for output image
						if 0 <= sy < H and 0 <= sx < W:
							out_np[int(sy), int(sx)] = np.array(col[:3], dtype=np.uint8)
			else:
				# No fg_mask provided, apply blending to all known pairs
				out_np[src_xy[:,1], src_xy[:,0]] = blended

	return Image.fromarray(out_np)


def weighted_blending_dual(main_img, pano_img, npz_path: str, alpha: float = 0.5):
	"""
	Blend main image with panorama image using pixel correspondences with detail-preserving averaging.
	
	Args:
		main_img: Main image as a file path (str), a PIL Image, or a NumPy array (H, W, 3).
		pano_img: Panorama image as a file path (str), a PIL Image, or a NumPy array (H, W, 3).
		npz_path: Path to NPZ file containing 'main' and 'pano' pixel pair arrays of shape (N, 2).
		pano_weight: Weight for the panorama pixel (0.0 = only main, 1.0 = only pano, 0.5 = equal blend).
		alpha: Detail-preserving averaging parameter (0.0 = normal averaging, 1.0 = value-weighted averaging).
	
	Returns:
		Tuple of (updated_main_img, updated_pano_img) as PIL Images with blended pixels applied using detail-preserving averaging.
	"""
	# Normalize main image to ndarray
	if isinstance(main_img, str):
		main_obj = Image.open(main_img).convert('RGB')
		main_np = np.array(main_obj)
	elif isinstance(main_img, Image.Image):
		main_np = np.array(main_img.convert('RGB'))
	elif isinstance(main_img, np.ndarray):
		main_np = main_img
	else:
		raise TypeError("main_img must be a file path, PIL Image, or numpy ndarray")
	
	# Normalize pano image to ndarray
	if isinstance(pano_img, str):
		pano_obj = Image.open(pano_img).convert('RGB')
		pano_np = np.array(pano_obj)
	elif isinstance(pano_img, Image.Image):
		pano_np = np.array(pano_img.convert('RGB'))
	elif isinstance(pano_img, np.ndarray):
		pano_np = pano_img
	else:
		raise TypeError("pano_img must be a file path, PIL Image, or numpy ndarray")
	
	# Load correspondence pairs
	data = np.load(npz_path)
	main_pixels = data.get('main')
	pano_pixels = data.get('pano')
	if main_pixels is None or pano_pixels is None:
		raise ValueError("NPZ file must contain 'main' and 'pano' arrays")
	
	# Initialize output arrays
	updated_main_np = main_np.copy().astype(np.float32)
	updated_pano_np = pano_np.copy().astype(np.float32)
	
	# Vectorized processing: filter valid correspondences
	if len(main_pixels) == len(pano_pixels) and len(main_pixels) > 0:
		main_arr = np.array(main_pixels, dtype=np.int32)
		pano_arr = np.array(pano_pixels, dtype=np.int32)
		
		# Filter to bounds for both images
		H_main, W_main = main_np.shape[0], main_np.shape[1]
		H_pano, W_pano = pano_np.shape[0], pano_np.shape[1]
		
		main_in_bounds = (main_arr[:, 1] >= 0) & (main_arr[:, 1] < H_main) & (main_arr[:, 0] >= 0) & (main_arr[:, 0] < W_main)
		pano_in_bounds = (pano_arr[:, 1] >= 0) & (pano_arr[:, 1] < H_pano) & (pano_arr[:, 0] >= 0) & (pano_arr[:, 0] < W_pano)
		valid_correspondences = main_in_bounds & pano_in_bounds
		
		if np.any(valid_correspondences):
			main_xy = main_arr[valid_correspondences]
			pano_xy = pano_arr[valid_correspondences]
			
			# Get colors from both images using advanced indexing
			main_colors = main_np[main_xy[:, 1], main_xy[:, 0]].astype(np.float32)  # Shape: (N, 3)
			pano_colors = pano_np[pano_xy[:, 1], pano_xy[:, 0]].astype(np.float32)  # Shape: (N, 3)
			
			# Normalize to [-1, 1] range for detail-preserving averaging
			main_norm = (main_colors / 127.5) - 1.0
			pano_norm = (pano_colors / 127.5) - 1.0
			
			# Convert to torch tensors for detail-preserving averaging
			main_tensor = torch.from_numpy(main_norm)
			pano_tensor = torch.from_numpy(pano_norm)
			
			# Apply detail-preserving averaging for main image
			blended_main_tensor = detail_preserving_average([main_tensor, pano_tensor], alpha)
			blended_main_norm = blended_main_tensor.numpy()
			blended_main_norm = np.nan_to_num(blended_main_norm, nan=0.0)  # Handle NaN before casting
			blended_main = ((blended_main_norm + 1.0) * 127.5).astype(np.uint8)
			updated_main_np[main_xy[:, 1], main_xy[:, 0]] = blended_main
			
			# Apply detail-preserving averaging for pano image
			blended_pano_tensor = detail_preserving_average([pano_tensor, main_tensor], alpha)
			blended_pano_norm = blended_pano_tensor.numpy()
			blended_pano_norm = np.nan_to_num(blended_pano_norm, nan=0.0)  # Handle NaN before casting
			blended_pano = ((blended_pano_norm + 1.0) * 127.5).astype(np.uint8)
			updated_pano_np[pano_xy[:, 1], pano_xy[:, 0]] = blended_pano
	
	# Convert back to PIL Images
	updated_main_img = Image.fromarray(updated_main_np.astype(np.uint8))
	updated_pano_img = Image.fromarray(updated_pano_np.astype(np.uint8))
	
	return updated_main_img, updated_pano_img


def weighted_blending_latent(latent: torch.Tensor, npz_path: str, pipeline, alpha: float = 0.5) -> torch.Tensor:
    """
    Create a detail-preserving blended visualization in latent space using the paper's averaging method.

    Args:
        latent: Latent tensor of shape (B, C, H, W) or packed format
        npz_path: Path to NPZ file containing 'src' and 'dst' pixel pair arrays of shape (N, 2)
        pipeline: The pipeline instance for VAE operations
        alpha: Detail-preserving averaging parameter (0.0 = normal averaging, 1.0 = value-weighted averaging)

    Returns:
        Modified latent tensor with detail-preserving blended pixels applied in latent space
    """
    # Load pairs - handle both 'src'/'dst' and 'main'/'pano' formats
    data = np.load(npz_path)
    src_pixels = data.get('src')
    dst_pixels = data.get('dst')
    
    # If 'src'/'dst' not found, try 'main'/'pano' format
    if src_pixels is None or dst_pixels is None:
        src_pixels = data.get('main')
        dst_pixels = data.get('pano')
    
    if src_pixels is None or dst_pixels is None:
        raise ValueError("NPZ file must contain either 'src'/'dst' arrays or 'main'/'pano' arrays")

    # Get VAE scale factor
    vae_scale_factor = getattr(pipeline, 'vae_scale_factor', 8)
    
    # Handle different tensor shapes
    if len(latent.shape) == 5:  # Packed format (B, C, H, W, 2)
        batch_size, num_channels, height, width, _ = latent.shape
        # Unpack the latent
        latent_unpacked = pipeline._unpack_latents(latent, height * vae_scale_factor, width * vae_scale_factor, vae_scale_factor)
    elif len(latent.shape) == 4:  # Standard format (B, C, H, W)
        latent_unpacked = latent
        batch_size, num_channels, height, width = latent_unpacked.shape
    elif len(latent.shape) == 3:  # Single image format (C, H, W)
        latent_unpacked = latent.unsqueeze(0)  # Add batch dimension
        batch_size, num_channels, height, width = latent_unpacked.shape
    else:
        raise ValueError(f"Unsupported latent tensor shape: {latent.shape}")
    
    # Create output tensor
    out_latent = latent_unpacked.clone()
    
    # Convert pixel coordinates to latent space coordinates
    if len(src_pixels) == len(dst_pixels) and len(src_pixels) > 0:
        for (sx, sy), (dx, dy) in zip(src_pixels, dst_pixels):
            # Convert pixel coordinates to latent coordinates
            sx_latent = int(sx // vae_scale_factor)
            sy_latent = int(sy // vae_scale_factor)
            dx_latent = int(dx // vae_scale_factor)
            dy_latent = int(dy // vae_scale_factor)
            
            # Ensure integer indices and within bounds
            if (0 <= sy_latent < height and 0 <= sx_latent < width and 
                0 <= dy_latent < height and 0 <= dx_latent < width):
                # Get original latent values
                source_value = latent_unpacked[:, :, sy_latent, sx_latent]
                dest_value = latent_unpacked[:, :, dy_latent, dx_latent]
                
                # Apply detail-preserving averaging in latent space
                blended_tensor = detail_preserving_average([source_value, dest_value], alpha)
                out_latent[:, :, sy_latent, sx_latent] = blended_tensor
    
    # Convert back to original format
    if len(latent.shape) == 5:
        out_latent = pipeline._pack_latents(out_latent, batch_size, num_channels, height, width)
    elif len(latent.shape) == 3:
        out_latent = out_latent.squeeze(0)  # Remove batch dimension
    
    return out_latent


def laplacian_pyramid_blending(pyramids: List[List[torch.Tensor]], alpha: float = 0.5) -> List[torch.Tensor]:
    """
    Blend multiple Laplacian pyramids using detail-preserving averaging at each level.
    
    Args:
        pyramids: List of Laplacian pyramids. Each pyramid is a list of tensors [C, H, W].
        alpha: Detail-preserving averaging parameter (0.0 = normal averaging, 1.0 = value-weighted averaging).
    
    Returns:
        Blended Laplacian pyramid as a list of torch tensors [C, H, W], values expected in [-1, 1].
    """
    if not isinstance(pyramids, (list, tuple)) or len(pyramids) == 0:
        return []

    num_pyramids = len(pyramids)
    num_levels = len(pyramids[0])
    # Ensure all pyramids have the same number of levels
    for idx, pyr in enumerate(pyramids):
        if len(pyr) != num_levels:
            raise ValueError(f"All pyramids must have the same number of levels. Pyramid 0 has {num_levels}, pyramid {idx} has {len(pyr)}")

    # Ensure all levels across pyramids are same shape/dtype/device per level, and move to common device
    device = pyramids[0][0].device
    normalized_pyramids: List[List[torch.Tensor]] = []
    for pyr in pyramids:
        norm_levels = []
        for lvl in pyr:
            if lvl.device != device:
                lvl = lvl.to(device)
            norm_levels.append(lvl)
        normalized_pyramids.append(norm_levels)

    blended_pyramid: List[torch.Tensor] = []
    for i in range(num_levels):
        # Collect the i-th level from each pyramid
        levels_i = [normalized_pyramids[p][i] for p in range(num_pyramids)]

        # Validate shapes/dtypes per level
        shape_i = levels_i[0].shape
        dtype_i = levels_i[0].dtype
        for j, lvl in enumerate(levels_i):
            if lvl.shape != shape_i:
                raise ValueError(f"Level {i} shapes don't match across pyramids: {lvl.shape} vs {shape_i}")
            if lvl.dtype != dtype_i:
                levels_i[j] = lvl.to(dtype=dtype_i)

        # Blend this level using multi-input detail-preserving averaging
        blended_level = detail_preserving_average(levels_i, alpha)
        blended_pyramid.append(blended_level)
    
    return blended_pyramid


def reconstruction(laplacian_pyramid: List[torch.Tensor]) -> torch.Tensor:
    """
    Reconstruct an image from a Laplacian pyramid.
    
    Starting from the coarsest level (L-1): x_{L-1} = L_{L-1}
    Recursively reconstruct: x_k = L_k + U(x_{k+1}), k = L-2, ..., 0
    where U is an upsampling operator (bilinear interpolation).
    
    Args:
        laplacian_pyramid: List of torch tensors [C, H, W] representing the Laplacian pyramid levels
        
    Returns:
        Reconstructed image as torch tensor [C, H, W]
    """
    if len(laplacian_pyramid) == 0:
        raise ValueError("Laplacian pyramid cannot be empty")
    
    # Start with the coarsest level (last in the list)
    x = laplacian_pyramid[-1].clone()  # x_{L-1} = L_{L-1}
    
    # Recursively reconstruct from coarsest to finest
    for k in range(len(laplacian_pyramid) - 2, -1, -1):  # k = L-2, ..., 0
        L_k = laplacian_pyramid[k]  # Current level L_k
        
        # Upsample the previous reconstruction x_{k+1} to match L_k's size
        # x_k has shape [C, H_k, W_k], we need to upsample to match L_k's shape
        target_height, target_width = L_k.shape[1], L_k.shape[2]
        current_height, current_width = x.shape[1], x.shape[2]
        
        if current_height != target_height or current_width != target_width:
            # Upsample using bilinear interpolation
            x_upsampled = torch.nn.functional.interpolate(
                x.unsqueeze(0),  # Add batch dimension [1, C, H, W]
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Remove batch dimension [C, H, W]
        else:
            x_upsampled = x
        
        # Add the current Laplacian level: x_k = L_k + U(x_{k+1})
        x = L_k + x_upsampled
    
    return x

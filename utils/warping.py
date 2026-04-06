import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import os
from typing import Tuple, List, Optional
import math
import trimesh
import cv2
import argparse

from utils.panorama_sampling import direction_to_uv, sample_bilinear, sample_nearest, uv_to_direction
from utils.ray_tracer import load_camera_params, create_ray_directions, load_mesh, cast_rays, cast_rays_no_refraction


def _compute_lod_level(mapping: np.ndarray, maxLOD: int = 5) -> torch.Tensor:
    """
    Compute per-pixel Level of Detail (LOD) from a UV mapping.

    The mapping is expected to be an array of shape (H, W, 2) where mapping[..., 0] is
    the source u (x in source image) and mapping[..., 1] is the source v (y in source image)
    for each destination pixel at coordinates (x, y) in the output image grid.

    Correct LOD definition used here:
        lod = log2(max( sqrt((∂u/∂x)^2 + (∂u/∂y)^2), sqrt((∂v/∂x)^2 + (∂v/∂y)^2) ))

    Invalid mapping entries should be marked with negative coordinates (e.g., -1). These
    will produce lod = 0.
    """
    if mapping is None or mapping.ndim != 3 or mapping.shape[2] != 2:
        raise ValueError("mapping must have shape (H, W, 2)")

    H, W, _ = mapping.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    u_src = torch.from_numpy(mapping[..., 0].astype(np.float32).copy()).to(device)
    v_src = torch.from_numpy(mapping[..., 1].astype(np.float32).copy()).to(device)

    valid = (u_src >= 0.0) & (v_src >= 0.0)

    def shift_left(t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        out[:, :-1] = t[:, 1:]
        out[:, -1] = t[:, -1]
        return out

    def shift_right(t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        out[:, 1:] = t[:, :-1]
        out[:, 0] = t[:, 0]
        return out

    def shift_up(t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        out[:-1, :] = t[1:, :]
        out[-1, :] = t[-1, :]
        return out

    def shift_down(t: torch.Tensor) -> torch.Tensor:
        out = t.clone()
        out[1:, :] = t[:-1, :]
        out[0, :] = t[0, :]
        return out

    u_l = shift_left(u_src);  u_r = shift_right(u_src)
    v_l = shift_left(v_src);  v_r = shift_right(v_src)
    u_u = shift_up(u_src);    u_d = shift_down(u_src)
    v_u = shift_up(v_src);    v_d = shift_down(v_src)

    val_l = shift_left(valid)
    val_r = shift_right(valid)
    val_u = shift_up(valid)
    val_d = shift_down(valid)

    use_central_h = val_l & val_r
    use_forward_h = (~use_central_h) & val_r
    use_backward_h = (~use_central_h) & (~use_forward_h) & val_l

    du_dx = torch.zeros((H, W), dtype=torch.float32, device=device)
    dv_dx = torch.zeros((H, W), dtype=torch.float32, device=device)
    du_dx[use_central_h] = 0.5 * (u_r[use_central_h] - u_l[use_central_h])
    dv_dx[use_central_h] = 0.5 * (v_r[use_central_h] - v_l[use_central_h])
    du_dx[use_forward_h] = (u_r[use_forward_h] - u_src[use_forward_h])
    dv_dx[use_forward_h] = (v_r[use_forward_h] - v_src[use_forward_h])
    du_dx[use_backward_h] = (u_src[use_backward_h] - u_l[use_backward_h])
    dv_dx[use_backward_h] = (v_src[use_backward_h] - v_l[use_backward_h])

    use_central_v = val_u & val_d
    use_forward_v = (~use_central_v) & val_d
    use_backward_v = (~use_central_v) & (~use_forward_v) & val_u

    du_dy = torch.zeros((H, W), dtype=torch.float32, device=device)
    dv_dy = torch.zeros((H, W), dtype=torch.float32, device=device)
    du_dy[use_central_v] = 0.5 * (u_d[use_central_v] - u_u[use_central_v])
    dv_dy[use_central_v] = 0.5 * (v_d[use_central_v] - v_u[use_central_v])
    du_dy[use_forward_v] = (u_d[use_forward_v] - u_src[use_forward_v])
    dv_dy[use_forward_v] = (v_d[use_forward_v] - v_src[use_forward_v])
    du_dy[use_backward_v] = (u_src[use_backward_v] - u_u[use_backward_v])
    dv_dy[use_backward_v] = (v_src[use_backward_v] - v_u[use_backward_v])

    mag_u = torch.sqrt(du_dx ** 2 + du_dy ** 2)
    mag_v = torch.sqrt(dv_dx ** 2 + dv_dy ** 2)
    rho = torch.maximum(mag_u, mag_v)

    eps = 1e-8
    lod = torch.zeros((H, W), dtype=torch.float32, device=device)
    lod[valid] = torch.log2(torch.clamp(rho[valid], min=eps))
    lod = torch.clamp(lod, 0.0, float(maxLOD))
    return lod


def safe_minmax(img):
    # Replace NaNs with +inf / -inf so they don’t affect min/max
    finite_mask = torch.isfinite(img)
    if not torch.any(finite_mask):
        # if all values are NaN or Inf, return 0s
        return torch.tensor(0.0, device=img.device), torch.tensor(0.0, device=img.device)
    finite_vals = img[finite_mask]
    return finite_vals.min(), finite_vals.max()


def build_gaussian_pyramid(image: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """
    Build a Gaussian pyramid G(x) = {G_0(x), G_1(x), ..., G_{L-1}(x)}.

    - G_0(x) is the original image x
    - For l > 0: G_l(x) is obtained by Gaussian blurring G_{l-1}(x) and downsampling by 2

    Args:
        image: Torch image tensor. Accepted shapes: [C, H, W] or [1, C, H, W].
                Values will be normalized to [-1, 1] range before pyramid construction.
        levels: Number of levels L in the pyramid (L >= 1).

    Returns:
        List of torch float32 tensors [C, H, W] in [-1, 1] for each level.
    """
    if levels <= 0:
        raise ValueError("levels must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    img = image.to(device).float()
    if img.ndim == 3:  # [C, H, W]
        img = img.unsqueeze(0)
    elif img.ndim == 4:  # [1, C, H, W] or [N, C, H, W]
        if img.shape[0] != 1:
            raise ValueError("image batch dimension must be 1")
    else:
        raise ValueError("image tensor must be 3D or 4D")

    # Don't normalize the input to preserve reconstruction property
    # The image should be used as-is to ensure that reconstruction returns the original
    
    # NaN-aware processing: keep original (with NaNs) for level 0, and
    # for blurred/downsampled levels, compute using numerator/denominator scheme
    valid_mask = torch.isfinite(img).float()  # [1, C, H, W]
    img_filled = torch.nan_to_num(img, nan=0.0)

    def gaussian_kernel1d(sigma: float, kernel_size: int) -> torch.Tensor:
        half = (kernel_size - 1) / 2.0
        x = torch.linspace(-half, half, steps=kernel_size, device=device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        return kernel

    def gaussian_blur(img: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        # Create separable Gaussian kernel
        kernel_size = int(max(3, 2 * round(3 * sigma) + 1))
        k1d = gaussian_kernel1d(sigma, kernel_size)
        k2d = torch.outer(k1d, k1d)
        C = img.shape[1]
        weight = k2d.expand(C, 1, -1, -1).contiguous()
        padding = kernel_size // 2
        return F.conv2d(img, weight, bias=None, stride=1, padding=padding, groups=C)

    pyramid: List[torch.Tensor] = []
    # Store level 0 directly (retain NaNs)
    pyramid.append(img.squeeze(0))

    # Prepare current numerator and denominator for nan-aware blur/downsample
    current_num = img_filled
    current_den = valid_mask

    for _ in range(1, levels):
        num_blur = gaussian_blur(current_num, sigma=0.5)
        den_blur = gaussian_blur(current_den, sigma=0.5)
        
        H, W = num_blur.shape[2], num_blur.shape[3]
        new_h = max(1, H // 2)
        new_w = max(1, W // 2)

        # Downsample numerator and denominator
        num_down = F.interpolate(num_blur, size=(new_h, new_w), mode='bilinear', align_corners=False)
        den_down = F.interpolate(den_blur, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Avoid division by zero: where den==0, keep NaN
        with torch.no_grad():
            den_safe = den_down.clone()
            den_safe[den_safe == 0] = float('nan')
        level = num_down / den_safe

        pyramid.append(level.squeeze(0))

        # Update current for next iteration
        current_num = num_down
        current_den = den_down

        if new_w == 1 and new_h == 1:
            break

    return pyramid


def build_laplacian_pyramid(image: torch.Tensor, levels: int) -> List[torch.Tensor]:
    """
    Build a Laplacian pyramid L(x) from image x using the Gaussian pyramid G(x).

    For k in [0, L-2]:
        L_k(x) = G_k(x) - U(G_{k+1}(x))
    For k = L-1:
        L_{L-1}(x) = G_{L-1}(x)

    Where U is an upsampling operator (bilinear interpolation to the size of G_k).

    Args:
        image: Torch image tensor. Accepted shapes: [C, H, W] or [1, C, H, W], values in [-1, 1].
        levels: Number of levels L in the pyramid (L >= 1).

    Returns:
        List of torch float32 tensors [C, H, W] containing Laplacian levels. 
        All levels are in [-1, 1] range.
    """
    if levels <= 0:
        raise ValueError("levels must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Gaussian pyramid first (returns list of [C, H, W] torch tensors)
    g_pyr = build_gaussian_pyramid(image, levels)

    # Compute Laplacian levels
    l_pyr: List[torch.Tensor] = []
    for k in range(levels - 1):
        gk = g_pyr[k].unsqueeze(0)      # [1, C, H, W]
        gk1 = g_pyr[k + 1].unsqueeze(0) # [1, C, h, w]
        up = F.interpolate(gk1, size=(gk.shape[2], gk.shape[3]), mode='bilinear', align_corners=False)
        lap = gk - up
        # Preserve NaNs: set to NaN wherever inputs are NaN
        invalid = ~torch.isfinite(gk) | ~torch.isfinite(up)
        lap[invalid] = float('nan')
        l_pyr.append(lap.squeeze(0))      # [C, H, W]

    # Last level: append gaussian pyramid last level
    l_pyr.append(g_pyr[-1])

    return l_pyr


def load_uv_map(path: str) -> Optional[torch.Tensor]:
    if not os.path.exists(path):
        return None
    data = np.load(path)
    uv = data["uv"].astype(np.float32).copy()
    uv = torch.from_numpy(uv)
    return uv


def generate_main_to_pano_uv_map(
    camera_params: str = "results/moge/other/main/living_room_30_main/camera.json",
    bg_mesh: str = "results/bg_mesh.glb",
    fg_mesh: str = "results/fg_mesh.glb",
    main_w: int = 720,
    main_h: int = 1280,
    pano_w: int = 2048,
    pano_h: int = 1024,
    debug_rays: int = None,
    mode: str = "with_fg",
    output_dir: str = "results/warpings",
) -> str:
    """
    Generate UV map for main camera to panorama warping using 3D mesh ray tracing.
    
    Args:
        camera_params: Path to camera parameters JSON file
        bg_mesh: Path to background 3D mesh file (.glb)
        fg_mesh: Path to foreground 3D mesh file (.glb) - used for panorama camera position
        main_w: Main camera image width in pixels
        main_h: Main camera image height in pixels
        pano_w: Panorama width in pixels
        pano_h: Panorama height in pixels
        debug_rays: If set, only process this many rays for debugging (default: None for all rays)
        mode: Mode for occlusion checking. "with_fg" (default) checks both fg_mesh and bg_mesh,
              "without_fg" only checks bg_mesh during re-casting validation
        output_dir: Directory to save all output files (UV maps, masks, etc.)
    
    Returns:
        Path to the saved UV map file
    """
    # Validate mode parameter
    if mode not in ["with_fg", "without_fg"]:
        raise ValueError(f"mode must be 'with_fg' or 'without_fg', got '{mode}'")
    
    # Load camera parameters
    cam_params = load_camera_params(camera_params)
    fov_x = cam_params['fov_x']
    fov_y = cam_params['fov_y']
    
    # Load background 3D mesh
    print(f"Loading background mesh from: {bg_mesh}")
    bg_mesh_obj = load_mesh(bg_mesh)
    
    # Load foreground 3D mesh and use its center as panorama camera location
    print(f"Loading foreground mesh from: {fg_mesh}")
    fg_mesh_obj = load_mesh(fg_mesh)
    
    # Use foreground mesh center as panorama camera location
    pano_cam_loc = fg_mesh_obj.centroid
    
    main_width, main_height = main_w, main_h
    print(f"Main image size: {main_width}x{main_height}")
    print(f"Panorama size: {pano_w}x{pano_h}")
    print(f"FOV: {fov_x}° x {fov_y}°")
    
    # Create ray directions for main camera (used for projecting 3D points to pixel coordinates)
    ray_directions, _ = create_ray_directions(main_width, main_height, fov_x, fov_y)
    # Flatten ray_directions for efficient lookup: [H*W, 3]
    ray_dirs_flat = ray_directions.reshape(-1, 3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Main camera position (at origin)
    main_cam_pos = np.array([0.0, 0.0, 0.0])
    # Use torch.tensor (copy) to avoid non-writable NumPy view warnings
    pano_cam_pos = torch.tensor(np.array(pano_cam_loc, dtype=np.float32, copy=True), device=device)
    
    # Set UV map save path
    os.makedirs(output_dir, exist_ok=True)
    if mode == "with_fg":
        uv_cache_path = os.path.join(output_dir, "main_to_pano_uv_with_fg.npz")
    else:
        uv_cache_path = os.path.join(output_dir, "main_to_pano_uv_without_fg.npz")

    # Build mask and UV in one pass by iterating all pano pixels and casting once
    print("Building mask and UV in one pass (pano->BG intersections, project to main)...")

    # Initialize UV map for pano (u,v -> main x,y). Default invalid to -1
    uv_map = torch.full((pano_h, pano_w, 2), -1.0, dtype=torch.float32, device=device)

    # Iterate over all pixels on the panorama
    total = pano_w * pano_h
    if debug_rays is not None:
        total = min(total, int(debug_rays))
        print(f"Debug mode: processing only {total} rays")

    pano_us = np.arange(total, dtype=np.int32) % pano_w
    pano_vs = np.arange(total, dtype=np.int32) // pano_w

    # Build directions for all (or debug-limited) pano pixels
    # Vectorized build of pano ray directions via equirectangular mapping
    dirs = torch.empty((total, 3), dtype=torch.float32, device=device)
    u_t = torch.from_numpy(pano_us[:total]).to(device=device, dtype=torch.float32)
    v_t = torch.from_numpy(pano_vs[:total]).to(device=device, dtype=torch.float32)
    phi = 2.0 * math.pi * (u_t / float(pano_w)) - math.pi
    lam = (math.pi / 2.0) - math.pi * (v_t / float(pano_h))
    cos_lam = torch.cos(lam)
    dirs[:, 0] = cos_lam * torch.sin(phi)
    dirs[:, 1] = cos_lam * torch.cos(phi)
    dirs[:, 2] = torch.sin(lam)
    origins = pano_cam_pos.reshape(1, 3).repeat(total, 1).cpu().numpy()

    # Cast rays from panorama camera to intersect with bg_mesh using cast_rays_no_refraction()
    print(f"Casting {total} rays from panorama camera to bg_mesh...")
    locs, idx_ray, _ = cast_rays_no_refraction(bg_mesh_obj, origins, dirs.cpu().numpy())

    # Batch validation by re-casting from main camera to the same 3D points
    num_hits = len(locs)
    if num_hits > 0:
        intersections_np = np.asarray(locs)
        rays_idx = np.asarray(idx_ray)
        pixel_us = pano_us[rays_idx]
        pixel_vs = pano_vs[rays_idx]

        # Connect the intersection with the main camera and do re-casting from the main camera center for validation check
        # Build directions from main cam (origin) to intersections
        dirs_main = intersections_np  # main cam at origin
        norms = np.linalg.norm(dirs_main, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0
        dirs_main = dirs_main.copy()
        dirs_main[nonzero] = dirs_main[nonzero] / norms[nonzero]
        origins_main = np.zeros((num_hits, 3), dtype=np.float32)
        
        # Expected distances from main camera to intersection points
        expected_dists = norms[nonzero, 0].flatten()

        # Only cast rays for valid (nonzero) directions to save computation
        nonzero_indices = np.where(nonzero)[0]
        num_valid_rays = len(nonzero_indices)
        
        print(f"Re-casting {num_valid_rays} rays from main camera to check for occlusions (out of {num_hits} total)...")
        
        if num_valid_rays > 0:
            # Extract only valid rays for casting
            origins_main_valid = origins_main[nonzero_indices]
            dirs_main_valid = dirs_main[nonzero_indices]
            
            # Cast rays separately for fg_mesh and bg_mesh to check if the ray hits either fg_mesh or bg_mesh
            if mode == "with_fg":
                fg_locs, fg_idx_ray_valid, _ = cast_rays_no_refraction(fg_mesh_obj, origins_main_valid, dirs_main_valid)
                # Map back to original indices
                if len(fg_idx_ray_valid) > 0:
                    fg_idx_ray = nonzero_indices[fg_idx_ray_valid]
                else:
                    fg_idx_ray = np.array([], dtype=np.int32)
            else:
                # For "without_fg" mode, skip fg_mesh casting
                fg_locs = []
                fg_idx_ray = np.array([], dtype=np.int32)
            
            bg_locs, bg_idx_ray_valid, _ = cast_rays_no_refraction(bg_mesh_obj, origins_main_valid, dirs_main_valid)
            # Map back to original indices
            if len(bg_idx_ray_valid) > 0:
                bg_idx_ray = nonzero_indices[bg_idx_ray_valid]
            else:
                bg_idx_ray = np.array([], dtype=np.int32)
        else:
            # No valid rays to cast
            fg_locs = []
            fg_idx_ray = np.array([], dtype=np.int32)
            bg_locs = []
            bg_idx_ray = np.array([], dtype=np.int32)

        print("Finished Re-casting rays from main camera to check for occlusions...")

        valid = np.zeros(num_hits, dtype=bool)
        eps = 1e-4
        
        # Batch process: validate all intersections by computing distances vectorized
        # If the ray hits either fg_mesh or bg_mesh before the expected point, mask out the ray
        # Only validate if we have valid rays and re-casting results
        if num_valid_rays > 0:
            print(f"Validating {num_valid_rays} rays with vectorized distance computation...")
            
            # Collect all intersection locations and their corresponding ray indices
            all_intersection_locs = []
            all_intersection_ray_indices = []
            
            # Add fg_mesh intersections (only if mode is "with_fg")
            if mode == "with_fg" and len(fg_locs) > 0:
                all_intersection_locs.append(fg_locs)
                all_intersection_ray_indices.append(fg_idx_ray)
            
            # Add bg_mesh intersections
            if len(bg_locs) > 0:
                all_intersection_locs.append(bg_locs)
                all_intersection_ray_indices.append(bg_idx_ray)
            
            if len(all_intersection_locs) > 0:
                # Stack all intersections: [total_intersections, 3]
                all_locs = np.vstack(all_intersection_locs)
                all_ray_idx = np.concatenate(all_intersection_ray_indices)
                
                # Compute distances from origins_main to all intersections in one go
                # all_locs: [total_intersections, 3]
                # origins_main[all_ray_idx]: [total_intersections, 3]
                diffs = all_locs - origins_main[all_ray_idx]
                all_dists = np.linalg.norm(diffs, axis=1)  # [total_intersections]
                
                # Vectorized: find minimum distance per ray using sorting and reduceat
                # Sort by ray index to group intersections by ray
                sort_idx = np.argsort(all_ray_idx)
                sorted_ray_idx = all_ray_idx[sort_idx]
                sorted_dists = all_dists[sort_idx]
                
                # Find boundaries between groups (where ray index changes)
                boundaries = np.concatenate([[0], np.where(np.diff(sorted_ray_idx))[0] + 1, [len(sorted_ray_idx)]])
                
                # Compute minimum distance per group using reduceat
                min_dists_sorted = np.minimum.reduceat(sorted_dists, boundaries[:-1])
                unique_ray_indices = sorted_ray_idx[boundaries[:-1]]
                
                # Vectorized: map ray indices to minimum distances using advanced indexing
                # Create an array that maps ray index to minimum distance
                # Use max(unique_ray_indices) + 1 as size, or use a sparse approach
                max_ray_idx = max(np.max(unique_ray_indices) if len(unique_ray_indices) > 0 else 0, 
                                np.max(nonzero_indices) if len(nonzero_indices) > 0 else 0) + 1
                ray_to_min_dist = np.full(max_ray_idx, np.inf, dtype=np.float32)
                ray_to_min_dist[unique_ray_indices] = min_dists_sorted
                
                # For each ray in nonzero_indices, get its minimum distance (or inf if no intersection)
                min_dists_per_ray = ray_to_min_dist[nonzero_indices]
                
                # Compare with expected distances
                valid_mask = np.abs(min_dists_per_ray - expected_dists) <= eps
                valid[nonzero_indices[valid_mask]] = True

        # Process only validated hits: project each to main using manual projection (much faster than dot products)
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            print(f"Processing {len(valid_indices)} validated intersections...")
            
            # Vectorized manual projection: project all 3D points to pixel coordinates at once
            intersections_valid = intersections_np[valid_indices]  # [num_valid, 3]
            
            # Manual projection using camera intrinsics (FOV)
            # Camera is looking in (0,1,0) direction, so Y is forward
            # Project to pixel coordinates: x_cam = X/Y, z_cam = Z/Y
            y_coords = intersections_valid[:, 1]  # Y coordinates (forward direction)
            nonzero_y = np.abs(y_coords) > 1e-12
            
            # Initialize pixel coordinates
            x_pix_all = np.full(len(valid_indices), -1.0, dtype=np.float32)
            y_pix_all = np.full(len(valid_indices), -1.0, dtype=np.float32)
            
            if nonzero_y.any():
                # Compute camera space coordinates
                x_cam = intersections_valid[nonzero_y, 0] / y_coords[nonzero_y]
                z_cam = intersections_valid[nonzero_y, 2] / y_coords[nonzero_y]
                
                # Convert to NDC (normalized device coordinates)
                x_ndc = x_cam / math.tan(math.radians(fov_x) / 2.0)
                y_ndc = -z_cam / math.tan(math.radians(fov_y) / 2.0)
                
                # Convert to pixel coordinates
                x_pix_all[nonzero_y] = (x_ndc + 1.0) * 0.5 * float(main_width)
                y_pix_all[nonzero_y] = (y_ndc + 1.0) * 0.5 * float(main_height)
            
            # Filter valid pixel coordinates and fill UV map
            valid_pixels = (nonzero_y & 
                          (x_pix_all >= 0.0) & (x_pix_all < (main_width - 1)) &
                          (y_pix_all >= 0.0) & (y_pix_all < (main_height - 1)))
            
            valid_pixel_indices = np.where(valid_pixels)[0]
            if len(valid_pixel_indices) > 0:
                # Get corresponding panorama pixel coordinates
                uu_all = pixel_us[valid_indices[valid_pixel_indices]]
                vv_all = pixel_vs[valid_indices[valid_pixel_indices]]
                x_pix_valid = x_pix_all[valid_pixel_indices]
                y_pix_valid = y_pix_all[valid_pixel_indices]
                
                # Convert numpy arrays to torch tensors on the same device as uv_map
                x_pix_valid_t = torch.from_numpy(x_pix_valid).to(device=device, dtype=torch.float32)
                y_pix_valid_t = torch.from_numpy(y_pix_valid).to(device=device, dtype=torch.float32)
                
                # Fill UV map
                uv_map[vv_all, uu_all, 0] = x_pix_valid_t
                uv_map[vv_all, uu_all, 1] = y_pix_valid_t

    # Save UV map
    uv_np = uv_map.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(uv_cache_path, uv=uv_np, shape=np.array(uv_map.shape, dtype=np.int32))
    print(f"Saved UV map for main->pano to {uv_cache_path}")

    return uv_np


def generate_pano_to_main_uv_map(
    camera_params: str = "results/moge/other/main/living_room_30_main/camera.json",
    bg_mesh: str = "results/bg_mesh.glb",
    fg_mesh: str = "results/fg_mesh.glb",
    main_w: int = 720,
    main_h: int = 1280,
    pano_w: int = 2048,
    pano_h: int = 1024,
    output_dir: str = "results/warpings",
    ior: float = 1.5,
) -> str:
    """
    Generate UV map for panorama to main camera warping using 3D mesh ray tracing.
    
    Pipeline:
      1) Cast rays from the main camera to intersect the background mesh.
      2) For each intersection, compute the ray direction from the panorama camera (fg mesh centroid) to the same 3D point.
      3) Convert this direction to panorama UV and store in the UV map.
    
    Args:
        camera_params: Path to camera parameters JSON file
        bg_mesh: Path to background 3D mesh file (.glb)
        fg_mesh: Path to foreground 3D mesh file (.glb) - used for panorama camera position
        main_w: Main camera image width in pixels
        main_h: Main camera image height in pixels
        pano_w: Panorama width in pixels
        pano_h: Panorama height in pixels
        output_dir: Directory to save all output files (UV maps, masks, etc.)
        ior: Index of refraction for the foreground mesh (default: 1.5)
    
    Returns:
        Path to the saved UV map file
    """
    # Load camera parameters
    cam_params = load_camera_params(camera_params)
    fov_x = cam_params['fov_x']
    fov_y = cam_params['fov_y']

    # Load meshes
    bg_mesh_obj = load_mesh(bg_mesh)
    fg_mesh_obj = load_mesh(fg_mesh)
    pano_cam_loc = fg_mesh_obj.centroid

    # Create bounding box mesh and combine with bg_mesh
    # This ensures all rays hit something, avoiding the need for fallback logic
    bbox_min, bbox_max = bg_mesh_obj.bounds
    bbox_mesh = trimesh.creation.box(
        extents=bbox_max - bbox_min,
        transform=trimesh.transformations.translation_matrix((bbox_min + bbox_max) / 2)
    )
    # Combine bg_mesh with bbox mesh (invert normals of bbox so we hit from inside)
    bbox_mesh.invert()
    bg_mesh_with_bbox = trimesh.util.concatenate([bg_mesh_obj, bbox_mesh])
    print(f"Combined bg_mesh ({len(bg_mesh_obj.faces)} faces) with bbox ({len(bbox_mesh.faces)} faces) -> {len(bg_mesh_with_bbox.faces)} faces")

    main_width, main_height = main_w, main_h

    # Set UV map save path
    os.makedirs(output_dir, exist_ok=True)
    uv_cache_path = os.path.join(output_dir, "pano_to_main_uv.npz")
    cached_uv = load_uv_map(uv_cache_path)

    # Create main camera ray directions
    ray_directions, _ = create_ray_directions(main_width, main_height, fov_x, fov_y)

    # Cast rays from main camera
    print("Casting rays from main camera...")
    ray_origins = np.zeros((main_height, main_width, 3))
    
    # First pass: Cast rays through fg_mesh -> bg_mesh (without bbox)
    # This ensures direct rays only hit real geometry
    locations, index_ray, index_tri, rays_hit_fg, reflection_dirs, fresnel_Rs = cast_rays(
        fg_mesh_obj, bg_mesh_obj, ray_origins, ray_directions, 
        ior=ior, return_fg_rays=True, return_reflections=True
    )
    print(f"Rays that hit fg_mesh: {len(rays_hit_fg)}")
    print(f"Reflection data computed for {len(reflection_dirs)} rays")
    
    # Find refracted rays that missed bg_mesh
    total_rays = main_height * main_width
    all_indices = set(range(total_rays))
    hit_indices = set(index_ray.tolist()) if len(index_ray) > 0 else set()
    miss_indices = all_indices - hit_indices
    refracted_miss_indices = miss_indices & rays_hit_fg
    
    print(f"Refracted rays that missed bg_mesh: {len(refracted_miss_indices)}")
    
    # Second pass: For refracted rays that missed bg_mesh, cast against bbox only
    if len(refracted_miss_indices) > 0:
        refracted_miss_list = sorted(list(refracted_miss_indices))
        refracted_miss_array = np.array(refracted_miss_list, dtype=np.int64)
        
        # Get ray origins and directions for missed rays
        miss_ys = refracted_miss_array // main_width
        miss_xs = refracted_miss_array % main_width
        miss_origins = ray_origins[miss_ys, miss_xs, :]
        miss_directions = ray_directions[miss_ys, miss_xs, :]
        
        # Cast against bbox only
        bbox_locs, bbox_idx_ray, bbox_idx_tri = cast_rays(
            fg_mesh_obj, bbox_mesh, miss_origins, miss_directions, ior=ior
        )
        
        print(f"Bbox intersections for refracted misses: {len(bbox_locs)}")
        
        # Add bbox hits to the main results
        if len(bbox_locs) > 0:
            # Map back to global indices
            global_bbox_indices = refracted_miss_array[bbox_idx_ray]
            
            # Append to main results
            if len(locations) > 0:
                locations = np.vstack([locations, bbox_locs])
                index_ray = np.concatenate([index_ray, global_bbox_indices])
                index_tri = np.concatenate([index_tri, bbox_idx_tri])
            else:
                locations = bbox_locs
                index_ray = global_bbox_indices
                index_tri = bbox_idx_tri

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uv_map = torch.full((main_height, main_width, 2), -1.0, dtype=torch.float32, device=device)

    pano_cam_pos = np.array(pano_cam_loc)
    eps = 1e-4
    # Batch process: validate all intersections by casting from pano cam simultaneously
    num_hits = len(locations)
    intersections_np = np.asarray(locations) if num_hits > 0 else np.zeros((0, 3), dtype=np.float32)
    rays_idx = np.asarray(index_ray) if num_hits > 0 else np.zeros((0,), dtype=int)

    if num_hits > 0:
        pixel_xs = (rays_idx % main_width).astype(int)
        pixel_ys = (rays_idx // main_width).astype(int)

        print(f"Processing {num_hits} rays with re-casting validation...")
        dirs = intersections_np - pano_cam_pos.reshape(1, 3)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 1e-6
        dirs_normalized = np.zeros_like(dirs)
        if nonzero.any():
            dirs_normalized[nonzero] = dirs[nonzero] / norms[nonzero]

        origins = np.repeat(pano_cam_pos.reshape(1, 3), num_hits, axis=0)
        pano_locs, pano_idx_ray, _ = cast_rays_no_refraction(bg_mesh_with_bbox, origins, dirs_normalized)

        valid = np.zeros(num_hits, dtype=bool)
        if len(pano_locs) > 0 and len(pano_idx_ray) > 0:
            diffs = pano_locs - intersections_np[pano_idx_ray]
            dists = np.linalg.norm(diffs, axis=1)
            ok = dists <= eps
            if ok.any():
                valid[pano_idx_ray[ok]] = True

        for idx_hit in range(num_hits):
            if not nonzero[idx_hit]:
                continue
            if not valid[idx_hit]:
                continue
            u, v = direction_to_uv(dirs_normalized[idx_hit], pano_w, pano_h)
            if 0 <= u < pano_w and 0 <= v < pano_h:
                y = int(pixel_ys[idx_hit])
                x = int(pixel_xs[idx_hit])
                uv_map[y, x, 0] = float(u)
                uv_map[y, x, 1] = float(v)

    # Save UV map
    uv_np = uv_map.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(uv_cache_path, uv=uv_np, shape=np.array(uv_map.shape, dtype=np.int32))
    print(f"Saved UV map for pano->main to {uv_cache_path}")

    # Create reflection UV map
    print("\nGenerating reflection UV map...")
    uv_map_reflection = torch.full((main_height, main_width, 2), -1.0, dtype=torch.float32, device=device)
    
    # Process reflection directions to create UV map
    # reflection_dirs maps ray_idx -> reflection direction (3D vector)
    num_reflections = 0
    for ray_idx, reflection_dir in reflection_dirs.items():
        # Get pixel coordinates from ray index
        y = int(ray_idx // main_width)
        x = int(ray_idx % main_width)
        
        # Skip if out of bounds
        if y >= main_height or x >= main_width:
            continue
        
        # Normalize reflection direction
        reflection_dir_norm = reflection_dir / (np.linalg.norm(reflection_dir) + 1e-8)
        
        # Convert reflection direction to panorama UV coordinates
        # The reflection direction is from the main camera position, looking in the reflected direction
        u, v = direction_to_uv(reflection_dir_norm, pano_w, pano_h)
        
        # Check if UV is valid
        if 0 <= u < pano_w and 0 <= v < pano_h:
            uv_map_reflection[y, x, 0] = float(u)
            uv_map_reflection[y, x, 1] = float(v)
            num_reflections += 1
    
    print(f"Generated {num_reflections} reflection UV mappings")
    
    # Save reflection UV map
    reflection_uv_path = uv_cache_path.replace('.npz', '_reflection.npz')
    np.savez_compressed(
        reflection_uv_path, 
        uv=uv_map_reflection.detach().cpu().numpy().astype(np.float32), 
        shape=np.array(uv_map_reflection.shape, dtype=np.int32)
    )
    print(f"Saved reflection UV map to {reflection_uv_path}")
    
    # Save Fresnel coefficients as grayscale image (0-1 range)
    fresnel_map = np.zeros((main_height, main_width), dtype=np.float32)
    for ray_idx, R in fresnel_Rs.items():
        y = int(ray_idx // main_width)
        x = int(ray_idx % main_width)
        if y < main_height and x < main_width:
            fresnel_map[y, x] = R
    
    # Convert to uint8 grayscale image (0-255 range)
    fresnel_image = (np.clip(fresnel_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    fresnel_path = os.path.join(output_dir, "fresnel_reflection_ratio.png")
    Image.fromarray(fresnel_image, mode='L').save(fresnel_path)
    print(f"Saved Fresnel reflection ratio image to {fresnel_path}")

    return uv_np


def generate_self_uv_map(
    camera_params: str = "results/moge/main/living_room_30_main/camera.json",
    bg_mesh: str = "results/bg_mesh.glb",
    fg_mesh: str = "results/fg_mesh.glb",
    main_w: int = 1280,
    main_h: int = 720,
    fg_mask: str = "results/sphere_mask.png",
    ior: float = 1.5,
    output_dir: str = "results/warpings",
) -> str:
    """
    Generate a self UV map for the main camera by refracting rays through the foreground mesh.

    For each pixel where fg_mask == 1:
      - Cast a ray from the camera center through that pixel direction.
      - Refract through fg_mesh (treated as transparent glass with IoR=1.5) and intersect bg_mesh.
      - If intersection exists, connect intersection to camera center and project this direction
        back to a pixel coordinate; store that (x, y) in the UV map at the original pixel.
      - If no hit, store invalid (-1, -1).

    Args:
        camera_params: Path to camera parameters JSON file
        bg_mesh: Path to background 3D mesh file (.glb)
        fg_mesh: Path to foreground 3D mesh file (.glb)
        main_w: Main camera image width in pixels
        main_h: Main camera image height in pixels
        fg_mask: Path to foreground mask image (will be resized to match main image dimensions)
        ior: Index of refraction for the foreground mesh
        output_dir: Directory to save the UV map

    Returns:
        Path to the saved UV map file (npz with key 'uv').
    """
    # Load camera parameters
    cam_params = load_camera_params(camera_params)
    fov_x = cam_params['fov_x']
    fov_y = cam_params['fov_y']

    # Load meshes
    bg_mesh_obj = load_mesh(bg_mesh)
    fg_mesh_obj = load_mesh(fg_mesh)

    main_width, main_height = main_w, main_h

    # Load and normalize fg_mask to match main image size
    mask_arr = None
    if fg_mask and os.path.exists(fg_mask):
        mask_img = Image.open(fg_mask).convert('L')
        if mask_img.size != (main_width, main_height):
            from PIL import Image as PILImage
            mask_img = mask_img.resize((main_width, main_height), PILImage.NEAREST)
        mask_arr = np.array(mask_img)
        if mask_arr.max() > 1:
            mask_arr = (mask_arr / 255.0).astype(np.float32)
    else:
        # If no mask is provided, default to process no pixels (empty mask)
        mask_arr = np.zeros((main_height, main_width), dtype=np.float32)

    # Create ray directions for main camera
    ray_directions, _ = create_ray_directions(main_width, main_height, fov_x, fov_y)

    # Collect masked pixel indices (row-major)
    mask_bool = mask_arr >= 0.5
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        print("fg_mask contains no active pixels (>=0.5). Nothing to do.")
        os.makedirs(output_dir, exist_ok=True)
        uv_cache_path = os.path.join(output_dir, "self_uv_map.npz")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        uv_map = torch.full((main_height, main_width, 2), -1.0, dtype=torch.float32, device=device)
        uv_np = uv_map.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(uv_cache_path, uv=uv_np, shape=np.array(uv_map.shape, dtype=np.int32))
        return uv_np

    # Build subset of ray directions for masked pixels
    dirs_subset = ray_directions[ys, xs, :]  # [N, 3]
    origins_subset = np.zeros((dirs_subset.shape[0], 3), dtype=np.float32)

    # Cast refractive rays through fg -> bg
    print(f"Casting refractive rays for {dirs_subset.shape[0]} masked pixels...")
    locations, index_ray, index_tri = cast_rays(
        fg_mesh_obj, bg_mesh_obj, origins_subset, dirs_subset, ior=ior
    )

    # Prepare UV map initialized to invalid (-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uv_map = torch.full((main_height, main_width, 2), -1.0, dtype=torch.float32, device=device)

    if len(locations) == 0:
        print("No intersections found for self UV map.")
    else:
        # For each hit, compute direction from camera center to intersection and project back
        intersections_np = np.asarray(locations)
        rays_idx = np.asarray(index_ray)
        hit_xs = xs[rays_idx]
        hit_ys = ys[rays_idx]

        # Vectorized projection
        if len(intersections_np) > 0:
            # Filter out points where y-coordinate is near zero
            valid_mask = np.abs(intersections_np[:, 1]) > 1e-12
            
            if np.any(valid_mask):
                # Get valid intersections and corresponding indices
                valid_intersections = intersections_np[valid_mask]
                valid_hit_xs = hit_xs[valid_mask]
                valid_hit_ys = hit_ys[valid_mask]
                
                # Vectorized camera projection
                x_cam = valid_intersections[:, 0] / valid_intersections[:, 1]
                z_cam = valid_intersections[:, 2] / valid_intersections[:, 1]
                
                # Vectorized NDC conversion
                tan_fov_x_half = np.tan(np.radians(fov_x) / 2.0)
                tan_fov_y_half = np.tan(np.radians(fov_y) / 2.0)
                x_ndc = x_cam / tan_fov_x_half
                y_ndc = -z_cam / tan_fov_y_half
                
                # Vectorized pixel coordinates
                x_pix = (x_ndc + 1.0) * 0.5 * main_width
                y_pix = (y_ndc + 1.0) * 0.5 * main_height
                
                # Vectorized bounds checking
                in_bounds = (x_pix >= 0.0) & (x_pix < (main_width - 1)) & \
                            (y_pix >= 0.0) & (y_pix < (main_height - 1))
                
                if np.any(in_bounds):
                    # Get final valid coordinates
                    final_x_pix = x_pix[in_bounds]
                    final_y_pix = y_pix[in_bounds]
                    final_hit_xs = valid_hit_xs[in_bounds].astype(np.int64)
                    final_hit_ys = valid_hit_ys[in_bounds].astype(np.int64)
                    
                    # Batch update to uv_map using advanced indexing
                    uv_map[final_hit_ys, final_hit_xs, 0] = torch.from_numpy(final_x_pix.astype(np.float32)).to(uv_map.device)
                    uv_map[final_hit_ys, final_hit_xs, 1] = torch.from_numpy(final_y_pix.astype(np.float32)).to(uv_map.device)

    # Save UV map
    os.makedirs(output_dir, exist_ok=True)
    uv_cache_path = os.path.join(output_dir, "self_uv_map.npz")
    uv_np = uv_map.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(uv_cache_path, uv=uv_np, shape=np.array(uv_map.shape, dtype=np.int32))
    print(f"Saved self UV map to {uv_cache_path}")

    return uv_np


def generate_uv_map(
    camera_params: str = "results/moge/other/main/living_room_30_main/camera.json",
    image: str = "results/moge/other/main/living_room_30_main/image.jpg",
    bg_mesh: str = "results/bg_mesh.glb",
    fg_mesh: str = "results/fg_mesh.glb",
    fg_mask: str = "results/sphere_mask.png",
    output_dir: str = "results/warpings",
    pano_w: int = 2048,
    pano_h: int = 1024,
    ior: float = 1.5,
):
    """
    Generate all UV maps by combining self, pano-to-main, and main-to-pano UV map generation.
    
    This function calls the following in order:
    1. generate_self_uv_map
    2. generate_pano_to_main_uv_map
    3. generate_main_to_pano_uv_map (with_fg mode)
    4. generate_main_to_pano_uv_map (without_fg mode)
    
    After each UV map is generated, the corresponding warped image is saved for validation.
    
    Args:
        camera_params: Path to camera parameters JSON file
        image: Path to main camera image
        bg_mesh: Path to background 3D mesh file (.glb)
        fg_mesh: Path to foreground 3D mesh file (.glb)
        fg_mask: Path to foreground mask image
        output_dir: Directory to save UV maps and warped images
        pano_w: Panorama width in pixels
        pano_h: Panorama height in pixels
        ior: Index of refraction for the foreground mesh (default: 1.5)
    
    Returns:
        Dictionary with paths to all generated UV maps and warped images
    """
    maxLOD = 5


    print("=" * 80)
    print("Starting combined UV map generation")
    print("=" * 80)
    
    # Get main image dimensions
    with Image.open(image) as img:
        main_w, main_h = img.size
    
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images once at the beginning
    print("Loading images...")
    main_img = torch.from_numpy(np.array(Image.open(image).convert('RGB')).astype(np.float32)).permute(2, 0, 1)
    # Normalize to [-1, 1] range
    main_img = (main_img / 255.0) * 2.0 - 1.0
    
    # Build a synthetic panorama preview image: smooth gradient from top-left to bottom-right.
    x_grad = np.linspace(0.0, 1.0, pano_w, dtype=np.float32)
    y_grad = np.linspace(0.0, 1.0, pano_h, dtype=np.float32)
    xx, yy = np.meshgrid(x_grad, y_grad)
    grad = (xx + yy) * 0.5
    pano_preview_np = np.stack([xx, yy, grad], axis=-1)
    pano_preview_u8 = (np.clip(pano_preview_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    pano_preview_path = os.path.join(output_dir, "panorama_preview.png")
    Image.fromarray(pano_preview_u8).save(pano_preview_path)
    pano_img = torch.from_numpy(pano_preview_u8.astype(np.float32)).permute(2, 0, 1)
    # Normalize to [-1, 1] range
    pano_img = (pano_img / 255.0) * 2.0 - 1.0
    print(f"✓ Loaded main image: {main_w}x{main_h}")
    print(f"✓ Generated pano preview image: {pano_w}x{pano_h} -> {pano_preview_path}")
    
    try:
        # 1. Generate self UV map
        print("\n" + "=" * 80)
        print("Step 1: Generating self UV map")
        print("=" * 80)
        # Check if self UV map already exists or warped_self.png pixels are all black
        if os.path.exists(os.path.join(output_dir, "self_uv_map.npz")):
            print("Self UV map already exists, skipping generation...")
            # Also check if warped_self.png exists and is all black
            warped_self_path = os.path.join(output_dir, "warped_self.png")
            if os.path.exists(warped_self_path):
                warped_img = np.array(Image.open(warped_self_path))
                if np.all(warped_img == 0):
                    print("Warped self image is all black, regenerating...")
                    uv_map_self = generate_self_uv_map(
                        camera_params=camera_params,
                        main_w=main_w,
                        main_h=main_h,
                        bg_mesh=bg_mesh,
                        fg_mesh=fg_mesh,
                        fg_mask=fg_mask,
                        ior=ior,
                        output_dir=output_dir,
                    )
                    self_uv_path = os.path.join(output_dir, "self_uv_map.npz")
                    results['self_uv_map'] = self_uv_path
                    print(f"✓ Regenerated self UV map at: {self_uv_path}")
                    
                    # Warp and save self image
                    print("Warping main image with self UV map...")
                    warped = laplacian_pyramid_warping(main_img, uv_map_self, levels=6, interpolation="trilinear")
                    warped_np = warped.permute(1, 2, 0).detach().cpu().numpy()
                    warped_np = (np.clip(warped_np, -1.0, 1.0) + 1.0) * 0.5
                    warped_np = (warped_np * 255.0).astype(np.uint8)
                    warped_np[~np.isfinite(warped_np)] = 0
                    Image.fromarray(warped_np).save(warped_self_path)
                    results['warped_self'] = warped_self_path
                    print(f"✓ Saved warped self image to {warped_self_path}")
                else:
                    print("Warped self image exists and is not all black, skipping...")
            else:
                print("Warped self image not found, regenerating...")
                uv_map_self = generate_self_uv_map(
                    camera_params=camera_params,
                    main_w=main_w,
                    main_h=main_h,
                    bg_mesh=bg_mesh,
                    fg_mesh=fg_mesh,
                    fg_mask=fg_mask,
                    ior=ior,
                    output_dir=output_dir,
                )
                self_uv_path = os.path.join(output_dir, "self_uv_map.npz")
                results['self_uv_map'] = self_uv_path
                print(f"✓ Regenerated self UV map at: {self_uv_path}")
                
                # Warp and save self image
                print("Warping main image with self UV map...")
                warped = laplacian_pyramid_warping(main_img, uv_map_self, levels=6, interpolation="trilinear")
                warped_np = warped.permute(1, 2, 0).detach().cpu().numpy()
                warped_np = (np.clip(warped_np, -1.0, 1.0) + 1.0) * 0.5
                warped_np = (warped_np * 255.0).astype(np.uint8)
                warped_np[~np.isfinite(warped_np)] = 0
                out_path = os.path.join(output_dir, "warped_self.png")
                Image.fromarray(warped_np).save(out_path)
                results['warped_self'] = out_path
                print(f"✓ Saved warped self image to {out_path}")
        else:
            uv_map_self = generate_self_uv_map(
                camera_params=camera_params,
                main_w=main_w,
                main_h=main_h,
                bg_mesh=bg_mesh,
                fg_mesh=fg_mesh,
                fg_mask=fg_mask,
                ior=ior,
                output_dir=output_dir,
            )
            self_uv_path = os.path.join(output_dir, "self_uv_map.npz")
            results['self_uv_map'] = self_uv_path
            print(f"✓ Generated self UV map at: {self_uv_path}")
            
            # Warp and save self image
            print("Warping main image with self UV map...")
            # Load self UV map
            warped = laplacian_pyramid_warping(main_img, uv_map_self, levels=6, interpolation="trilinear")
            warped_np = warped.permute(1, 2, 0).detach().cpu().numpy()
            warped_np = (np.clip(warped_np, -1.0, 1.0) + 1.0) * 0.5
            warped_np = (warped_np * 255.0).astype(np.uint8)
            warped_np[~np.isfinite(warped_np)] = 0
            out_path = os.path.join(output_dir, "warped_self.png")
            Image.fromarray(warped_np).save(out_path)
            results['warped_self'] = out_path
            print(f"✓ Saved warped self image to {out_path}")
        
        # 2. Generate pano-to-main UV map
        print("\n" + "=" * 80)
        print("Step 2: Generating pano-to-main UV map")
        print("=" * 80)
        # Check if pano_to_main UV map already exists to avoid redundant computation
        if os.path.exists(os.path.join(output_dir, "pano_to_main_uv.npz")):
            print("Pano-to-main UV map already exists, skipping generation...")
        else:
            uv_map_pano_to_main = generate_pano_to_main_uv_map(
                camera_params=camera_params,
                main_w=main_w,
                main_h=main_h,
                bg_mesh=bg_mesh,
                fg_mesh=fg_mesh,
                pano_w=pano_w,
                pano_h=pano_h,
                output_dir=output_dir,
                ior=ior,
            )
            pano_to_main_path = os.path.join(output_dir, "pano_to_main_uv.npz")
            results['pano_to_main_uv_map'] = pano_to_main_path
            print(f"✓ Generated pano-to-main UV map at: {pano_to_main_path}")
            
            # Warp and save pano-to-main image
            print("Warping panorama image to main view...")
            warped = laplacian_pyramid_warping(pano_img, uv_map_pano_to_main, levels=maxLOD, interpolation="trilinear")
            warped_np = warped.permute(1, 2, 0).detach().cpu().numpy()
            warped_np = (np.clip(warped_np, -1.0, 1.0) + 1.0) * 0.5
            warped_np = (warped_np * 255.0).astype(np.uint8)
            warped_np[~np.isfinite(warped_np)] = 0
            out_path = os.path.join(output_dir, "warped_pano_to_main.png")
            Image.fromarray(warped_np).save(out_path)
            results['warped_pano_to_main'] = out_path
            print(f"✓ Saved warped pano-to-main image to {out_path}")
            
            # Warp and save pano-to-main reflection image
            print("Warping panorama image to main view using reflections...")
            reflection_uv_path = pano_to_main_path.replace('.npz', '_reflection.npz')
            if os.path.exists(reflection_uv_path):
                reflection_uv_map_data = np.load(reflection_uv_path)
                uv_map_reflection = reflection_uv_map_data["uv"].astype(np.float32)
                warped = laplacian_pyramid_warping(pano_img, uv_map_reflection, levels=maxLOD, interpolation="trilinear")
                warped_np = warped.permute(1, 2, 0).detach().cpu().numpy()
                warped_np = (np.clip(warped_np, -1.0, 1.0) + 1.0) * 0.5
                warped_np = (warped_np * 255.0).astype(np.uint8)
                warped_np[~np.isfinite(warped_np)] = 0
                out_path = os.path.join(output_dir, "warped_pano_to_main_reflection.png")
                Image.fromarray(warped_np).save(out_path)
                results['warped_pano_to_main_reflection'] = out_path
                results['pano_to_main_uv_reflection'] = reflection_uv_path
                print(f"✓ Saved warped pano-to-main reflection image to {out_path}")
            else:
                print(f"⚠ Reflection UV map not found at {reflection_uv_path}, skipping reflection warping")
        
        # 3. Generate main-to-pano UV map (with_fg mode)
        print("\n" + "=" * 80)
        print("Step 3: Generating main-to-pano UV map (with_fg mode)")
        print("=" * 80)
        # Check if main-to-pano UV map (with_fg) already exists to avoid redundant computation
        if os.path.exists(os.path.join(output_dir, "main_to_pano_uv_with_fg.npz")):
            print("Main-to-pano UV map (with_fg) already exists, skipping generation...")
        else:
            uv_map_with_fg = generate_main_to_pano_uv_map(
                camera_params=camera_params,
                main_w=main_w,
                main_h=main_h,
                bg_mesh=bg_mesh,
                fg_mesh=fg_mesh,
                pano_w=pano_w,
                pano_h=pano_h,
                mode="with_fg",
                output_dir=output_dir,
            )
            uv_map_path_with_fg = os.path.join(output_dir, "main_to_pano_uv_with_fg.npz")
            results['main_to_pano_uv_map_with_fg'] = uv_map_path_with_fg
            print(f"✓ Generated main-to-pano UV map (with_fg) at: {uv_map_path_with_fg}")
            
            # Warp and save main-to-pano image (with_fg)
            print("Warping main image to panorama (with_fg)...")
            warped = laplacian_pyramid_warping(main_img, uv_map_with_fg, levels=maxLOD, interpolation="trilinear")
            warped_np = warped.permute(1, 2, 0).detach().cpu().numpy()
            warped_np = (np.clip(warped_np, -1.0, 1.0) + 1.0) * 0.5
            warped_np = (warped_np * 255.0).astype(np.uint8)
            warped_np[~np.isfinite(warped_np)] = 0
            out_path = os.path.join(output_dir, "warped_main_to_pano_with_fg.png")
            Image.fromarray(warped_np).save(out_path)
            results['warped_main_to_pano_with_fg'] = out_path
            print(f"✓ Saved warped main-to-pano image (with_fg) to {out_path}")
        
        # 4. Generate main-to-pano UV map (without_fg mode)
        print("\n" + "=" * 80)
        print("Step 4: Generating main-to-pano UV map (without_fg mode)")
        print("=" * 80)
        # Check if main-to-pano UV map (without_fg) already exists to avoid redundant computation
        if os.path.exists(os.path.join(output_dir, "main_to_pano_uv_without_fg.npz")):
            print("Main-to-pano UV map (without_fg) already exists, skipping generation...")
        else:
            uv_map_without_fg = generate_main_to_pano_uv_map(
                camera_params=camera_params,
                main_w=main_w,
                main_h=main_h,
                bg_mesh=bg_mesh,
                fg_mesh=fg_mesh,
                pano_w=pano_w,
                pano_h=pano_h,
                mode="without_fg",
                output_dir=output_dir,
            )
            uv_map_path_without_fg = os.path.join(output_dir, "main_to_pano_uv_without_fg.npz")
            results['main_to_pano_uv_map_without_fg'] = uv_map_path_without_fg
            print(f"✓ Generated main-to-pano UV map (without_fg) at: {uv_map_path_without_fg}")
            
            # Warp and save main-to-pano image (without_fg)
            print("Warping main image to panorama (without_fg)...")
            warped = laplacian_pyramid_warping(main_img, uv_map_without_fg, levels=maxLOD, interpolation="trilinear")
            warped_np = warped.permute(1, 2, 0).detach().cpu().numpy()
            warped_np = (np.clip(warped_np, -1.0, 1.0) + 1.0) * 0.5
            warped_np = (warped_np * 255.0).astype(np.uint8)
            warped_np[~np.isfinite(warped_np)] = 0
            out_path = os.path.join(output_dir, "warped_main_to_pano_without_fg.png")
            Image.fromarray(warped_np).save(out_path)
            results['warped_main_to_pano_without_fg'] = out_path
            print(f"✓ Saved warped main-to-pano image (without_fg) to {out_path}")
        
        print("\n" + "=" * 80)
        print("All UV maps and warped images generated successfully!")
        print("=" * 80)
        for key, path in results.items():
            print(f"  {key}: {path}")
        
        return results
        
    except Exception as e:
        print(f"\nError during UV map generation: {e}")
        import traceback
        traceback.print_exc()
        raise


def laplacian_pyramid_warping(
    image: torch.Tensor,
    uv_map: np.ndarray = None,
    levels: int = 5,
    interpolation: str = "trilinear",
) -> torch.Tensor:
    """
    Warp an image to the target view using a UV map and LOD.

    Args:
        image: Input image tensor [C, H, W]
        uv_map: Preloaded UV map array [H, W, 2] (preferred for efficiency)
        levels: Number of pyramid levels
        interpolation: Interpolation method ("trilinear", "bilinear", or "nearest")

    Steps:
      1) Build Gaussian pyramid from the input image
      2) Load UV map (main_h, main_w, 2) with entries in pano pixels; compute LOD
      3) Compute LOD map from the UV map
      4) Use LOD + UV to sample from the Gaussian pyramid into the target view
      5) Return the warped target-view image

    Returns:
        Torch tensor [C, H, W] with values expected in [-1, 1].
    """
    if levels <= 0:
        raise ValueError("levels must be >= 1")
    
    if interpolation not in ["trilinear", "bilinear", "nearest"]:
        raise ValueError("interpolation must be 'trilinear', 'bilinear', or 'nearest'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Build Gaussian pyramid from the input image
    image = image.to(device).float()
    if image.ndim == 3:  # [C, H, W]
        pass
    elif image.ndim == 4:  # [1, C, H, W] or [N, C, H, W]
        if image.shape[0] != 1:
            raise ValueError("image batch dimension must be 1")
        image = image.squeeze(0)  # [C, H, W]
    else:
        raise ValueError("image tensor must be 3D or 4D")
    
    # Get original image dimensions for UV mapping
    H0, W0 = image.shape[1], image.shape[2]
    g_pyr = build_gaussian_pyramid(image, levels=levels)

    # 2) Load or use preloaded UV map (main_h, main_w, 2) with entries in pano pixels; compute LOD
    if uv_map is not None:
        uv = uv_map.astype(np.float32) if uv_map.dtype != np.float32 else uv_map
    else:
        raise ValueError("uv_map")
    
    if uv.ndim != 3 or uv.shape[2] != 2:
        raise ValueError("UV map must have shape (H, W, 2)")
    main_h, main_w = int(uv.shape[0]), int(uv.shape[1])

    # 3) Compute LOD map from the UV map
    lod = _compute_lod_level(uv, maxLOD=levels - 1)
    lod_clamped = lod.clamp(0.0, float(levels - 1))
    level_idx = torch.round(lod_clamped).to(torch.int64)  # [H, W]

    # 4) Use LOD + UV to sample from a stacked pyramid volume with a 3D grid
    uv = torch.from_numpy(uv).to(device)
    valid_mask = (uv[..., 0] >= 0.0) & (uv[..., 1] >= 0.0)

    # Upsample all pyramid levels to the highest resolution and stack along depth (levels)
    # Resulting volume has shape [1, C, D=levels, H, W]
    target_h, target_w = H0, W0
    upsampled_levels: List[torch.Tensor] = []
    for l in range(levels):
        lvl = g_pyr[l]
        if lvl.shape[1] != target_h or lvl.shape[2] != target_w:
            lvl_up = F.interpolate(lvl.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=True).squeeze(0)
        else:
            lvl_up = lvl
        # Ensure NaNs are preserved from source
        upsampled_levels.append(lvl_up)
    volume = torch.stack(upsampled_levels, dim=0)              # [D, C, H, W]
    volume = volume.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1, C, D, H, W]

    # Build normalized sampling grid in [-1, 1] for (x=u, y=v, z=lod)
    with torch.no_grad():
        u_pix = uv[..., 0]
        v_pix = uv[..., 1]
        denom_w = max(1, target_w - 1)
        denom_h = max(1, target_h - 1)
        u01 = (u_pix / float(denom_w)).clamp(0.0, 1.0)
        v01 = (v_pix / float(denom_h)).clamp(0.0, 1.0)
        # normalize to [-1, 1]
        u_norm = u01 * 2.0 - 1.0
        v_norm = v01 * 2.0 - 1.0
        # lod already computed above; normalize to [-1, 1]
        max_lod = max(1.0, float(levels - 1))
        lod_norm = (lod_clamped * (2.0 / max_lod)) - 1.0  # [H, W]

        # Compose 5D grid: [N=1, outD=1, outH, outW, 3], order (x=W, y=H, z=D)
        grid_5d = torch.stack((u_norm, v_norm, lod_norm), dim=-1)  # [H,W,3]
        grid_5d = grid_5d.unsqueeze(0).unsqueeze(1)                # [1,1,H,W,3]

        # Invalidate grid where uv is invalid
        invalid = ~valid_mask
        if bool(invalid.any()):
            g = grid_5d.view(1, 1, -1, 3)
            inv_flat = invalid.view(-1)
            g[:, :, inv_flat, :] = float('nan')
            grid_5d = g.view_as(grid_5d)

    # Single 3D grid_sample over the stacked volume
    # For 5D tensors, 'bilinear' mode performs trilinear interpolation
    sampled_5d = F.grid_sample(
        volume,
        grid_5d,
        mode="bilinear" if interpolation == "trilinear" else interpolation,
        padding_mode="zeros",
        align_corners=True,
    )  # [1, C, 1, H, W]

    # Drop the singleton depth dimension
    warped = sampled_5d.squeeze(2)  # [1, C, H, W]
    # Mark invalid pixels as NaN (based on invalid mask, not value check)
    # This prevents legitimate black pixels (0.0) from being converted to NaN
    invalid_4d = invalid.unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
    warped = torch.where(invalid_4d.expand_as(warped), torch.full_like(warped, float('nan')), warped)
    warped = warped.squeeze(0)
    
    return warped


# Test the warping function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate UV maps for image warping between main and panorama views"
    )
    
    # Required arguments
    parser.add_argument("--camera_params", type=str, required=True, help="Path to camera parameters JSON file")
    parser.add_argument("--image", type=str, required=True, help="Path to main camera image")
    parser.add_argument("--bg_mesh", type=str, required=True, help="Path to background 3D mesh file (.glb)")
    parser.add_argument("--fg_mesh", type=str, required=True, help="Path to foreground 3D mesh file (.glb)")
    parser.add_argument("--fg_mask", type=str, required=True, help="Path to foreground mask image")
    
    # Optional arguments
    parser.add_argument("--output_dir", type=str, default="results/warpings", help="Directory to save UV maps and warped images (default: results/warpings)")
    parser.add_argument("--pano_w", type=int, default=2048, help="Panorama width in pixels (default: 2048)")
    parser.add_argument("--pano_h", type=int, default=1024, help="Panorama height in pixels (default: 1024)")
    parser.add_argument("--ior", type=float, default=1.5, help="Index of refraction for the foreground mesh (default: 1.5)")

    args = parser.parse_args()
    
    import time
    start_time = time.time()
    
    generate_uv_map(
        camera_params=args.camera_params,
        image=args.image,
        bg_mesh=args.bg_mesh,
        fg_mesh=args.fg_mesh,
        fg_mask=args.fg_mask,
        output_dir=args.output_dir,
        pano_w=args.pano_w,
        pano_h=args.pano_h,
        ior=args.ior,
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

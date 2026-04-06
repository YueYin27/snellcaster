#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
import sys

# Import GPU acceleration functions from ray_tracer
from utils.ray_tracer import get_available_devices, ray_triangle_intersection_torch
BATCH_SIZE = 64


def load_camera_params(json_file):
    """Load camera parameters from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_camera_matrix(intrinsics, width, height):
    """Create camera matrix from intrinsics and image dimensions."""
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]
    
    # Scale intrinsics to image dimensions
    fx_scaled = fx * width
    fy_scaled = fy * height
    cx_scaled = cx * width
    cy_scaled = cy * height
    
    return np.array([
        [fx_scaled, 0, cx_scaled],
        [0, fy_scaled, cy_scaled],
        [0, 0, 1]
    ])


def ray_mesh_intersection_torch_mask(mesh, ray_origins, ray_directions, devices):
    """GPU-accelerated batch ray-mesh intersection returning boolean hits per ray.
    Optimized for mask generation - returns only boolean array, not intersection details.
    """
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    origins = torch.from_numpy(ray_origins).float()
    directions = torch.from_numpy(ray_directions).float()
    
    total_rays = len(origins)
    rays_per_device = total_rays // len(devices)
    all_hits = []
    
    for i, device in enumerate(devices):
        device_start = i * rays_per_device
        device_end = device_start + rays_per_device if i < len(devices) - 1 else total_rays
        
        device_origins = origins[device_start:device_end].to(device)
        device_directions = directions[device_start:device_end].to(device)
        device_vertices = vertices.to(device)
        device_faces = faces.to(device)
        
        # Use the imported ray_triangle_intersection_torch from ray_tracer
        # It returns (intersection_points, valid_rays, valid_triangles)
        # We only need to know which rays hit something
        intersection_points, valid_rays, _ = ray_triangle_intersection_torch(
            device_origins, device_directions, device_vertices, device_faces
        )
        
        # Convert valid_rays indices to boolean array
        device_hits = torch.zeros(device_end - device_start, dtype=torch.bool, device=device)
        if len(valid_rays) > 0:
            device_hits[valid_rays] = True
        
        # Synchronize CUDA operations and get result
        if device.startswith('cuda'):
            device_idx = int(device.split(':')[1]) if ':' in device else 0
            torch.cuda.synchronize(device=device_idx)
        
        all_hits.append(device_hits.cpu().numpy())
        
        # Clear GPU memory after each device to prevent OOM
        del device_origins, device_directions, device_vertices, device_faces
        del intersection_points, valid_rays, device_hits
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    if all_hits:
        hits_bool = np.concatenate(all_hits)
        return hits_bool
    else:
        return np.zeros(total_rays, dtype=bool)


def render_mask(mesh, width, height, fov_x, fov_y):
	"""Render a binary mask of the mesh from the camera view using GPU-accelerated ray casting.
	This uses the same ray generation convention as ray_tracer.py (camera looks along +Y).
	"""
	# Apply Blender-to-our coordinate transformation (same as in ray_tracer.py)
	blender_to_our = np.array([
		[1, 0, 0, 0],
		[0, 0, -1, 0],
		[0, 1, 0, 0],
		[0, 0, 0, 1]
	])
	
	# Extract vertices and faces, transform, and create new mesh
	vertices = np.array(mesh.vertices, dtype=np.float64)
	faces = np.array(mesh.faces, dtype=np.int64)
	
	# Transform vertices
	vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
	vertices_transformed = vertices_homogeneous @ blender_to_our.T
	vertices = vertices_transformed[:, :3]
	
	# Create fresh mesh from transformed vertices
	transformed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

	# Generate ray directions for each pixel (center of pixel) in camera space
	x_coords, y_coords = np.meshgrid(
		np.arange(width) + 0.5,
		np.arange(height) + 0.5,
		indexing='xy'
	)

	# Normalized device coordinates (-1 to 1)
	x_ndc = (x_coords / width) * 2 - 1
	y_ndc = (y_coords / height) * 2 - 1

	# Convert to camera space using FOV (camera looks along +Y)
	fov_x_rad = np.radians(fov_x)
	fov_y_rad = np.radians(fov_y)
	ray_dirs = np.zeros((height, width, 3))
	ray_dirs[:, :, 0] = x_ndc * np.tan(fov_x_rad / 2)  # X
	ray_dirs[:, :, 1] = 1.0                              # Y (forward)
	ray_dirs[:, :, 2] = -y_ndc * np.tan(fov_y_rad / 2)  # Z
	# Normalize directions
	ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)

	# Flatten for batch ray casting
	ray_directions_flat = ray_dirs.reshape(-1, 3)
	ray_origins_flat = np.zeros_like(ray_directions_flat)

	# Get available GPU devices
	devices = get_available_devices()
	
	# GPU-accelerated batch ray casting
	total_rays = len(ray_directions_flat)
	print(f"Casting {total_rays} rays for mask generation...")
	
	# Reduced batch size to avoid OOM errors with high-poly meshes
	# Memory usage scales as O(batch_size * num_faces)
	batch_size = BATCH_SIZE  # Conservative batch size to prevent CUDA OOM
	all_hits = []
	
	print(f"Processing rays in batches of {batch_size}")
	for batch_start in range(0, total_rays, batch_size):
		batch_end = min(batch_start + batch_size, total_rays)
		progress = (batch_start / total_rays) * 100
		bar_length = 50
		filled_length = int(bar_length * batch_start // total_rays)
		bar = '█' * filled_length + '-' * (bar_length - filled_length)
		print(f"\rProgress: |{bar}| {progress:.1f}% ({batch_start}/{total_rays})", end="", flush=True)
		
		batch_origins = ray_origins_flat[batch_start:batch_end]
		batch_directions = ray_directions_flat[batch_start:batch_end]
		
		try:
			batch_hits = ray_mesh_intersection_torch_mask(
				transformed_mesh, batch_origins, batch_directions, devices
			)
			all_hits.append(batch_hits)
			
			# Clear any remaining GPU memory after each batch
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				
		except torch.cuda.OutOfMemoryError as e:
			print(f"\n\n{'='*70}")
			print(f"FATAL ERROR: CUDA Out of Memory at batch {batch_start}-{batch_end}")
			print(f"{'='*70}")
			print(f"Your mesh has too many faces for the current batch size.")
			print(f"Try reducing batch_size further or using a lower-poly mesh.")
			print(f"Error details: {e}")
			print(f"{'='*70}\n")
			sys.exit(1)
		except Exception as e:
			print(f"\n\n{'='*70}")
			print(f"ERROR: Unexpected error at batch {batch_start}-{batch_end}")
			print(f"{'='*70}")
			import traceback
			traceback.print_exc()
			print(f"{'='*70}\n")
			sys.exit(1)
	
	# Final progress bar
	bar = '█' * 50
	print(f"\rProgress: |{bar}| 100.0% ({total_rays}/{total_rays}) - Complete!")
	
	# Concatenate all batch results
	hits_bool = np.concatenate(all_hits)

	# Build mask from hits
	mask = np.zeros((height, width), dtype=np.uint8)
	# Map flat indices back to (y, x)
	indices = np.where(hits_bool)[0]
	if indices.size > 0:
		ys = indices // width
		xs = indices % width
		mask[ys, xs] = 255

	return mask


def main():
    parser = argparse.ArgumentParser(description='Generate object mask from camera view')
    parser.add_argument('camera_params', help='Path to camera parameters JSON file')
    parser.add_argument('width', type=int, help='Image width')
    parser.add_argument('height', type=int, help='Image height')
    parser.add_argument('mesh_file', help='Path to GLB mesh file')
    parser.add_argument('output_mask', help='Path to output mask image')
    
    args = parser.parse_args()
    
    try:
        # Load camera parameters
        camera_params = load_camera_params(args.camera_params)
        
        # Load mesh (handle scene with transforms like in ray_tracer.py)
        scene = trimesh.load(args.mesh_file, force='mesh')
        if hasattr(scene, 'geometry'):
            meshes = []
            for name, mesh_obj in scene.geometry.items():
                if name in scene.graph.nodes:
                    transform = scene.graph.get(name)[0]
                    transformed_mesh = mesh_obj.copy()
                    transformed_mesh.apply_transform(transform)
                    meshes.append(transformed_mesh)
            if len(meshes) == 1:
                mesh = meshes[0]
            else:
                mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene
        
        # Render mask using per-pixel ray casting consistent with ray_tracer.py
        mask = render_mask(
            mesh=mesh,
            width=args.width,
            height=args.height,
            fov_x=camera_params['fov_x'],
            fov_y=camera_params['fov_y']
        )
        
        # Save mask
        mask_image = Image.fromarray(mask)
        mask_image.save(args.output_mask)
        
        print(f"Mask saved to {args.output_mask}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

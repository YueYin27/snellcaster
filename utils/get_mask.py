#!/usr/bin/env python3

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
import sys
from utils.ray_tracer import get_available_devices, ray_mesh_intersection_torch
BATCH_SIZE = 256


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

    Thin wrapper over ray_tracer.ray_mesh_intersection_torch, which already applies
    BVH candidate-triangle culling (so the Moller-Trumbore kernel only sees the few
    triangles whose bounding boxes a ray could hit, instead of every face), caches
    the mesh vertices on each device, and splits rays across devices. For mask
    generation we discard the locations/triangles and keep only which rays hit.
    """
    total_rays = len(ray_origins)
    if total_rays == 0:
        return np.zeros(0, dtype=bool)

    # ray_mesh_intersection_torch returns (locations, ray_indices, tri_indices).
    # A ray hit the mesh iff its index appears in ray_indices.
    _, ray_indices, _ = ray_mesh_intersection_torch(
        mesh, ray_origins, ray_directions, devices
    )

    hits_bool = np.zeros(total_rays, dtype=bool)
    if len(ray_indices) > 0:
        hits_bool[np.asarray(ray_indices, dtype=np.int64)] = True
    return hits_bool


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
	
	# Batch the rays; BVH culling inside ray_mesh_intersection_torch limits the
	# Moller-Trumbore tensor to candidate triangles, so memory no longer scales
	# with the full face count and a larger batch is safe.
	batch_size = BATCH_SIZE
	all_hits = []
	
	print(f"Processing rays in batches of {batch_size}")
	num_batches = (total_rays + batch_size - 1) // batch_size
	last_pct = -1
	for batch_idx, batch_start in enumerate(range(0, total_rays, batch_size)):
		batch_end = min(batch_start + batch_size, total_rays)
		batch_origins = ray_origins_flat[batch_start:batch_end]
		batch_directions = ray_directions_flat[batch_start:batch_end]

		try:
			batch_hits = ray_mesh_intersection_torch_mask(
				transformed_mesh, batch_origins, batch_directions, devices
			)
			all_hits.append(batch_hits)

			pct = int(batch_end * 100 // total_rays)
			if pct != last_pct:
				last_pct = pct
				bar_length = 50
				filled_length = bar_length * batch_end // total_rays
				bar = '█' * filled_length + '-' * (bar_length - filled_length)
				progress_str = (f"Progress: |{bar}| {pct}% "
								f"({batch_end}/{total_rays}) [Batch {batch_idx+1}/{num_batches}]")
				# ljust pads shorter lines so they fully overwrite longer earlier ones.
				print(f"\r{progress_str.ljust(120)}", end="", flush=True)

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

	bar = '█' * 50
	final_str = f"Progress: |{bar}| 100.0% ({total_rays}/{total_rays}) - Complete!"
	print(f"\r{final_str.ljust(120)}")
	
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

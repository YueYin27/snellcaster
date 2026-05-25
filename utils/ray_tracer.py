import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
import os
import sys

BATCH_SIZE = 256

def _get_material_image(mesh: trimesh.Trimesh):
    """
    Try to get a PIL image from a mesh's material if it has a texture.
    Returns None if not available.
    """
    try:
        if mesh.visual is not None and hasattr(mesh.visual, 'material'):
            material = mesh.visual.material
            # Direct image reference
            if hasattr(material, 'image') and material.image is not None:
                return material.image
            # glTF PBR baseColorTexture
            bct = getattr(material, 'baseColorTexture', None)
            if bct is not None:
                img = getattr(bct, 'image', None)
                if img is not None:
                    return img
    except Exception:
        pass
    return None


def load_camera_params(json_file):
    """Load camera parameters from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def load_mesh(mesh_file):
    """
    Load a 3D mesh from file and apply the Blender coordinate transformation.
    
    Args:
        mesh_file: Path to the mesh file (GLB/GLTF)
        
    Returns:
        trimesh.Trimesh: The loaded and transformed mesh
    """
    # Load mesh, force='mesh' to skip problematic visual properties
    scene = trimesh.load(mesh_file, force='mesh')

    # Extract mesh from scene with proper transformations
    if hasattr(scene, 'geometry'):
        # Scene with multiple meshes - combine them with transformations
        meshes = []
        for name, mesh_obj in scene.geometry.items():
            # Get the transformation for this mesh
            if name in scene.graph.nodes:
                transform = scene.graph.get(name)[0]
                # Apply transformation to mesh
                transformed_mesh = mesh_obj.copy()
                transformed_mesh.apply_transform(transform)
                meshes.append(transformed_mesh)

        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            # Combine multiple meshes
            mesh = trimesh.util.concatenate(meshes)
    else:
        # Single mesh
        mesh = scene

    # Apply coordinate transformation (trimesh loaded coordinate to Blender coordinate system)
    transformation_blender = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    # Extract vertices and faces, transform, and rebuild mesh
    vertices = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces, dtype=np.int64)
    
    # Transform vertices
    vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
    vertices_transformed = vertices_homogeneous @ transformation_blender.T
    vertices = vertices_transformed[:, :3]
    
    # Create fresh mesh from transformed vertices
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    return mesh


def create_ray_directions(width, height, fov_x, fov_y, mask=None):
    """Create ray directions for each pixel in the image, optionally filtered by mask."""
    # Create a grid of pixel coordinates (center of each pixel)
    x_coords, y_coords = np.meshgrid(
        np.arange(width) + 0.5,  # Add 0.5 to get pixel center
        np.arange(height) + 0.5,  # Add 0.5 to get pixel center
        indexing='xy'
    )

    # Convert to normalized device coordinates (-1 to 1)
    x_ndc = (x_coords / width) * 2 - 1
    y_ndc = (y_coords / height) * 2 - 1

    # Convert to camera space coordinates
    fov_x_rad = np.radians(fov_x)
    fov_y_rad = np.radians(fov_y)

    # Calculate ray directions in camera space
    # Camera is looking in (0,1,0) direction, so Y should be the main forward direction
    ray_dirs = np.zeros((height, width, 3))
    ray_dirs[:, :, 0] = x_ndc * np.tan(fov_x_rad / 2)  # X direction (left/right)
    ray_dirs[:, :, 1] = 1.0  # Y direction (forward - main camera direction)
    ray_dirs[:, :, 2] = -y_ndc * np.tan(fov_y_rad / 2)  # Z direction (up/down)

    # Normalize ray directions
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=2, keepdims=True)

    # If mask is provided, filter ray directions
    if mask is not None:
        # Create filtered ray directions and pixel coordinates
        mask_bool = mask > 0  # White pixels in mask
        valid_pixels = np.where(mask_bool)

        filtered_ray_dirs = ray_dirs[valid_pixels]
        filtered_pixel_coords = list(zip(valid_pixels[1], valid_pixels[0]))  # (x, y) coordinates

        return filtered_ray_dirs, filtered_pixel_coords

    return ray_dirs, None


BVH_LEAF_SIZE = 32


def _ray_aabb_intersects(origin: np.ndarray, inv_dir: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> bool:
    """Ray vs axis-aligned bounding box test."""
    t1 = (bbox_min - origin) * inv_dir
    t2 = (bbox_max - origin) * inv_dir
    tmin = np.maximum.reduce(np.minimum(t1, t2))
    tmax = np.minimum.reduce(np.maximum(t1, t2))
    return tmax >= max(tmin, 0.0)


def _build_bvh_for_mesh(mesh: trimesh.Trimesh, leaf_size: int = BVH_LEAF_SIZE):
    """Build a simple binary BVH for a mesh and cache nodes as numpy arrays."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        return None

    tri_vertices = vertices[faces]
    tri_bounds_min = tri_vertices.min(axis=1)
    tri_bounds_max = tri_vertices.max(axis=1)
    tri_centers = tri_vertices.mean(axis=1)

    nodes = []

    def _recurse(tri_indices: np.ndarray) -> int:
        bbox_min = tri_bounds_min[tri_indices].min(axis=0)
        bbox_max = tri_bounds_max[tri_indices].max(axis=0)
        node_index = len(nodes)
        nodes.append({
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "left": -1,
            "right": -1,
            "tri_indices": None,
        })

        if tri_indices.size <= leaf_size:
            nodes[node_index]["tri_indices"] = tri_indices
            return node_index

        extents = bbox_max - bbox_min
        axis = int(np.argmax(extents))
        sorted_idx = tri_indices[np.argsort(tri_centers[tri_indices, axis])]
        mid = sorted_idx.size // 2
        if mid == 0 or mid == sorted_idx.size:
            nodes[node_index]["tri_indices"] = tri_indices
            return node_index

        left_child = _recurse(sorted_idx[:mid])
        right_child = _recurse(sorted_idx[mid:])
        nodes[node_index]["left"] = left_child
        nodes[node_index]["right"] = right_child
        return node_index

    _ = _recurse(np.arange(len(faces)))

    bbox_min = np.stack([node["bbox_min"] for node in nodes], axis=0)
    bbox_max = np.stack([node["bbox_max"] for node in nodes], axis=0)
    left = np.array([node["left"] for node in nodes], dtype=np.int32)
    right = np.array([node["right"] for node in nodes], dtype=np.int32)
    tri_refs = [node["tri_indices"] for node in nodes]

    return {
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "left": left,
        "right": right,
        "tri_refs": tri_refs,
        "leaf_size": leaf_size,
        "face_count": len(faces),
    }


def _get_or_create_bvh(mesh: trimesh.Trimesh):
    """Return cached BVH for mesh, building it if necessary."""
    cache = getattr(mesh, "_bvh_cache", None)
    cache_version = getattr(mesh, "_bvh_cache_version", None)
    face_count = len(mesh.faces)
    if cache is None or cache_version != face_count:
        cache = _build_bvh_for_mesh(mesh)
        mesh._bvh_cache = cache
        mesh._bvh_cache_version = face_count
    return cache


def _batch_ray_aabb_any(origins: np.ndarray, inv_dirs: np.ndarray,
                        bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Vectorized slab test: is each (node) hit by ANY of the rays?

    origins, inv_dirs: [R, 3]
    bbox_min, bbox_max: [N, 3]
    returns: bool array of shape [N], True where any ray intersects that node's AABB.
    """
    if origins.size == 0 or bbox_min.shape[0] == 0:
        return np.zeros((bbox_min.shape[0],), dtype=bool)

    # Broadcast to [R, N, 3]
    o = origins[:, None, :]
    inv = inv_dirs[:, None, :]
    bmin = bbox_min[None, :, :]
    bmax = bbox_max[None, :, :]
    t1 = (bmin - o) * inv
    t2 = (bmax - o) * inv
    tmin = np.minimum(t1, t2).max(axis=2)
    tmax = np.maximum(t1, t2).min(axis=2)
    hit = (tmax >= np.maximum(tmin, 0.0))  # [R, N]
    return hit.any(axis=0)  # [N]


def _collect_bvh_candidate_triangles(bvh, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return sorted unique triangle indices whose BVH nodes intersect any ray in the batch.

    Vectorized BFS down the BVH: at each level, test all active nodes against all rays
    in one broadcasted slab test, then descend to children of hit nodes.
    """
    if bvh is None or origins.size == 0:
        return np.array([], dtype=np.int64)

    bbox_min = bvh["bbox_min"]
    bbox_max = bvh["bbox_max"]
    left = bvh["left"]
    right = bvh["right"]
    tri_refs = bvh["tri_refs"]
    face_count = bvh["face_count"]

    origins_f = np.ascontiguousarray(origins, dtype=np.float32)
    dirs_f = np.ascontiguousarray(directions, dtype=np.float32)
    safe_dirs = np.where(np.abs(dirs_f) > 1e-9, dirs_f, np.float32(1e-9))
    inv_dirs = (1.0 / safe_dirs).astype(np.float32)

    # Batch-process rays against nodes in chunks to bound memory of the R×N tensor.
    # Number of nodes is ~2*face_count/leaf_size, typically small.
    n_rays = origins_f.shape[0]
    ray_chunk = max(1, min(n_rays, 4096))

    leaf_tri_lists: list = []
    active = np.array([0], dtype=np.int64)  # start at root
    visited_leaves = 0
    while active.size > 0:
        # Slab-test all rays (chunked) against active nodes
        node_min = bbox_min[active]
        node_max = bbox_max[active]
        any_hit = np.zeros(active.size, dtype=bool)
        for cs in range(0, n_rays, ray_chunk):
            ce = min(cs + ray_chunk, n_rays)
            any_hit |= _batch_ray_aabb_any(origins_f[cs:ce], inv_dirs[cs:ce], node_min, node_max)
            if any_hit.all():
                break
        if not any_hit.any():
            break
        hit_nodes = active[any_hit]

        next_nodes: list = []
        for node_idx in hit_nodes:
            tris = tri_refs[node_idx]
            if tris is not None:
                leaf_tri_lists.append(tris)
                visited_leaves += tris.size
                if visited_leaves >= face_count:
                    return np.arange(face_count, dtype=np.int64)
            else:
                l = left[node_idx]
                r = right[node_idx]
                if l >= 0:
                    next_nodes.append(l)
                if r >= 0:
                    next_nodes.append(r)
        if not next_nodes:
            break
        active = np.array(next_nodes, dtype=np.int64)

    if not leaf_tri_lists:
        return np.array([], dtype=np.int64)

    combined = np.concatenate(leaf_tri_lists)
    return np.unique(combined).astype(np.int64, copy=False)


def get_available_devices():
    """Get list of available CUDA devices."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = [f'cuda:{i}' for i in range(device_count)]
        print(f"Found {device_count} CUDA devices: {devices}")
        return devices
    else:
        print("CUDA not available, falling back to CPU")
        return ['cpu']


def ray_triangle_intersection_torch(ray_origins, ray_directions, vertices, faces):
    """PyTorch ray-triangle intersection (Möller-Trumbore).
    Handles both front-face and back-face intersections (needed for inside-outside ray casting).
    """
    ray_directions = F.normalize(ray_directions, dim=1)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = torch.cross(edge1, edge2, dim=1)
    normal = F.normalize(normal, dim=1)
    ray_origins_expanded = ray_origins.unsqueeze(1)
    ray_directions_expanded = ray_directions.unsqueeze(1)
    v0_expanded = v0.unsqueeze(0)
    edge1_expanded = edge1.unsqueeze(0)
    edge2_expanded = edge2.unsqueeze(0)
    h = torch.cross(ray_directions_expanded, edge2_expanded, dim=2)
    det = torch.sum(edge1_expanded * h, dim=2)
    # Preserve sign when det is close to zero (needed for back-face intersections)
    det_sign = torch.sign(det)
    det_abs = torch.abs(det)
    det = torch.where(det_abs < 1e-8, torch.tensor(1e-8, device=det.device) * det_sign, det)
    s = ray_origins_expanded - v0_expanded
    u = torch.sum(s * h, dim=2) / det
    q = torch.cross(s, edge1_expanded, dim=2)
    v = torch.sum(ray_directions_expanded * q, dim=2) / det
    t = torch.sum(edge2_expanded * q, dim=2) / det
    
    # Accept both front-face (det > 0) and back-face (det < 0) intersections
    # This is needed when casting rays from inside the mesh
    # The Möller-Trumbore algorithm works for both cases
    # Only require that det is not too close to zero and t is positive (forward intersection)
    valid_intersection = (torch.abs(det) > 1e-8) & (t > 1e-8)
    # Barycentric coordinates should be in [0,1] for valid intersections
    valid_intersection = valid_intersection & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1) & ((u + v) <= 1)
    intersection_mask = valid_intersection.any(dim=1)
    if not intersection_mask.any():
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    valid_rays = torch.where(intersection_mask)[0]
    t_valid = t[valid_rays]
    valid_mask = valid_intersection[valid_rays]
    t_valid = torch.where(valid_mask, t_valid, torch.tensor(float('inf'), device=t_valid.device))
    valid_triangles = torch.argmin(t_valid, dim=1)
    intersection_points = ray_origins[valid_rays] + ray_directions[valid_rays] * t[valid_rays, valid_triangles].unsqueeze(1)
    return intersection_points, valid_rays, valid_triangles


def ray_mesh_intersection_torch(mesh, ray_origins, ray_directions, devices):
    bvh = _get_or_create_bvh(mesh)
    vertices_np = np.asarray(mesh.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh.faces, dtype=np.int64)
    ray_origins_np = np.asarray(ray_origins, dtype=np.float32)
    ray_directions_np = np.asarray(ray_directions, dtype=np.float32)

    if ray_origins_np.ndim != 2:
        ray_origins_np = ray_origins_np.reshape(-1, 3)
    if ray_directions_np.ndim != 2:
        ray_directions_np = ray_directions_np.reshape(-1, 3)

    origins = torch.from_numpy(ray_origins_np).float()
    directions = torch.from_numpy(ray_directions_np).float()
    vertices = torch.from_numpy(vertices_np).float()
    faces = torch.from_numpy(faces_np).long()

    total_rays = origins.shape[0]
    if total_rays == 0 or len(faces_np) == 0:
        return np.array([]), np.array([]), np.array([])

    rays_per_device = max(total_rays // len(devices), 1)
    all_locations = []
    all_ray_indices = []
    all_tri_indices = []
    vertex_device_cache = {}

    def _vertices_on_device(dev):
        if dev not in vertex_device_cache:
            vertex_device_cache[dev] = vertices.to(dev)
        return vertex_device_cache[dev]

    for i, device in enumerate(devices):
        device_start = i * rays_per_device
        device_end = device_start + rays_per_device if i < len(devices) - 1 else total_rays
        if device_start >= total_rays:
            continue

        batch_origins_np = ray_origins_np[device_start:device_end]
        batch_directions_np = ray_directions_np[device_start:device_end]
        candidate_triangles = _collect_bvh_candidate_triangles(bvh, batch_origins_np, batch_directions_np)
        if candidate_triangles.size == 0:
            continue

        candidate_idx_tensor = torch.from_numpy(candidate_triangles).long()
        faces_subset = faces.index_select(0, candidate_idx_tensor)

        try:
            device_origins = origins[device_start:device_end].to(device)
            device_directions = directions[device_start:device_end].to(device)
            device_vertices = _vertices_on_device(device)
            device_faces = faces_subset.to(device)

            device_locations, device_ray_indices, device_tri_indices = ray_triangle_intersection_torch(
                device_origins, device_directions, device_vertices, device_faces
            )

            if device.startswith('cuda'):
                device_idx = int(device.split(':')[1]) if ':' in device else 0
                torch.cuda.synchronize(device=device_idx)

            if len(device_locations) > 0:
                device_ray_indices += device_start
                all_locations.append(device_locations.cpu().numpy())
                all_ray_indices.append(device_ray_indices.cpu().numpy())

                tri_indices_np = candidate_triangles[device_tri_indices.cpu().numpy()]
                all_tri_indices.append(tri_indices_np)
        except Exception as e:
            print(f"\nError processing on device {device} (rays {device_start}-{device_end}): {e}")
            import traceback
            traceback.print_exc()
            continue

    if all_locations:
        locations = np.vstack(all_locations)
        ray_indices = np.concatenate(all_ray_indices)
        tri_indices = np.concatenate(all_tri_indices)
        return locations, ray_indices, tri_indices

    return np.array([]), np.array([]), np.array([])


def snell_fn(incident_direction, normal, ior):
    """
    Apply Snell's law to calculate refracted ray direction.
    Mesh normals always point outward. This function checks the cos between
    normal and incident ray to determine if entering or exiting.
    
    Args:
        incident_direction: Normalized incident ray direction
        normal: Normalized surface normal (points outward)
        ior: Index of refraction of glass (e.g., 1.5)
    
    Returns:
        refracted_direction: Normalized refracted ray direction
    """
    # Ensure vectors are normalized
    incident_direction = incident_direction / np.linalg.norm(incident_direction)
    normal = normal / np.linalg.norm(normal)

    # Check cos between normal and incident direction to determine entering/exiting
    # If cos < 0: normal points opposite to ray → entering glass (from air)
    # If cos > 0: normal points same as ray → exiting glass (to air)
    cos_theta = np.dot(incident_direction, normal)
    
    if cos_theta < 0:
        # Entering glass: n1=1.0 (air), n2=ior (glass), ratio = n1/n2 = 1.0/ior
        ior_ratio = 1.0 / ior
        # Normal points outward, but for Snell's law we need it to point into the medium we're coming from
        # Since we're entering, we're coming from air, so normal should point into air (outward is correct)
        snell_normal = normal
    else:
        # Exiting glass: n1=ior (glass), n2=1.0 (air), ratio = n1/n2 = ior/1.0
        ior_ratio = ior / 1.0
        # Normal points outward, but for Snell's law we need it to point into the medium we're coming from
        # Since we're exiting, we're coming from glass, so normal should point into glass (inward)
        snell_normal = -normal

    # Calculate cosine of the angle between surface normal and ray direction
    # Using negative dot product for Snell's law calculation
    c = -np.dot(incident_direction, snell_normal)

    # Calculate sqrt_term for total internal reflection check
    sqrt_term = 1 - (ior_ratio ** 2) * (1 - c ** 2)

    # Check for total internal reflection
    if sqrt_term < 1e-6:
        # Total internal reflection - calculate reflected direction
        refracted_direction = incident_direction + 2 * c * snell_normal
    else:
        # Refracted direction for non-total-reflection cases
        refracted_direction = ior_ratio * incident_direction + (ior_ratio * c - np.sqrt(np.clip(sqrt_term, 0, None))) * snell_normal

    # Normalize the result
    return refracted_direction / np.linalg.norm(refracted_direction)


def fresnel_reflectance(cos_theta_i, n1, n2):
    """
    Calculate Fresnel reflection coefficient using Schlick's approximation.
    
    Args:
        cos_theta_i: Cosine of incident angle (absolute value of dot product between incident and normal)
        n1: Refractive index of first medium (e.g., 1.0 for air)
        n2: Refractive index of second medium (e.g., 1.5 for glass)
    
    Returns:
        R: Fresnel reflection coefficient (0 to 1)
    """
    # Schlick's approximation
    R0 = ((n1 - n2) / (n1 + n2)) ** 2
    cos_theta = abs(cos_theta_i)
    R = R0 + (1.0 - R0) * ((1.0 - cos_theta) ** 5)
    return R


def reflect_ray(incident_direction, normal):
    """
    Calculate reflected ray direction using law of reflection.
    
    Args:
        incident_direction: Normalized incident ray direction
        normal: Normalized surface normal
    
    Returns:
        reflected_direction: Normalized reflected ray direction
    """
    # Ensure vectors are normalized
    incident_direction = incident_direction / np.linalg.norm(incident_direction)
    normal = normal / np.linalg.norm(normal)
    
    # Reflection formula: r = d - 2(d·n)n
    reflected = incident_direction - 2 * np.dot(incident_direction, normal) * normal
    
    return reflected / np.linalg.norm(reflected)


def _snell_batch(incident: np.ndarray, normal: np.ndarray, ior: float) -> np.ndarray:
    """Vectorized Snell's law over [N, 3] arrays. Mirrors snell_fn() semantics.

    - normals point outward (as built from face winding)
    - if cos(incident, normal) < 0: entering medium (n1=1, n2=ior)
    - else: exiting medium (n1=ior, n2=1)
    - on TIR (sqrt_term < 1e-6) returns the reflected direction (matches original)
    """
    incident = incident / (np.linalg.norm(incident, axis=1, keepdims=True) + 1e-12)
    normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)

    cos_theta = np.einsum('ij,ij->i', incident, normal)  # [N]
    entering = cos_theta < 0.0
    ior_ratio = np.where(entering, 1.0 / ior, ior / 1.0)  # [N]
    snell_normal = np.where(entering[:, None], normal, -normal)  # [N, 3]

    c = -np.einsum('ij,ij->i', incident, snell_normal)  # [N]
    sqrt_term = 1.0 - (ior_ratio ** 2) * (1.0 - c ** 2)
    tir = sqrt_term < 1e-6

    # Refracted direction
    refr_scalar = (ior_ratio * c - np.sqrt(np.clip(sqrt_term, 0.0, None)))  # [N]
    refracted = ior_ratio[:, None] * incident + refr_scalar[:, None] * snell_normal
    # TIR fallback: incident + 2*c*snell_normal
    tir_dir = incident + 2.0 * c[:, None] * snell_normal
    out = np.where(tir[:, None], tir_dir, refracted)
    out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out


def cast_rays(fg_mesh, bg_mesh, ray_origins, ray_directions, ior=1.5, return_ray_info=False, return_fg_rays=False, return_reflections=False):
    """Cast rays that refract through fg_mesh (glass, ior=1.5) and then intersect bg_mesh.

    Signature and return are consistent with cast_rays_no_refraction (plus fg_mesh):
    returns (locations, index_ray, index_tri) where intersections are on bg_mesh.
    If return_ray_info=True, also returns ray_tracing_info for visualization.
    If return_fg_rays=True, also returns a set of ray indices that hit fg_mesh.
    If return_reflections=True, also returns reflection directions and Fresnel coefficients at first hit.

    Vectorized: rays that hit fg_mesh are bounced through it in batched ray-mesh passes
    (one batched call per bounce iteration), instead of one Python loop per ray.
    """
    # Flatten rays if needed
    if ray_directions.ndim == 3:
        dirs_for_cast = ray_directions.reshape(-1, 3)
        origins_for_cast = ray_origins.reshape(-1, 3)
    else:
        dirs_for_cast = ray_directions
        origins_for_cast = ray_origins

    total_rays = len(dirs_for_cast)
    print(f"Number of rays to cast: {total_rays}")

    bg_all_locations = []
    bg_all_index_ray = []
    bg_all_index_tri = []
    ray_tracing_info = []  # Vectorized path does not populate this
    all_fg_ray_indices = set()  # Track all rays that hit fg_mesh (global indices)

    # Storage for reflection data at first hit
    reflection_directions = {}  # Maps global ray index to reflection direction
    fresnel_coefficients = {}  # Maps global ray index to Fresnel coefficient R

    # Get CUDA devices
    devices = get_available_devices()

    # Get fg_mesh face normals (pre-compute for efficiency)
    fg_vertices = np.array(fg_mesh.vertices, dtype=np.float32)
    fg_faces = np.array(fg_mesh.faces, dtype=np.int64)
    v0 = fg_vertices[fg_faces[:, 0]]
    v1 = fg_vertices[fg_faces[:, 1]]
    v2 = fg_vertices[fg_faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    fg_face_normals = np.cross(edge1, edge2)
    fg_face_normals = fg_face_normals / (np.linalg.norm(fg_face_normals, axis=1, keepdims=True) + 1e-8)

    batch_size = BATCH_SIZE
    print(f"Processing rays in batches of {batch_size}")

    for batch_start in range(0, total_rays, batch_size):
        batch_end = min(batch_start + batch_size, total_rays)
        progress = (batch_start / total_rays) * 100
        bar_length = 50
        filled_length = int(bar_length * batch_start // total_rays)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\rProgress: |{bar}| {progress:.1f}% ({batch_start}/{total_rays})", end="", flush=True)

        batch_origins = np.asarray(origins_for_cast[batch_start:batch_end], dtype=np.float32)
        batch_directions = np.asarray(dirs_for_cast[batch_start:batch_end], dtype=np.float32)
        batch_n = batch_origins.shape[0]

        # Step 1: Batch intersect with fg_mesh (CUDA-accelerated, one hit per ray = nearest)
        fg_locs_all, fg_idx_ray_all, fg_idx_tri_all = ray_mesh_intersection_torch(
            fg_mesh, batch_origins, batch_directions, devices
        )

        # Boolean mask of which rays (within this batch) hit fg
        hit_fg_mask = np.zeros(batch_n, dtype=bool)
        # Per-batch-ray entry data (only valid where hit_fg_mask is True)
        entry_loc = np.zeros((batch_n, 3), dtype=np.float32)
        entry_tri = np.full(batch_n, -1, dtype=np.int64)
        if len(fg_locs_all) > 0:
            idx = np.asarray(fg_idx_ray_all, dtype=np.int64)
            hit_fg_mask[idx] = True
            entry_loc[idx] = np.asarray(fg_locs_all, dtype=np.float32)
            entry_tri[idx] = np.asarray(fg_idx_tri_all, dtype=np.int64)

        # Track global indices of rays that hit fg_mesh
        if return_fg_rays and hit_fg_mask.any():
            hit_local = np.where(hit_fg_mask)[0]
            all_fg_ray_indices.update((hit_local + batch_start).tolist())

        # --- Rays that missed fg_mesh: direct cast to bg_mesh ---
        miss_local = np.where(~hit_fg_mask)[0]
        if miss_local.size > 0:
            miss_origins = batch_origins[miss_local]
            miss_directions = batch_directions[miss_local]
            bg_locs_direct, bg_rays_direct, bg_tris_direct = ray_mesh_intersection_torch(
                bg_mesh, miss_origins, miss_directions, devices
            )
            if len(bg_locs_direct) > 0:
                bg_rays_direct_global = miss_local[np.asarray(bg_rays_direct, dtype=np.int64)] + batch_start
                bg_all_locations.append(np.asarray(bg_locs_direct, dtype=np.float32))
                bg_all_index_ray.extend(bg_rays_direct_global.tolist())
                bg_all_index_tri.extend(np.asarray(bg_tris_direct, dtype=np.int64).tolist())

        # --- Rays that hit fg_mesh: vectorized refraction ---
        if hit_fg_mask.any():
            active_local = np.where(hit_fg_mask)[0]  # local batch indices, all currently refracting
            initial_dirs = batch_directions[active_local]
            entry_locs = entry_loc[active_local]
            entry_tris = entry_tri[active_local]
            entry_normals = fg_face_normals[entry_tris]  # outward

            # Reflection direction & Fresnel at first hit (vectorized)
            cos_theta_i = np.einsum('ij,ij->i', initial_dirs, entry_normals)  # [A]
            refl_normal = np.where(cos_theta_i[:, None] > 0.0, -entry_normals, entry_normals)
            dot_dn = np.einsum('ij,ij->i', initial_dirs, refl_normal)
            reflected_dirs = initial_dirs - 2.0 * dot_dn[:, None] * refl_normal
            reflected_dirs = reflected_dirs / (np.linalg.norm(reflected_dirs, axis=1, keepdims=True) + 1e-12)
            R0 = ((1.0 - ior) / (1.0 + ior)) ** 2
            fresnel_R = R0 + (1.0 - R0) * ((1.0 - np.abs(cos_theta_i)) ** 5)
            global_active = active_local + batch_start
            for k, g in enumerate(global_active):
                reflection_directions[int(g)] = reflected_dirs[k].copy()
                fresnel_coefficients[int(g)] = float(fresnel_R[k])

            # Bounce iteration state
            current_location = entry_locs.copy()
            current_direction = initial_dirs.copy()
            current_normal = entry_normals.copy()
            # `alive` marks rays still bouncing inside fg_mesh
            alive = np.ones(active_local.size, dtype=bool)
            # `final_hit` / `final_dir` store exit ray data for rays that have exited
            final_hit = np.zeros_like(current_location)
            final_dir = np.zeros_like(current_direction)
            exited = np.zeros(active_local.size, dtype=bool)

            max_iterations = 12
            offset_distance = np.float32(0.01)
            for _ in range(max_iterations):
                if not alive.any():
                    break
                alive_idx = np.where(alive)[0]
                new_direction = _snell_batch(current_direction[alive_idx],
                                             current_normal[alive_idx], ior).astype(np.float32)
                ray_origin_offset = (current_location[alive_idx] + new_direction * offset_distance).astype(np.float32)

                # One batched ray cast for ALL alive rays
                next_locs, next_idx_ray, next_idx_tri = ray_mesh_intersection_torch(
                    fg_mesh, ray_origin_offset, new_direction, devices
                )
                # next_idx_ray is local to alive_idx. Build full result over alive set.
                hit_in_alive = np.zeros(alive_idx.size, dtype=bool)
                if len(next_locs) > 0:
                    nir = np.asarray(next_idx_ray, dtype=np.int64)
                    hit_in_alive[nir] = True
                    new_loc_for_hit = np.asarray(next_locs, dtype=np.float32)
                    new_tri_for_hit = np.asarray(next_idx_tri, dtype=np.int64)

                # Rays that found no next intersection -> exited
                miss_in_alive = ~hit_in_alive
                if miss_in_alive.any():
                    exit_local = alive_idx[miss_in_alive]
                    final_hit[exit_local] = ray_origin_offset[miss_in_alive]
                    final_dir[exit_local] = new_direction[miss_in_alive]
                    exited[exit_local] = True
                    alive[exit_local] = False

                # Rays that hit another fg surface -> continue bouncing
                if hit_in_alive.any():
                    cont_in_alive_idx = np.where(hit_in_alive)[0]
                    cont_local = alive_idx[cont_in_alive_idx]
                    # Map back new_loc/new_tri rows (indexed by nir order) to cont positions
                    # Since hit_in_alive[nir] = True, we can index `new_loc_for_hit` by the
                    # position within nir corresponding to each cont entry. The easiest mapping:
                    # build a per-alive lookup using `nir`.
                    per_alive_loc = np.zeros((alive_idx.size, 3), dtype=np.float32)
                    per_alive_tri = np.full(alive_idx.size, -1, dtype=np.int64)
                    per_alive_loc[nir] = new_loc_for_hit
                    per_alive_tri[nir] = new_tri_for_hit
                    current_location[cont_local] = per_alive_loc[cont_in_alive_idx]
                    current_direction[cont_local] = new_direction[cont_in_alive_idx]
                    current_normal[cont_local] = fg_face_normals[per_alive_tri[cont_in_alive_idx]]

            # Any still-alive rays exhausted iterations without exiting -> drop them
            # (matches original: `if final_hit is None: continue`)
            if not exited.any():
                # No fg-refracted ray exited; nothing to add for fg path
                pass
            else:
                ex_local = np.where(exited)[0]
                fh = final_hit[ex_local]
                fd = final_dir[ex_local]
                bg_offset = np.float32(0.1)
                final_hit_for_bg = (fh - fd * bg_offset).astype(np.float32)

                bg_locs, bg_idx_ray, bg_idx_tri = ray_mesh_intersection_torch(
                    bg_mesh, final_hit_for_bg, fd.astype(np.float32), devices
                )
                if len(bg_locs) > 0:
                    bir = np.asarray(bg_idx_ray, dtype=np.int64)
                    # bir indexes into ex_local; map to active_local, then to global
                    refracted_global = active_local[ex_local[bir]] + batch_start
                    bg_all_locations.append(np.asarray(bg_locs, dtype=np.float32))
                    bg_all_index_ray.extend(refracted_global.tolist())
                    bg_all_index_tri.extend(np.asarray(bg_idx_tri, dtype=np.int64).tolist())

    # Final progress completion
    bar = '█' * 50
    print(f"\rProgress: |{bar}| 100.0% ({total_rays}/{total_rays}) - Complete!")

    if len(bg_all_locations) > 0:
        locations = np.vstack(bg_all_locations)
        index_ray = np.array(bg_all_index_ray, dtype=int)
        index_tri = np.array(bg_all_index_tri, dtype=int)
        
        # Build return tuple based on flags
        result = [locations, index_ray, index_tri]
        if return_ray_info:
            result.append(ray_tracing_info)
        if return_fg_rays:
            result.append(all_fg_ray_indices)
        if return_reflections:
            result.extend([reflection_directions, fresnel_coefficients])
        
        return tuple(result) if len(result) > 3 else (locations, index_ray, index_tri)
    else:
        # Build return tuple for empty results
        result = [np.array([]), np.array([]), np.array([])]
        if return_ray_info:
            result.append([])
        if return_fg_rays:
            result.append(set())
        if return_reflections:
            result.extend([{}, {}])
        
        return tuple(result) if len(result) > 3 else (np.array([]), np.array([]), np.array([]))


def cast_rays_no_refraction(mesh, ray_origins, ray_directions):
    """CUDA-accelerated batched ray->mesh intersection used by warping, no refraction."""
    if ray_directions.ndim == 3:
        height, width, _ = ray_directions.shape
        dirs_for_cast = ray_directions.reshape(-1, 3)
        origins_for_cast = ray_origins.reshape(-1, 3)
    else:
        dirs_for_cast = ray_directions
        origins_for_cast = ray_origins
    total_rays = len(dirs_for_cast)
    print(f"Number of rays to cast: {total_rays}")
    devices = get_available_devices()
    batch_size = BATCH_SIZE
    all_locations = []
    all_index_ray = []
    all_index_tri = []
    print(f"Processing rays in batches of {batch_size}")
    num_batches = (total_rays + batch_size - 1) // batch_size
    for batch_idx, batch_start in enumerate(range(0, total_rays, batch_size)):
        batch_end = min(batch_start + batch_size, total_rays)
        progress = (batch_start / total_rays) * 100
        bar_length = 50
        filled_length = int(bar_length * batch_start // total_rays)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        # Use \r to overwrite the same line, and add spaces at the end to clear any remaining characters
        # Pad with spaces to ensure we clear the entire line (assuming max ~100 characters)
        progress_str = f"Progress: |{bar}| {progress:.1f}% ({batch_start}/{total_rays}) [Batch {batch_idx+1}/{num_batches}]"
        padded_str = progress_str.ljust(120)  # Pad to 120 characters to clear any leftover text
        print(f"\r{padded_str}", end="", flush=True)
        
        batch_origins = origins_for_cast[batch_start:batch_end]
        batch_directions = dirs_for_cast[batch_start:batch_end]
        
        try:
            batch_locations, batch_index_ray, batch_index_tri = ray_mesh_intersection_torch(
                mesh, batch_origins, batch_directions, devices
            )
            if len(batch_locations) > 0:
                batch_index_ray += batch_start
                all_locations.append(batch_locations)
                all_index_ray.append(batch_index_ray)
                all_index_tri.append(batch_index_tri)
        except Exception as e:
            # Only print error on a new line, then continue updating progress on same line
            print(f"\nError processing batch {batch_idx+1} (rays {batch_start}-{batch_end}): {e}")
            import traceback
            traceback.print_exc()
            # Continue with next batch instead of crashing
            continue
    # Final progress bar update on the same line
    bar = '█' * 50
    final_str = f"Progress: |{bar}| 100.0% ({total_rays}/{total_rays}) - Complete!"
    print(f"\r{final_str.ljust(120)}")  # Pad and add newline at the end
    if all_locations:
        locations = np.vstack(all_locations)
        index_ray = np.concatenate(all_index_ray)
        index_tri = np.concatenate(all_index_tri)
        return locations, index_ray, index_tri
    else:
        return np.array([]), np.array([]), np.array([])


def main():
    parser = argparse.ArgumentParser(description='Cast rays from camera and find mesh intersections')
    parser.add_argument('camera_params', help='Path to camera parameters JSON file')
    parser.add_argument('width', type=int, help='Image width')
    parser.add_argument('height', type=int, help='Image height')
    parser.add_argument('mesh_file', help='Path to GLB mesh file')
    parser.add_argument('bg_mesh_file', help='Path to background GLB mesh file')
    parser.add_argument('mask_file', help='Path to mask image file (white pixels = cast rays)')
    parser.add_argument('base_img', help='Path to base image for blending')
    parser.add_argument('--out_npz', default='results/ray_pairs.npz', help='Path to output NPZ file for pixel pairs (default: results/ray_pairs.npz)')

    args = parser.parse_args()

    try:
        # Load camera parameters
        camera_params = load_camera_params(args.camera_params)

        # Load foreground mesh with transformation
        mesh = load_mesh(args.mesh_file)
        print(f"Loaded foreground mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")

        # Load background mesh with transformation
        bg_mesh = load_mesh(args.bg_mesh_file)
        print(f"Loaded background mesh with {len(bg_mesh.vertices)} vertices and {len(bg_mesh.faces)} faces")

        # Load mask if provided
        mask = None
        if args.mask_file:
            mask_image = Image.open(args.mask_file).convert('L')
            mask = np.array(mask_image)

        # Create ray directions for each pixel (filtered by mask if provided)
        ray_directions, pixel_coords = create_ray_directions(
            args.width, 
            args.height, 
            camera_params['fov_x'], 
            camera_params['fov_y'],
            mask
        )

        # Set camera origin (in camera space, camera is at origin)
        if pixel_coords is not None:
            # Filtered rays - create origins for each ray
            ray_origins = np.zeros((len(ray_directions), 3))
        else:
            # All rays - create origins for full image
            ray_origins = np.zeros((args.height, args.width, 3))

        print(f"FG center coordinate after transformation: {mesh.centroid}")
        print(f"BG center coordinate after transformation: {bg_mesh.centroid}")

        # Cast rays and find intersections with refraction, then BG intersection and projection
        locations, index_ray, index_tri, src_pixels, dst_pixels, ray_tracing_info, dst_colors = cast_rays(
            mesh, bg_mesh, ray_origins, ray_directions, pixel_coords, ior=1.5,
            image_width=args.width, image_height=args.height,
            fov_x=camera_params['fov_x'], fov_y=camera_params['fov_y']
        )

        # Save pixel pairs if any
        if len(src_pixels) > 0:
            # Ensure output directory exists
            out_dir = os.path.dirname(args.out_npz)
            if out_dir and not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            np.savez(args.out_npz, src=src_pixels, dst=dst_pixels)
            print(f"Saved {len(src_pixels)} pixel pairs to {args.out_npz}")
            # Save point-color correspondences (dst pixel -> mesh RGBA at first hit)
            color_npz_path = os.path.join(os.path.dirname(args.out_npz) if os.path.dirname(args.out_npz) else 'results', 'point_color_pairs.npz')
            os.makedirs(os.path.dirname(color_npz_path), exist_ok=True)
            np.savez(color_npz_path, dst=dst_pixels, rgba=dst_colors)
            print(f"Saved {len(dst_pixels)} dst pixel-color pairs to {color_npz_path}")
        else:
            print("No pixel pairs to save.")

        # Create refracted replacement visualization if any pairs exist
        if len(src_pixels) > 0:
            from utils.blending import weighted_blending
            # Save NPZ first (if not already saved above), then reuse here
            npz_path = args.out_npz
            if not os.path.exists(npz_path):
                out_dir = os.path.dirname(npz_path)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                np.savez(npz_path, src=src_pixels, dst=dst_pixels)

            # Generate and save weighted blended image
            blended_img = weighted_blending(args.base_img, npz_path, dst_weight=1.0)
            os.makedirs('results', exist_ok=True)
            blended_img.save('results/blended_pixels.png')
            print(f"Saved weighted blended image to results/blended_pixels.png")
        else:
            print("No pixel pairs; skipped refracted replacement.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
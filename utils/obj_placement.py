import numpy as np
import trimesh
import argparse
import sys
import os
import json
from scipy.spatial import cKDTree
from collections import defaultdict
from ray_tracer import load_mesh
from sam3_infer import sam3_infer

# ANSI color codes for terminal output
_RED = '\033[91m'
_GREEN = '\033[92m'
_RESET = '\033[0m'

_CHECK = '✔'
__EXCL = '!!!'


def get_default_bottom_center_cache_path(scene_dir: str) -> str:
    """
    Build default path for caching placed foreground bottom-center coordinates.

    Args:
        scene_dir: Scene directory path

    Returns:
        str: Path to .npy cache file
    """
    return os.path.join(scene_dir, "mesh_fg_bottom_center.npy")


def _min_surface_distance(mesh, points):
    """
    Compute nearest-point distance from points to a mesh surface.

    Args:
        mesh: Target mesh
        points: Query points, shape (N, 3)

    Returns:
        np.ndarray: Distances of shape (N,)
    """
    if points is None or len(points) == 0:
        return np.array([], dtype=np.float64)

    _, distances, _ = trimesh.proximity.closest_point(mesh, np.asarray(points, dtype=np.float64))
    return np.asarray(distances, dtype=np.float64)


def _sample_points(points, max_points):
    """
    Evenly subsample points to bound runtime.
    """
    points = np.asarray(points)
    n = len(points)
    if n <= max_points:
        return points
    idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return points[idx]


def _filter_points_in_aabb(points, aabb_min, aabb_max, pad):
    """
    Keep points inside an expanded axis-aligned bounding box.
    """
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    lo = aabb_min - pad
    hi = aabb_max + pad
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    return points[mask]


def _print_progress(progress, label="", width=28):
    """
    Print a single-line progress bar (0.0 to 1.0).
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    done = int(round(progress * width))
    bar = "#" * done + "-" * (width - done)
    pct = int(round(progress * 100.0))
    suffix = f" {label}" if label else ""
    print(f"\rCollision check [{bar}] {pct:3d}%{suffix}", end="", flush=True)
    if progress >= 1.0:
        print("")


def _any_edge_intersects_mesh(source_mesh, target_mesh, overlap_min, overlap_max, tolerance=1e-5, max_edges=20000, chunk_size=2000, progress_cb=None):
    """
    Check whether any edge of source_mesh intersects/touches target_mesh.

    This catches many face-face intersection cases where vertex containment
    is insufficient.
    """
    if len(source_mesh.vertices) == 0 or len(source_mesh.edges_unique) == 0:
        return False

    v = np.asarray(source_mesh.vertices, dtype=np.float64)
    edges = np.asarray(source_mesh.edges_unique, dtype=np.int64)
    p0 = v[edges[:, 0]]
    p1 = v[edges[:, 1]]

    # Keep only edges near overlap region to reduce ray workload.
    edge_min = np.minimum(p0, p1)
    edge_max = np.maximum(p0, p1)
    near = np.all((edge_max >= (overlap_min - tolerance)) & (edge_min <= (overlap_max + tolerance)), axis=1)
    if not np.any(near):
        return False

    p0 = p0[near]
    p1 = p1[near]

    if len(p0) > max_edges:
        idx = np.linspace(0, len(p0) - 1, num=max_edges, dtype=np.int64)
        p0 = p0[idx]
        p1 = p1[idx]

    edge_vec = p1 - p0
    edge_len = np.linalg.norm(edge_vec, axis=1)

    valid = edge_len > 1e-12
    if not np.any(valid):
        return False

    p0 = p0[valid]
    edge_vec = edge_vec[valid]
    edge_len = edge_len[valid]
    directions = edge_vec / edge_len[:, None]

    # Offset ray starts slightly backwards so touching at endpoints is detected.
    origins = p0 - directions * tolerance
    max_t = edge_len + 2.0 * tolerance

    n = len(origins)
    if n == 0:
        return False

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_origins = origins[start:end]
        chunk_dirs = directions[start:end]

        try:
            locations, ray_indices, _ = target_mesh.ray.intersects_location(
                ray_origins=chunk_origins,
                ray_directions=chunk_dirs,
                multiple_hits=True
            )
        except Exception:
            continue

        if len(locations) > 0:
            for loc, ridx in zip(locations, ray_indices):
                global_idx = start + ridx
                t = np.dot(loc - origins[global_idx], directions[global_idx])
                if -tolerance <= t <= max_t[global_idx]:
                    return True

        if progress_cb is not None:
            progress_cb(end / n)

    return False


def meshes_overlap(mesh_bg, mesh_fg, tolerance=1e-4, show_progress=True):
    """
    Check whether two meshes overlap, including surface touching.

    Overlap is considered True if there is any geometric intersection or
    contact between the two meshes (face touching also counts as overlap).

    Method (in order):
    1) Fast AABB reject
    2) CollisionManager exact check (if python-fcl is available)
    3) Vertex-to-surface touch/proximity check (both directions)
    4) Edge-to-mesh intersection check (both directions)
    5) Watertight containment fallback

    Args:
        mesh_bg: Background mesh (trimesh.Trimesh)
        mesh_fg: Foreground mesh (trimesh.Trimesh)
        tolerance: Numerical tolerance for AABB overlap
        show_progress: Whether to print a progress bar

    Returns:
        bool: True if overlap is detected, False otherwise
    """
    def stage_progress(stage_idx, total_stages, label="", frac=1.0):
        if not show_progress:
            return
        progress = ((stage_idx - 1) + float(np.clip(frac, 0.0, 1.0))) / float(total_stages)
        _print_progress(progress, label=label)

    total_stages = 6

    # Fast reject using AABB overlap
    stage_progress(1, total_stages, label="AABB precheck", frac=0.15)
    bg_min, bg_max = mesh_bg.bounds
    fg_min, fg_max = mesh_fg.bounds

    # No overlap if separated on any axis
    separated = np.any((fg_max < bg_min - tolerance) | (fg_min > bg_max + tolerance))
    stage_progress(1, total_stages, label="AABB precheck", frac=1.0)
    if separated:
        stage_progress(total_stages, total_stages, label="Done", frac=1.0)
        return False

    overlap_min = np.maximum(bg_min, fg_min)
    overlap_max = np.minimum(bg_max, fg_max)

    # Preferred exact check (if python-fcl is available)
    stage_progress(2, total_stages, label="CollisionManager", frac=0.1)
    try:
        manager = trimesh.collision.CollisionManager()
        manager.add_object("background", mesh_bg)
        hit = bool(manager.in_collision_single(mesh_fg))
        stage_progress(2, total_stages, label="CollisionManager", frac=1.0)
        if hit:
            stage_progress(total_stages, total_stages, label="Done", frac=1.0)
            return True
    except Exception:
        stage_progress(2, total_stages, label="CollisionManager unavailable", frac=1.0)

    # Touch/intersection check: vertex distances to opposite mesh surface.
    # If any vertex is on or very near the opposite surface, count as overlap.
    stage_progress(3, total_stages, label="Vertex proximity", frac=0.05)
    try:
        fg_vertices_roi = _filter_points_in_aabb(mesh_fg.vertices, overlap_min, overlap_max, pad=tolerance)
        fg_vertices_roi = _sample_points(fg_vertices_roi, max_points=12000)
        fg_to_bg = _min_surface_distance(mesh_bg, fg_vertices_roi)
        if fg_to_bg.size > 0 and np.any(fg_to_bg <= tolerance):
            stage_progress(3, total_stages, label="Vertex proximity", frac=1.0)
            stage_progress(total_stages, total_stages, label="Done", frac=1.0)
            return True

        bg_vertices_roi = _filter_points_in_aabb(mesh_bg.vertices, overlap_min, overlap_max, pad=tolerance)
        bg_vertices_roi = _sample_points(bg_vertices_roi, max_points=8000)
        bg_to_fg = _min_surface_distance(mesh_fg, bg_vertices_roi)
        if bg_to_fg.size > 0 and np.any(bg_to_fg <= tolerance):
            stage_progress(3, total_stages, label="Vertex proximity", frac=1.0)
            stage_progress(total_stages, total_stages, label="Done", frac=1.0)
            return True

        stage_progress(3, total_stages, label="Vertex proximity", frac=1.0)
    except Exception:
        stage_progress(3, total_stages, label="Vertex proximity skipped", frac=1.0)

    # Face-center touch check (cheaply catches broad face contact without dense edge checks).
    stage_progress(4, total_stages, label="Face-center proximity", frac=0.05)
    try:
        fg_face_centers = _filter_points_in_aabb(mesh_fg.triangles_center, overlap_min, overlap_max, pad=tolerance)
        fg_face_centers = _sample_points(fg_face_centers, max_points=6000)
        center_to_bg = _min_surface_distance(mesh_bg, fg_face_centers)
        if center_to_bg.size > 0 and np.any(center_to_bg <= tolerance):
            stage_progress(4, total_stages, label="Face-center proximity", frac=1.0)
            stage_progress(total_stages, total_stages, label="Done", frac=1.0)
            return True
        stage_progress(4, total_stages, label="Face-center proximity", frac=1.0)
    except Exception:
        stage_progress(4, total_stages, label="Face-center proximity skipped", frac=1.0)

    # Face-face intersection check through edge-ray tests (both directions)
    stage_progress(5, total_stages, label="Edge intersections", frac=0.02)
    if _any_edge_intersects_mesh(
        mesh_fg,
        mesh_bg,
        overlap_min,
        overlap_max,
        tolerance=tolerance,
        max_edges=18000,
        chunk_size=1500,
        progress_cb=lambda f: stage_progress(5, total_stages, label="Edge intersections", frac=0.5 * f)
    ):
        stage_progress(5, total_stages, label="Edge intersections", frac=1.0)
        stage_progress(total_stages, total_stages, label="Done", frac=1.0)
        return True
    if _any_edge_intersects_mesh(
        mesh_bg,
        mesh_fg,
        overlap_min,
        overlap_max,
        tolerance=tolerance,
        max_edges=12000,
        chunk_size=1500,
        progress_cb=lambda f: stage_progress(5, total_stages, label="Edge intersections", frac=0.5 + 0.5 * f)
    ):
        stage_progress(5, total_stages, label="Edge intersections", frac=1.0)
        stage_progress(total_stages, total_stages, label="Done", frac=1.0)
        return True
    stage_progress(5, total_stages, label="Edge intersections", frac=1.0)

    # Final fallback: containment checks for watertight meshes
    stage_progress(6, total_stages, label="Containment fallback", frac=0.1)
    try:
        if mesh_bg.is_watertight and len(mesh_fg.vertices) > 0:
            if np.any(mesh_bg.contains(mesh_fg.vertices)):
                stage_progress(6, total_stages, label="Containment fallback", frac=1.0)
                stage_progress(total_stages, total_stages, label="Done", frac=1.0)
                return True

        if mesh_fg.is_watertight and len(mesh_bg.vertices) > 0:
            if np.any(mesh_fg.contains(mesh_bg.vertices)):
                stage_progress(6, total_stages, label="Containment fallback", frac=1.0)
                stage_progress(total_stages, total_stages, label="Done", frac=1.0)
                return True
    except Exception:
        pass

    stage_progress(6, total_stages, label="Containment fallback", frac=1.0)
    stage_progress(total_stages, total_stages, label="Done", frac=1.0)

    return False


def get_face_normals(mesh):
    """
    Compute face normals for a mesh.
    
    Args:
        mesh: trimesh.Trimesh object
    
    Returns:
        np.ndarray: Face normals of shape (N_faces, 3)
    """
    # Get face vertices
    face_vertices = mesh.vertices[mesh.faces]  # (N_faces, 3, 3)
    
    # Compute face normals using cross product
    v0 = face_vertices[:, 0, :]
    v1 = face_vertices[:, 1, :]
    v2 = face_vertices[:, 2, :]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)
    
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normals = normals / norms
    
    return normals


def is_horizontal_face(normal, threshold_degrees=20):
    """
    Check if a face normal is horizontal (pointing up, close to [0,0,1]).
    
    Args:
        normal: Face normal vector (3,)
        threshold_degrees: Maximum angle deviation from [0,0,1] in degrees
    
    Returns:
        bool: True if face is horizontal
    """
    up_vector = np.array([0, 0, 1])
    # Compute angle between normal and up vector
    dot_product = np.clip(np.dot(normal, up_vector), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg <= threshold_degrees


def get_face_edges(faces):
    """
    Get all edges from faces, with face indices.
    
    Args:
        faces: Face array of shape (N_faces, 3)
    
    Returns:
        dict: Mapping from edge (sorted tuple) to list of face indices
    """
    edge_to_faces = defaultdict(list)
    
    for face_idx, face in enumerate(faces):
        # Each face has 3 edges
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        for edge in edges:
            edge_to_faces[edge].append(face_idx)
    
    return edge_to_faces


def cluster_horizontal_surfaces(mesh, horizontal_face_mask, min_faces_per_surface=100):
    """
    Cluster horizontal faces into separate surfaces by edge connectivity, then merge spatially close surfaces.
    
    Args:
        mesh: trimesh.Trimesh object
        horizontal_face_mask: Boolean array of shape (N_faces,) indicating horizontal faces
        min_faces_per_surface: Minimum number of faces required for a valid surface
    
    Returns:
        list: List of face index arrays, each representing a connected horizontal surface
    """
    horizontal_face_indices = np.where(horizontal_face_mask)[0]
    
    if len(horizontal_face_indices) == 0:
        return []
    
    # Get horizontal faces
    all_faces = mesh.faces
    horizontal_faces = all_faces[horizontal_face_indices]
    vertices = mesh.vertices
    
    # Compute face centers and heights for spatial merging
    face_centers = []
    face_heights = []
    for face in horizontal_faces:
        face_verts = vertices[face]
        center = np.mean(face_verts, axis=0)
        face_centers.append(center)
        face_heights.append(center[2])  # Z coordinate (up in Blender)
    
    face_centers = np.array(face_centers)
    face_heights = np.array(face_heights)
    
    # Step 1: Edge-based clustering
    edge_to_faces = defaultdict(list)
    for local_idx, face in enumerate(horizontal_faces):
        edges = [
            tuple(sorted([int(face[0]), int(face[1])])),
            tuple(sorted([int(face[1]), int(face[2])])),
            tuple(sorted([int(face[2]), int(face[0])]))
        ]
        for edge in edges:
            edge_to_faces[edge].append(local_idx)
    
    # Build adjacency graph: faces sharing an edge are adjacent
    num_horizontal = len(horizontal_face_indices)
    adjacency = defaultdict(set)
    
    for edge, face_list in edge_to_faces.items():
        if len(face_list) > 1:
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    adjacency[face_list[i]].add(face_list[j])
                    adjacency[face_list[j]].add(face_list[i])
    
    # Find connected components using BFS
    visited = np.zeros(num_horizontal, dtype=bool)
    initial_clusters = []
    
    for start_idx in range(num_horizontal):
        if visited[start_idx]:
            continue
        
        cluster = []
        queue = [start_idx]
        visited[start_idx] = True
        
        while queue:
            current = queue.pop(0)
            cluster.append(current)
            
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        if len(cluster) >= min_faces_per_surface:
            global_cluster = [horizontal_face_indices[local_idx] for local_idx in cluster]
            initial_clusters.append(np.array(global_cluster))
    
    if len(initial_clusters) == 0:
        return []
    
    # Step 2: Merge spatially close clusters that are at similar heights
    # Compute cluster properties
    cluster_properties = []
    for cluster in initial_clusters:
        # Get all face centers for this cluster
        cluster_face_indices_local = [np.where(horizontal_face_indices == fid)[0][0] for fid in cluster]
        cluster_centers = face_centers[cluster_face_indices_local]
        cluster_heights = face_heights[cluster_face_indices_local]
        
        center_xy = np.mean(cluster_centers[:, [0, 1]], axis=0)
        height_z = np.mean(cluster_heights)
        height_std = np.std(cluster_heights)
        
        # Compute approximate size (bounding box in XY)
        xy_min = cluster_centers[:, [0, 1]].min(axis=0)
        xy_max = cluster_centers[:, [0, 1]].max(axis=0)
        size_xy = np.linalg.norm(xy_max - xy_min)
        
        cluster_properties.append({
            'cluster': cluster,
            'center_xy': center_xy,
            'height_z': height_z,
            'height_std': height_std,
            'size_xy': size_xy,
            'num_faces': len(cluster)
        })
    
    # Merge clusters that are close in space and height
    height_tolerance = 0.05  # 5cm height difference
    distance_tolerance = 0.2  # 20cm distance in XY plane
    
    merged = [False] * len(cluster_properties)
    final_clusters = []
    
    for i, prop_i in enumerate(cluster_properties):
        if merged[i]:
            continue
        
        # Start a new merged cluster
        merged_cluster_faces = list(prop_i['cluster'])
        merged[i] = True
        
        # Try to merge with other clusters
        for j in range(i + 1, len(cluster_properties)):
            if merged[j]:
                continue
            
            prop_j = cluster_properties[j]
            
            # Check if clusters are at similar height
            height_diff = abs(prop_i['height_z'] - prop_j['height_z'])
            if height_diff > height_tolerance:
                continue
            
            # Check if clusters are close in XY
            distance_xy = np.linalg.norm(prop_i['center_xy'] - prop_j['center_xy'])
            if distance_xy > distance_tolerance:
                continue
            
            # Merge clusters
            merged_cluster_faces.extend(list(prop_j['cluster']))
            merged[j] = True
        
        # Only keep merged clusters that meet minimum size
        if len(merged_cluster_faces) >= min_faces_per_surface:
            final_clusters.append(np.array(merged_cluster_faces))
    
    return final_clusters


def get_surface_center_and_height(mesh, face_indices):
    """
    Compute the bounding box center and height (Z coordinate) of a surface in Blender coordinates.
    In Blender: X=right, Y=forward, Z=up
    
    Args:
        mesh: trimesh.Trimesh object
        face_indices: Array of face indices belonging to the surface
    
    Returns:
        tuple: (center_xy, surface_height_z, center_z) where:
            - center_xy is (x, y) bounding box center
            - surface_height_z is max Z for placement
            - center_z is bounding box center Z coordinate
    """
    # Get all vertices of these faces
    face_vertices = mesh.vertices[mesh.faces[face_indices]]  # (N_faces, 3, 3)
    
    # Flatten to get all vertices
    all_vertices = face_vertices.reshape(-1, 3)  # (N_faces * 3, 3)
    
    # Remove duplicates by rounding to avoid precision issues
    # Use a simple approach: get unique vertices by position
    unique_vertices = np.unique(all_vertices.round(decimals=6), axis=0)
    
    # Compute bounding box of the surface
    bbox_min = np.min(unique_vertices, axis=0)
    bbox_max = np.max(unique_vertices, axis=0)
    
    # Center of bounding box
    center_xy = np.array([
        (bbox_min[0] + bbox_max[0]) / 2.0,  # center X
        (bbox_min[1] + bbox_max[1]) / 2.0   # center Y
    ])
    center_z = (bbox_min[2] + bbox_max[2]) / 2.0  # center Z
    
    # Find the maximum Z (top of the surface) for placing object on top
    surface_height_z = np.max(unique_vertices[:, 2])
    
    return center_xy, surface_height_z, center_z


def get_mesh_bottom_center(mesh):
    """
    Get the center of the bottom of the mesh (lowest Z coordinate) in Blender coordinates.
    In Blender: X=right, Y=forward, Z=up
    
    Args:
        mesh: trimesh.Trimesh object
    
    Returns:
        np.ndarray: Bottom center point (x, y, z)
    """
    vertices = mesh.vertices
    
    # Find the minimum Z coordinate (Z is up in Blender)
    min_z = np.min(vertices[:, 2])
    
    # Find all vertices at or near the bottom (within a small threshold)
    z_threshold = 0.01  # 1cm tolerance
    bottom_vertices = vertices[vertices[:, 2] <= min_z + z_threshold]
    
    # Center in XY plane (X=right, Y=forward in Blender)
    center_xy = np.mean(bottom_vertices[:, [0, 1]], axis=0)
    
    # Use minimum Z
    center_z = min_z
    
    return np.array([center_xy[0], center_xy[1], center_z])


def get_surface_normal(mesh, face_indices):
    """
    Compute the average surface normal of a surface.
    
    Args:
        mesh: trimesh.Trimesh object
        face_indices: Array of face indices belonging to the surface
    
    Returns:
        np.ndarray: Average normalized surface normal (3,)
    """
    # Get face normals
    face_normals = get_face_normals(mesh)
    surface_normals = face_normals[face_indices]
    
    # Average the normals (they should all point roughly the same direction for a horizontal surface)
    avg_normal = np.mean(surface_normals, axis=0)
    
    # Normalize
    norm = np.linalg.norm(avg_normal)
    if norm > 1e-6:
        avg_normal = avg_normal / norm
    else:
        # Fallback to upward direction
        avg_normal = np.array([0, 0, 1])
    
    return avg_normal


def get_surface_normal_near_point(mesh, point_3d):
    """
    Estimate surface normal near a 3D point by nearest face center.

    Args:
        mesh: trimesh.Trimesh object
        point_3d: 3D point in Blender coordinates

    Returns:
        np.ndarray: Normalized normal vector (3,)
    """
    face_centers = mesh.triangles_center
    if len(face_centers) == 0:
        return np.array([0, 0, 1], dtype=np.float64)

    tree = cKDTree(face_centers)
    _, nearest_idx = tree.query(np.asarray(point_3d, dtype=np.float64), k=1)

    face_normals = get_face_normals(mesh)
    normal = np.asarray(face_normals[int(nearest_idx)], dtype=np.float64)
    if normal[2] < 0:
        normal = -normal

    norm = np.linalg.norm(normal)
    if norm > 1e-8:
        normal = normal / norm
    else:
        normal = np.array([0, 0, 1], dtype=np.float64)

    return normal


def rotation_matrix_from_vectors(v1, v2):
    """
    Compute rotation matrix to rotate v1 to v2.
    
    Args:
        v1: Source vector (normalized)
        v2: Target vector (normalized)
    
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Normalize vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # If vectors are already aligned, return identity
    if np.allclose(v1, v2):
        return np.eye(3)
    
    # If vectors are opposite, need special handling
    if np.allclose(v1, -v2):
        # Find a perpendicular vector to rotate around
        if abs(v1[0]) < 0.9:
            perp = np.array([1, 0, 0])
        else:
            perp = np.array([0, 1, 0])
        axis = np.cross(v1, perp)
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        # Compute rotation axis and angle
        axis = np.cross(v1, v2)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    
    # Build rotation matrix using Rodrigues' rotation formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
    
    return R
    """
    Place the foreground object on a horizontal surface in the background mesh.
    
    Args:
        mesh_bg: Background mesh (trimesh.Trimesh)
        mesh_fg: Foreground mesh (trimesh.Trimesh)
    
    Returns:
        trimesh.Trimesh: Updated foreground mesh with new position
    """
    # Step 1: Identify horizontal surfaces
    face_normals = get_face_normals(mesh_bg)
    horizontal_mask = np.array([is_horizontal_face(normal, threshold_degrees=20) 
                                for normal in face_normals])
    
    num_horizontal = np.sum(horizontal_mask)
    
    if num_horizontal == 0:
        raise ValueError("No horizontal surfaces found in background mesh")
    
    # Step 2: Cluster horizontal faces into separate surfaces
    min_faces_per_surface = 100  # Minimum faces to count as a valid surface (increased to filter out small surfaces)
    surface_clusters = cluster_horizontal_surfaces(mesh_bg, horizontal_mask, min_faces_per_surface=min_faces_per_surface)
    
    # Step 3: Filter surfaces that are more than 1.5m below horizontal axis [0,1,0]
    # In Blender coordinates: X=right, Y=forward, Z=up
    # "Below horizontal axis [0,1,0]" means below Y=0 in the forward direction, but we interpret as Z < -1.5
    # Actually, re-reading: "1.5 m below the optical axis" - optical axis is typically the camera's forward direction
    # But the user said "excluding surfaces more than 1.5 m below horizontal axis [0,1,0]" 
    # Since Y is forward and we want to exclude floor, we should exclude surfaces with Z < -1.5 (below origin in up direction)
    valid_surfaces = []
    for cluster in surface_clusters:
        _, height_z, _ = get_surface_center_and_height(mesh_bg, cluster)
        if height_z >= -1.5:
            valid_surfaces.append((cluster, height_z))
    
    if len(valid_surfaces) == 0:
        raise ValueError("No valid horizontal surfaces found (all below -1.5m)")
    
    # Step 4: Choose the surface closest to the horizontal axis (Z-axis, i.e., Z=0)
    best_surface = None
    best_distance = float('inf')
    
    for cluster, height_z in valid_surfaces:
        distance_to_axis = abs(height_z)  # Distance from Z=0
        if distance_to_axis < best_distance:
            best_distance = distance_to_axis
            best_surface = cluster
    
    center_xy, surface_height_z, center_z = get_surface_center_and_height(mesh_bg, best_surface)
    print(f"Selected surface: center=({center_xy[0]:.3f}, {center_xy[1]:.3f}, {center_z:.3f})")
    
    # Step 5: Place mesh_fg on top of the surface
    # Requested order: translate first, then rotate.
    
    # Get original geometry center
    object_center_before = np.mean(mesh_fg.vertices, axis=0)
    print(f"FG object geometry center before operations: ({object_center_before[0]:.3f}, {object_center_before[1]:.3f}, {object_center_before[2]:.3f})")
    
    # Step 5a: Translate first to align tight bottom-center with surface center
    vertices_original = mesh_fg.vertices.copy()
    # Tight bottom (actual min Z) on original (unrotated) mesh
    min_z_orig = np.min(vertices_original[:, 2])
    z_threshold = 0.001  # 1mm tolerance
    bottom_vertices_orig = vertices_original[vertices_original[:, 2] <= min_z_orig + z_threshold]
    
    if len(bottom_vertices_orig) > 0:
        bottom_center_xy_orig = np.mean(bottom_vertices_orig[:, [0, 1]], axis=0)
        bbox_bottom_center_orig = np.array([bottom_center_xy_orig[0], bottom_center_xy_orig[1], min_z_orig])
    else:
        bbox_min = np.min(vertices_original, axis=0)
        bbox_max = np.max(vertices_original, axis=0)
        bbox_bottom_center_orig = np.array([
            (bbox_min[0] + bbox_max[0]) / 2.0,
            (bbox_min[1] + bbox_max[1]) / 2.0,
            min_z_orig
        ])
    
    target_position = np.array([center_xy[0], center_xy[1], center_z])
    translation = target_position - bbox_bottom_center_orig
    vertices_after_translation = vertices_original + translation
    
    # Geometry center after translation
    object_center_after_translation = np.mean(vertices_after_translation, axis=0)
    print(f"FG object geometry center after translation: ({object_center_after_translation[0]:.3f}, {object_center_after_translation[1]:.3f}, {object_center_after_translation[2]:.3f})")
    
    # Step 5b: Rotate around geometry center (after translation) to align up direction with surface normal
    surface_normal = get_surface_normal(mesh_bg, best_surface)
    up_direction = np.array([0, 0, 1])  # Z is up in Blender coordinates
    rotation_matrix = rotation_matrix_from_vectors(up_direction, surface_normal)
    
    vertices_centered = vertices_after_translation - object_center_after_translation
    vertices_rotated = (rotation_matrix @ vertices_centered.T).T
    new_vertices = vertices_rotated + object_center_after_translation
    
    # Geometry center after rotation (should match after-translation center)
    object_center_after_rotation = np.mean(new_vertices, axis=0)
    print(f"FG object geometry center after rotation: ({object_center_after_rotation[0]:.3f}, {object_center_after_rotation[1]:.3f}, {object_center_after_rotation[2]:.3f})")

    # Final de-overlap along Z: align actual bottom center with surface point directly beneath
    min_z_final = np.min(new_vertices[:, 2])
    bottom_vertices_final = new_vertices[new_vertices[:, 2] <= min_z_final + z_threshold]
    if len(bottom_vertices_final) > 0:
        bottom_center_final_xy = np.mean(bottom_vertices_final[:, [0, 1]], axis=0)
    else:
        bottom_center_final_xy = np.mean(new_vertices[:, [0, 1]], axis=0)
    bottom_center_final = np.array([bottom_center_final_xy[0], bottom_center_final_xy[1], min_z_final])

    surface_face_vertices = mesh_bg.vertices[mesh_bg.faces[best_surface]]
    surface_unique_vertices = np.unique(surface_face_vertices.reshape(-1, 3).round(decimals=6), axis=0)
    surface_xy = surface_unique_vertices[:, [0, 1]]
    distances_xy = np.linalg.norm(surface_xy - bottom_center_final_xy, axis=1)
    closest_idx = np.argmin(distances_xy)
    surface_point_under = surface_unique_vertices[closest_idx]

    eps = 0.01  # small upward epsilon to ensure no overlap
    dz = surface_point_under[2] - bottom_center_final[2] + eps
    new_vertices[:, 2] += dz
    object_center_after_rotation[2] += dz
    bottom_center_final[2] += dz
    print(f"FG object geometry center after final z-adjustment: ({object_center_after_rotation[0]:.3f}, {object_center_after_rotation[1]:.3f}, {object_center_after_rotation[2]:.3f})")
    
    # Rotate normals if they exist
    new_vertex_normals = None
    new_face_normals = None
    
    if hasattr(mesh_fg, 'vertex_normals') and mesh_fg.vertex_normals is not None:
        new_vertex_normals = (rotation_matrix @ mesh_fg.vertex_normals.T).T
    
    if hasattr(mesh_fg, 'face_normals') and mesh_fg.face_normals is not None:
        new_face_normals = (rotation_matrix @ mesh_fg.face_normals.T).T
    
    # Create new mesh with rotated and translated vertices
    # Use process=False to avoid any automatic processing that might change orientation
    mesh_fg_placed = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh_fg.faces.copy(),
        vertex_normals=new_vertex_normals,
        face_normals=new_face_normals,
        process=False,
        validate=False
    )
    
    # Preserve visual properties if they exist
    if hasattr(mesh_fg, 'visual') and mesh_fg.visual is not None:
        try:
            mesh_fg_placed.visual = mesh_fg.visual.copy()
        except:
            # If copying visual fails, try to preserve vertex colors
            if hasattr(mesh_fg.visual, 'vertex_colors'):
                mesh_fg_placed.visual = trimesh.visual.ColorVisuals(
                    mesh_fg_placed, 
                    vertex_colors=mesh_fg.visual.vertex_colors.copy()
                )
    
    return mesh_fg_placed


def place_object_on_surface_sam3(mesh_bg, mesh_fg, image_path: str, prompt: str, camera_json_path: str, target_position_3d=None):
    """
    Place the foreground object on a surface detected by SAM3 using camera intrinsics.
    
    Args:
        mesh_bg: Background mesh (trimesh.Trimesh)
        mesh_fg: Foreground mesh (trimesh.Trimesh)
        image_path: Path to input image for SAM3
        prompt: Text prompt for SAM3
        camera_json_path: Path to camera parameters JSON file
    
    Returns:
        trimesh.Trimesh: Updated foreground mesh with new position
    """
    # Step 1: Determine target position and local surface normal
    if target_position_3d is not None:
        intersection_point = np.asarray(target_position_3d, dtype=np.float64).reshape(3)
        print(
            "Using cached 3D target position: "
            f"({intersection_point[0]:.3f}, {intersection_point[1]:.3f}, {intersection_point[2]:.3f})"
        )
        surface_normal = get_surface_normal_near_point(mesh_bg, intersection_point)
    else:
        # Step 1a: Get 2D center point from SAM3
        centers_2d = sam3_infer(image_path, prompt=prompt, vis=False)
        if not centers_2d or centers_2d[0][0] is None:
            raise ValueError("No valid mask centers found from SAM3")

        # Use the first (highest confidence) center
        x_center_2d, y_center_2d = centers_2d[0]
        print(f"SAM3 detected center at 2D pixel coordinates: ({x_center_2d:.1f}, {y_center_2d:.1f})")

        # Step 2: Load camera parameters
        with open(camera_json_path, 'r') as f:
            camera_data = json.load(f)

        fov_x = camera_data['fov_x']
        fov_y = camera_data['fov_y']

        # Load image to get dimensions
        from PIL import Image
        image = Image.open(image_path)
        image_width, image_height = image.size

        # Step 3: Convert 2D pixel coordinates to ray direction
        x_ndc = (x_center_2d / image_width) * 2.0 - 1.0
        y_ndc = (y_center_2d / image_height) * 2.0 - 1.0

        fov_x_rad = np.radians(fov_x)
        fov_y_rad = np.radians(fov_y)

        ray_direction = np.array([
            x_ndc * np.tan(fov_x_rad / 2.0),
            1.0,
            -y_ndc * np.tan(fov_y_rad / 2.0)
        ])
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        ray_origin = np.array([0.0, 0.0, 0.0])
        print(f"Ray origin: {ray_origin}, direction: {ray_direction}")

        # Step 4: Cast ray to find intersection with mesh
        locations, ray_indices, face_indices = mesh_bg.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction]
        )

        if len(locations) == 0:
            raise ValueError("Ray did not intersect with background mesh")

        distances = np.linalg.norm(locations - ray_origin, axis=1)
        closest_idx = np.argmin(distances)
        intersection_point = locations[closest_idx]

        print(
            f"Ray intersection point: ({intersection_point[0]:.3f}, "
            f"{intersection_point[1]:.3f}, {intersection_point[2]:.3f})"
        )

        # Step 5: Find the surface normal at intersection point
        hit_face_idx = face_indices[closest_idx]
        face_normals = get_face_normals(mesh_bg)
        surface_normal = face_normals[hit_face_idx]

        if surface_normal[2] < 0:
            surface_normal = -surface_normal
    
    # Step 6: Place object similar to place_object_on_surface
    # Get original geometry center
    object_center_before = np.mean(mesh_fg.vertices, axis=0)
    print(f"FG object geometry center before operations: ({object_center_before[0]:.3f}, {object_center_before[1]:.3f}, {object_center_before[2]:.3f})")
    
    # Get bottom center of object
    vertices_original = mesh_fg.vertices.copy()
    min_z_orig = np.min(vertices_original[:, 2])
    z_threshold = 0.0
    bottom_vertices_orig = vertices_original[vertices_original[:, 2] <= min_z_orig + z_threshold]
    
    if len(bottom_vertices_orig) > 0:
        bottom_center_xy_orig = np.mean(bottom_vertices_orig[:, [0, 1]], axis=0)
        bbox_bottom_center_orig = np.array([bottom_center_xy_orig[0], bottom_center_xy_orig[1], min_z_orig])
    else:
        bbox_min = np.min(vertices_original, axis=0)
        bbox_max = np.max(vertices_original, axis=0)
        bbox_bottom_center_orig = np.array([
            (bbox_min[0] + bbox_max[0]) / 2.0,
            (bbox_min[1] + bbox_max[1]) / 2.0,
            min_z_orig
        ])
    
    # Target position: intersection point (on surface)
    target_position = intersection_point.copy()
    
    # Translate first
    translation = target_position - bbox_bottom_center_orig
    vertices_after_translation = vertices_original + translation
    
    object_center_after_translation = np.mean(vertices_after_translation, axis=0)
    print(f"FG object geometry center after translation: ({object_center_after_translation[0]:.3f}, {object_center_after_translation[1]:.3f}, {object_center_after_translation[2]:.3f})")
    
    # Rotate to align with surface normal
    up_direction = np.array([0, 0, 1])
    rotation_matrix = rotation_matrix_from_vectors(up_direction, surface_normal)
    
    vertices_centered = vertices_after_translation - object_center_after_translation
    vertices_rotated = (rotation_matrix @ vertices_centered.T).T
    new_vertices = vertices_rotated + object_center_after_translation
    
    object_center_after_rotation = np.mean(new_vertices, axis=0)
    print(f"FG object geometry center after rotation: ({object_center_after_rotation[0]:.3f}, {object_center_after_rotation[1]:.3f}, {object_center_after_rotation[2]:.3f})")
    
    # Add small epsilon offset to ensure object sits on surface
    eps = 0.005
    new_vertices[:, 2] += eps
    object_center_after_rotation[2] += eps
    print(f"FG object geometry center after epsilon offset: ({object_center_after_rotation[0]:.3f}, {object_center_after_rotation[1]:.3f}, {object_center_after_rotation[2]:.3f})")
    
    # Rotate normals if they exist
    new_vertex_normals = None
    new_face_normals = None
    
    if hasattr(mesh_fg, 'vertex_normals') and mesh_fg.vertex_normals is not None:
        new_vertex_normals = (rotation_matrix @ mesh_fg.vertex_normals.T).T
    
    if hasattr(mesh_fg, 'face_normals') and mesh_fg.face_normals is not None:
        new_face_normals = (rotation_matrix @ mesh_fg.face_normals.T).T
    
    # Create new mesh
    mesh_fg_placed = trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh_fg.faces.copy(),
        vertex_normals=new_vertex_normals,
        face_normals=new_face_normals,
        process=False,
        validate=False
    )
    
    # Preserve visual properties
    if hasattr(mesh_fg, 'visual') and mesh_fg.visual is not None:
        try:
            mesh_fg_placed.visual = mesh_fg.visual.copy()
        except:
            if hasattr(mesh_fg.visual, 'vertex_colors'):
                mesh_fg_placed.visual = trimesh.visual.ColorVisuals(
                    mesh_fg_placed,
                    vertex_colors=mesh_fg.visual.vertex_colors.copy()
                )
    
    return mesh_fg_placed


def save_mesh(mesh, output_path):
    """
    Save mesh to a file, transforming back from Blender coordinates to original coordinate system.
    
    Args:
        mesh: trimesh.Trimesh object (in Blender coordinates)
        output_path: Path to save the mesh
    """
    # Transform back from Blender coordinates to original coordinate system
    # Inverse of the transformation in load_mesh:
    # Blender -> Original: X stays X, Y becomes Z, Z becomes -Y
    transformation_back = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # Create a copy to avoid modifying the original
    mesh_to_save = mesh.copy()
    
    # Transform vertices back
    vertices = np.array(mesh_to_save.vertices, dtype=np.float64)
    faces = np.array(mesh_to_save.faces, dtype=np.int64)
    
    # Transform vertices
    vertices_homogeneous = np.hstack([vertices, np.ones((len(vertices), 1))])
    vertices_transformed = vertices_homogeneous @ transformation_back.T
    vertices_back = vertices_transformed[:, :3]
    
    # Create mesh with transformed vertices, preserving faces
    mesh_transformed = trimesh.Trimesh(
        vertices=vertices_back,
        faces=faces,
        process=False,
        validate=False
    )
    
    # Preserve visual properties if they exist
    if hasattr(mesh_to_save, 'visual') and mesh_to_save.visual is not None:
        try:
            mesh_transformed.visual = mesh_to_save.visual.copy()
        except:
            if hasattr(mesh_to_save.visual, 'vertex_colors'):
                mesh_transformed.visual = trimesh.visual.ColorVisuals(
                    mesh_transformed,
                    vertex_colors=mesh_to_save.visual.vertex_colors.copy()
                )
    
    mesh_transformed.export(output_path, file_type='glb')


def main():
    """
    Command-line interface for object placement.
    """
    parser = argparse.ArgumentParser(
        description='Place a foreground object on a horizontal surface in the background mesh'
    )
    parser.add_argument('mesh_bg', help='Path to background mesh GLB file')
    parser.add_argument('mesh_fg', help='Path to foreground mesh GLB file')
    parser.add_argument('output', help='Path to save the updated foreground mesh GLB file')
    parser.add_argument('--camera', help='Path to camera parameters JSON file (uses SAM3-based placement if provided)')
    parser.add_argument('--image', help='Path to input image for SAM3 (required if --camera is provided)')
    parser.add_argument('--prompt', default='tabletop', help='Text prompt for SAM3 (default: tabletop)')
    parser.add_argument('--bottom-center-npy', help='Path to cache/load foreground bottom-center 3D coordinate (.npy). If omitted, scene default is used.')
    parser.add_argument('--no-collision-check', action='store_true', help='Skip collision check after placement.')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.mesh_bg):
        print(f"Error: Background mesh file not found: {args.mesh_bg}")
        sys.exit(1)
    
    if not os.path.exists(args.mesh_fg):
        print(f"Error: Foreground mesh file not found: {args.mesh_fg}")
        sys.exit(1)
    
    # Validate camera/image arguments
    if args.camera:
        if not args.image:
            print("Error: --image is required when --camera is provided")
            sys.exit(1)
        if not os.path.exists(args.camera):
            print(f"Error: Camera JSON file not found: {args.camera}")
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)

    bottom_center_cache_path = None
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Load meshes using load_mesh from ray_tracer to apply coordinate transformation
    print(f"Loading background mesh from: {args.mesh_bg}")
    mesh_bg = load_mesh(args.mesh_bg)
    print(f"  Loaded {len(mesh_bg.vertices)} vertices, {len(mesh_bg.faces)} faces")
    
    print(f"Loading foreground mesh from: {args.mesh_fg}")
    mesh_fg = load_mesh(args.mesh_fg)
    print(f"  Loaded {len(mesh_fg.vertices)} vertices, {len(mesh_fg.faces)} faces")
    
    # Place object
    print(f"\nUsing SAM3-based placement with camera: {args.camera}")
    print(f"Image: {args.image}, Prompt: {args.prompt}")
    scene_dir = os.path.dirname(args.mesh_bg)
    sphere_mesh_path = os.path.join(scene_dir, "mesh_fg_sphere.glb")
    bottom_center_cache_path = (
        args.bottom_center_npy
        if args.bottom_center_npy
        else get_default_bottom_center_cache_path(scene_dir)
    )

    cached_target_3d = None
    if os.path.exists(sphere_mesh_path):
        try:
            mesh_fg_sphere = load_mesh(sphere_mesh_path)
            cached_target_3d = get_mesh_bottom_center(mesh_fg_sphere)
            print(
                "Loaded bottom-center 3D target from sphere mesh: "
                f"{sphere_mesh_path} -> ({cached_target_3d[0]:.3f}, {cached_target_3d[1]:.3f}, {cached_target_3d[2]:.3f})"
            )
        except Exception as e:
            print(f"Warning: failed to load sphere mesh for cached target ({sphere_mesh_path}): {e}")
    elif os.path.exists(bottom_center_cache_path):
        try:
            cached = np.asarray(np.load(bottom_center_cache_path), dtype=np.float32).reshape(-1)
            if cached.shape[0] >= 3:
                cached_target_3d = np.array([cached[0], cached[1], cached[2]], dtype=np.float32)
                print(
                    "Loaded cached bottom-center 3D target from npy: "
                    f"{bottom_center_cache_path} -> ({cached_target_3d[0]:.3f}, {cached_target_3d[1]:.3f}, {cached_target_3d[2]:.3f})"
                )
            else:
                print(f"Warning: cached bottom-center file has invalid shape, ignoring: {bottom_center_cache_path}")
        except Exception as e:
            print(f"Warning: failed to load cached bottom-center from {bottom_center_cache_path}: {e}")

    mesh_fg_placed = place_object_on_surface_sam3(
        mesh_bg,
        mesh_fg,
        args.image,
        args.prompt,
        args.camera,
        target_position_3d=cached_target_3d
    )

    # Check overlap after placement (optional) and print scene status
    scene_name = os.path.splitext(os.path.basename(args.mesh_bg))[0]
    has_overlap = False
    collision_checked = not args.no_collision_check
    if collision_checked:
        has_overlap = meshes_overlap(mesh_bg, mesh_fg_placed)
        if has_overlap:
            print(f"{_RED}{__EXCL} WARNING: overlap detected in scene '{scene_name}' {__EXCL}{_RESET}")
        else:
            print(f"{_GREEN}{_CHECK} No overlap detected for scene '{scene_name}'{_RESET}")
    else:
        print(f"Collision check skipped for scene '{scene_name}' (--no-collision-check)")

    # Cache placed foreground bottom-center 3D coordinate only when no collision is detected
    if args.camera:
        if not collision_checked:
            print(f"Skipped saving bottom-center 3D cache because collision check is disabled: {bottom_center_cache_path}")
        elif has_overlap:
            print(f"Skipped saving bottom-center 3D cache due to overlap: {bottom_center_cache_path}")
        else:
            placed_bottom_center = get_mesh_bottom_center(mesh_fg_placed).astype(np.float32)
            cache_dir = os.path.dirname(bottom_center_cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            np.save(bottom_center_cache_path, placed_bottom_center)
            print(
                "Saved bottom-center 3D cache to: "
                f"{bottom_center_cache_path} -> ({placed_bottom_center[0]:.3f}, {placed_bottom_center[1]:.3f}, {placed_bottom_center[2]:.3f})"
            )
    
    # Save result
    save_mesh(mesh_fg_placed, args.output)
    
    print(f"\nSuccess! Updated mesh saved to: {args.output}")


if __name__ == "__main__":
    main()

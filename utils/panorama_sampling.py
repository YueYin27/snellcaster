import torch
from PIL import Image
import math
import cv2


def direction_to_uv(direction, width, height):
    """Map a 3D unit direction to equirectangular pixel coordinates.

    Coordinate system: X right, Y forward (camera look dir), Z up.

    Inputs:
      - direction: iterable[float] of length 3 (x, y, z). Need not be unit length.
      - width: panorama width in pixels
      - height: panorama height in pixels
    Outputs:
      - (u, v): float pixel coordinates in the range u∈[0,width), v∈[0,height)
    """
    d = torch.as_tensor(direction, dtype=torch.float64)
    norm = torch.linalg.norm(d)
    if norm == 0:
        raise ValueError("direction must be non-zero")
    d = d / norm
    dx, dy, dz = d[0].item(), d[1].item(), d[2].item()

    # Longitude around Z-up axis, with +Y as forward (u center)
    # Center (u = width/2) occurs at direction [0, 1, 0]
    phi = math.atan2(dx, dy)  # longitude in [-pi, pi]

    # Latitude with Z as up; v=0 at dz=+1 (north pole), v=H-1 at dz=-1 (south pole)
    lam = math.asin(max(-1.0, min(1.0, dz)))  # latitude in [-pi/2, pi/2]

    u = (phi + math.pi) / (2.0 * math.pi) * width
    v = (math.pi / 2.0 - lam) / math.pi * height

    u = u % width
    v = float(max(0.0, min(height - 1.0, v)))
    return float(u), float(v)


def uv_to_direction(u, v, width, height):
    """Map equirectangular pixel coordinates to a 3D unit direction.

    Coordinate system: X right, Y forward (camera look dir), Z up.

    Inputs:
      - u, v: float pixel coordinates (u can be outside [0,width) and will wrap)
      - width, height: panorama dimensions in pixels
    Output:
      - np.ndarray shape (3,) direction vector (x, y, z), unit length
    """
    phi = 2.0 * math.pi * (u / width) - math.pi
    lam = math.pi / 2.0 - math.pi * (v / height)
    cos_lam = math.cos(lam)
    # x = cos(lam)*sin(phi), y = cos(lam)*cos(phi) (forward), z = sin(lam) (up)
    d = torch.tensor([cos_lam * math.sin(phi), cos_lam * math.cos(phi), math.sin(lam)], dtype=torch.float64)
    # already unit length by construction
    return d


def sample_bilinear(pano_img_np, u, v):
    """Bilinear sample from an equirectangular panorama (wraps horizontally).

    Inputs:
      - pano_img_np: numpy array of shape (H, W, C) with dtype uint8 or float
      - u, v: float pixel coordinates
    Output:
      - np.ndarray shape (C,) sampled color (same dtype as input)
    """
    if not isinstance(pano_img_np, torch.Tensor):
        raise TypeError("pano_img_np must be a torch.Tensor [H, W, C]")
    H, W = int(pano_img_np.shape[0]), int(pano_img_np.shape[1])
    u0 = int(math.floor(u)) % W
    v0 = int(math.floor(v))
    u1 = (u0 + 1) % W
    v1 = min(v0 + 1, H - 1)

    du = float(u - math.floor(u))
    dv = float(v - math.floor(v))

    c00 = pano_img_np[v0, u0].to(torch.float32)
    c10 = pano_img_np[v0, u1].to(torch.float32)
    c01 = pano_img_np[v1, u0].to(torch.float32)
    c11 = pano_img_np[v1, u1].to(torch.float32)

    c0 = c00 * (1.0 - du) + c10 * du
    c1 = c01 * (1.0 - du) + c11 * du
    out = c0 * (1.0 - dv) + c1 * dv
    return out


def sample_nearest(image, u: float, v: float):
    """Nearest-neighbor sampling at floating coord (u, v) from image [H, W, C].

    Note: horizontally clamped (no wrap) to be consistent with perspective images.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor [H, W, C]")
    H, W = int(image.shape[0]), int(image.shape[1])
    xi = int(round(u))
    yi = int(round(v))
    if xi < 0:
        xi = 0
    elif xi >= W:
        xi = W - 1
    if yi < 0:
        yi = 0
    elif yi >= H:
        yi = H - 1
    return image[yi, xi]


def sample_color_from_panorama(panorama_image, direction):
    """Fetch the color from a panorama for a given 3D direction.

    Inputs:
      - panorama_image: PIL.Image (mode RGB or RGBA or similar), equirectangular
      - direction: iterable[float] of length 3 (x, y, z)
    Output:
      - tuple of floats in [0,1] for RGB (or RGBA if present)
    """
    if not isinstance(panorama_image, Image.Image):
        raise TypeError("panorama_image must be a PIL.Image.Image")
    pano = panorama_image
    width, height = pano.size
    u, v = direction_to_uv(direction, width, height)
    # Convert PIL image to torch [H, W, C] uint8 without numpy
    if pano.mode != 'RGB':
        pano = pano.convert('RGB')
    w, h = pano.size
    # Ensure writable by copying bytes into a new tensor
    byte_tensor = torch.ByteTensor(list(pano.tobytes()))
    pano_t = byte_tensor.view(h, w, 3).to(torch.float32)
    color = sample_bilinear(pano_t, u, v)
    color = color / 255.0
    return tuple(color.detach().cpu().tolist())


def sample_normal_view_from_panorama(panorama_image, viewing_direction, fov_degrees=90, output_width=1024, output_height=512):
    """Sample a normal perspective view from an equirectangular panorama.
    
    This function creates a perspective projection of the panorama as seen from a virtual camera
    positioned at the center of the sphere, looking in the specified direction.
    
    Args:
        panorama_image: PIL.Image, equirectangular panorama (typically 2048x1024)
        viewing_direction: list/tuple of 3 floats, 3D direction vector (x, y, z) where Y is forward
        fov_degrees: float, horizontal field of view in degrees (default: 90)
        output_width: int, width of output image (default: 1024)
        output_height: int, height of output image (default: 512)
    
    Returns:
        PIL.Image: Normal perspective view of the panorama
        
    Note:
        The FOV parameter specifies the horizontal field of view. The vertical FOV is automatically
        adjusted based on the aspect ratio of the output image to prevent stretching.
    """
    if not isinstance(panorama_image, Image.Image):
        raise TypeError("panorama_image must be a PIL.Image.Image")
    
    # Convert direction to unit vector
    direction = torch.as_tensor(viewing_direction, dtype=torch.float64)
    direction = direction / torch.linalg.norm(direction)
    
    # Get panorama dimensions
    pano_width, pano_height = panorama_image.size
    if panorama_image.mode != 'RGB':
        panorama_image = panorama_image.convert('RGB')
    w, h = panorama_image.size
    pano_bytes = torch.ByteTensor(list(panorama_image.tobytes()))
    pano_np = pano_bytes.view(h, w, 3)
    
    # Convert FOV to radians
    fov_rad = math.radians(fov_degrees)
    
    # Create output image
    output_image = torch.zeros((output_height, output_width, int(pano_np.shape[2])), dtype=pano_np.dtype)
    
    # Calculate camera basis vectors
    # Forward direction (camera look direction)
    forward = direction
    
    # Right direction (perpendicular to forward and up)
    # Use cross product with world up (0, 0, 1) to get right vector
    world_up = torch.tensor([0, 0, 1], dtype=torch.float64)
    right = torch.cross(forward, world_up)
    
    # If forward is too close to world up, use a different reference
    if torch.linalg.norm(right) < 1e-6:
        world_right = torch.tensor([1, 0, 0], dtype=torch.float64)
        right = torch.cross(forward, world_right)
    
    right = right / torch.linalg.norm(right)
    
    # Up direction (perpendicular to forward and right)
    up = torch.cross(right, forward)
    up = up / torch.linalg.norm(up)
    
    # Calculate aspect ratio and adjust FOV accordingly
    aspect_ratio = output_width / output_height
    
    # Create coordinate grids for OpenCV remap
    y_coords, x_coords = torch.meshgrid(torch.arange(output_height), torch.arange(output_width), indexing='ij')
    
    # Convert pixel coordinates to normalized device coordinates [-1, 1]
    ndc_x = (2.0 * x_coords / output_width) - 1.0
    ndc_y = (2.0 * y_coords / output_height) - 1.0
    
    # Calculate ray directions in camera space (vectorized)
    ray_cam_x = ndc_x * math.tan(fov_rad / 2.0)
    ray_cam_y = ndc_y * math.tan(fov_rad / 2.0) / aspect_ratio
    ray_cam_z = torch.ones_like(ray_cam_x)
    
    # Convert to world space (vectorized)
    ray_world_x = ray_cam_x * right[0] + ray_cam_y * up[0] + ray_cam_z * forward[0]
    ray_world_y = ray_cam_x * right[1] + ray_cam_y * up[1] + ray_cam_z * forward[1]
    ray_world_z = ray_cam_x * right[2] + ray_cam_y * up[2] + ray_cam_z * forward[2]
    
    # Normalize ray directions
    ray_norms = torch.sqrt(ray_world_x**2 + ray_world_y**2 + ray_world_z**2)
    ray_world_x /= ray_norms
    ray_world_y /= ray_norms
    ray_world_z /= ray_norms
    
    # Convert to UV coordinates
    theta = torch.atan2(ray_world_x, ray_world_y)
    phi = torch.asin(ray_world_z)
    
    u_coords = (theta + math.pi) / (2 * math.pi) * pano_width
    v_coords = (phi + math.pi/2) / math.pi * pano_height
    
    # Clamp coordinates to valid range
    u_coords = torch.clamp(u_coords, 0, pano_width - 1)
    v_coords = torch.clamp(v_coords, 0, pano_height - 1)
    
    # Use OpenCV's optimized remap function
    map_x = u_coords.to(torch.float32).cpu().numpy()
    map_y = v_coords.to(torch.float32).cpu().numpy()
    
    # Convert PIL image to OpenCV format (BGR)
    pano_cv = cv2.cvtColor(pano_np.cpu().numpy(), cv2.COLOR_RGB2BGR)
    
    # Apply remap with bilinear interpolation
    output_cv = cv2.remap(pano_cv, map_x, map_y, cv2.INTER_LINEAR)
    
    # Convert back to RGB and PIL
    output_rgb = cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB)
    output_image = output_rgb
    
    return Image.fromarray(output_image)


def test_normal_view_sampling():
    """Test function to demonstrate normal view sampling from panorama with 6 direction options."""
    import os
    
    print("=== Test: Normal View Sampling from Panorama ===")
    
    # Load the panorama image
    panorama_path = "results/living_room_pano_30.png"
    if not os.path.exists(panorama_path):
        print(f"Panorama file not found: {panorama_path}")
        return
    
    panorama_image = Image.open(panorama_path)
    print(f"Panorama size: {panorama_image.size}")
    
    # 6 direction options - change the direction_name to test different views
    directions = {
        "forward": [0, 1, 0],      # Forward (center of panorama)
        "right": [1, 0, 0],        # Right
        "left": [-1, 0, 0],        # Left  
        "up": [0, 0, 1],           # Up
        "down": [0, 0, -1],        # Down
        "back": [0, -1, 0]         # Backward
    }
    
    # Change this to test different directions: "forward", "right", "left", "up", "down", "back"
    direction_name = "forward"
    
    if direction_name not in directions:
        print(f"Invalid direction name: {direction_name}")
        print(f"Available directions: {list(directions.keys())}")
        return
    
    direction = directions[direction_name]
    print(f"\nSampling {direction_name} view (direction: {direction})")
    
    # Sample normal view
    normal_view = sample_normal_view_from_panorama(
        panorama_image, 
        direction, 
        fov_degrees=120, 
        output_width=1024, 
        output_height=768
    )
    
    # Save the result
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"pano_{direction_name}_view.png")
    normal_view.save(output_path)
    print(f"Saved {direction_name} view: {output_path}")


if __name__ == "__main__":
    # Run normal view sampling test (forward view only)
    test_normal_view_sampling()



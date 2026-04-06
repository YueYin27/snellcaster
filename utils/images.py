import os
import torch
from PIL import Image


def save_single_tweedie_image(image, save_dir, step, timestep, image_name="tweedie_estimate"):
	"""
	Save a single Tweedie estimate image immediately.
	
	Args:
		image: PIL image to save
		save_dir: Directory to save images
		step: Current denoising step
		timestep: Current timestep value
		image_name: Base name for saved images
	"""
	# Create save directory if it doesn't exist
	os.makedirs(save_dir, exist_ok=True)
	# Save individual image immediately
	individual_filename = f"{image_name}_step_{step+1:03d}.png"
	individual_filepath = os.path.join(save_dir, individual_filename)
	image.save(individual_filepath)


def save_tweedie_images_grid(all_images, save_dir, total_steps, image_name="tweedie_estimate", save_step=1):
	"""
	Save all Tweedie estimate images in a grid layout.
	
	Args:
		all_images: List of (step, timestep, image) tuples
		save_dir: Directory to save images
		total_steps: Total number of denoising steps
		image_name: Base name for saved images
		save_step: Save every N steps (1 = save every step, 2 = save every 2 steps, etc.)
	"""
	# Create save directory if it doesn't exist
	os.makedirs(save_dir, exist_ok=True)
	if not all_images:
		print("No images to save in grid")
		return
	# Filter images based on save_step
	filtered_images = []
	for step, timestep, image in all_images:
		if step % save_step == 0:
			filtered_images.append((step, timestep, image))
	if not filtered_images:
		print("No images match the save_step criteria for grid")
		return
	# Calculate grid dimensions (auto-adapt)
	total_images = len(filtered_images)
	# Find the most square-like grid (cols >= rows)
	def best_grid(n):
		best_r, best_c = 1, n
		min_diff = n
		for r in range(1, int(n**0.5)+1):
			if n % r == 0:
				c = n // r
				if c >= r and (c - r) < min_diff:
					best_r, best_c = r, c
					min_diff = c - r
		return best_r, best_c
	rows, cols = best_grid(total_images)
	# Get image dimensions from the first image
	img_width, img_height = filtered_images[0][2].size
	# Create a large canvas for the grid
	grid_width = cols * img_width
	grid_height = rows * img_height
	grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
	# Place images in the grid
	for idx, (step, timestep, image) in enumerate(filtered_images):
		# Calculate position in grid
		row = idx // cols
		col = idx % cols
		# Calculate pixel position
		x = col * img_width
		y = row * img_height
		# Paste image at position
		grid_image.paste(image, (x, y))
	# Save the grid image in the same folder
	grid_filename = f"{image_name}_grid_{total_steps}_steps_save{save_step}.png"
	grid_filepath = os.path.join(save_dir, grid_filename)
	grid_image.save(grid_filepath)
	print(f"Saved Tweedie estimate grid image: {grid_filename} ({rows}x{cols} grid, {total_images} images, save_step={save_step})")


def create_grid_from_saved_images(save_dir, image_name="tweedie_estimate", save_step=1, cols=7):
	"""
	Create a grid from already saved individual images.
	
	Args:
		save_dir: Directory containing the saved images
		image_name: Base name for saved images
		save_step: Save every N steps (1 = save every step, 2 = save every 2 steps, etc.)
		cols: Number of columns in the grid
	"""
	if not os.path.exists(save_dir):
		print(f"Folder not found: {save_dir}")
		return
	# Get all image files in the folder
	image_files = []
	for filename in os.listdir(save_dir):
		if filename.startswith(f"{image_name}_step_") and filename.endswith(".png"):
			# Extract step number from filename
			try:
				step_str = filename.split("_step_")[1].split(".")[0]
				step = int(step_str) - 1  # Convert back to 0-based indexing
				if step % save_step == 0:
					image_files.append((step, os.path.join(save_dir, filename)))
			except (ValueError, IndexError):
				continue
	if not image_files:
		print(f"No valid image files found in {save_dir}")
		return
	# Sort by step number
	image_files.sort(key=lambda x: x[0])
	# Load images
	images = []
	for step, filepath in image_files:
		try:
			image = Image.open(filepath)
			images.append((step, image))
		except Exception as e:
			print(f"Error loading {filepath}: {e}")
	if not images:
		print("No images could be loaded")
		return
	# Calculate grid dimensions
	total_images = len(images)
	rows = (total_images + cols - 1) // cols  # Ceiling division
	# Get image dimensions from the first image
	img_width, img_height = images[0][1].size
	# Create a large canvas for the grid
	grid_width = cols * img_width
	grid_height = rows * img_height
	grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
	# Place images in the grid
	for idx, (step, image) in enumerate(images):
		# Calculate position in grid
		row = idx // cols
		col = idx % cols
		# Calculate pixel position
		x = col * img_width
		y = row * img_height
		# Paste image at position
		grid_image.paste(image, (x, y))
	# Save the grid image
	grid_filename = f"{image_name}_grid_from_saved_{total_images}_images_save{save_step}.png"
	grid_filepath = os.path.join(save_dir, grid_filename)
	grid_image.save(grid_filepath)
	print(f"Created grid from saved images: {grid_filename} ({rows}x{cols} grid, {total_images} images, save_step={save_step})")


def srgb_to_linear(x):
	return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
	return torch.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1/2.4)) - 0.055)


def tensor_to_pil(tensor):
	"""Convert tensor [C, H, W] with arbitrary range to 8-bit PIL image via min/max normalization."""
	import numpy as np
	if tensor.dim() == 4:  # [B, C, H, W]
		tensor = tensor.squeeze(0)

	t = tensor.detach()
	finite_mask = torch.isfinite(t)

	if bool(finite_mask.any()):
		t_norm = (t + 1.0) / 2.0
		t_norm = torch.clamp(t_norm, 0.0, 1.0)
	else:
		t_norm = torch.zeros_like(t)

	arr = t_norm.cpu().numpy()
	arr = np.transpose(arr, (1, 2, 0))
	arr = np.nan_to_num(arr, nan=0.0)
	arr = (arr * 255.0).astype(np.uint8)

	if arr.shape[2] == 1:
		arr = arr[:, :, 0]
		return Image.fromarray(arr, mode="L")
	if arr.shape[2] == 3:
		return Image.fromarray(arr, mode="RGB")
	if arr.shape[2] == 4:
		return Image.fromarray(arr, mode="RGBA")
	return Image.fromarray(arr[:, :, :3], mode="RGB")


def create_pyramid_visualization(pyramid, title_prefix, normalize=False):
	"""Create a visualization of a Laplacian/Gaussian pyramid as a horizontal strip."""
	import numpy as np
	from PIL import ImageDraw, ImageFont
	
	pad = 8
	border = 8
	
	def finite_minmax(t: torch.Tensor):
		finite_mask = torch.isfinite(t)
		if not bool(finite_mask.any()):
			return 0.0, 1.0
		vmin = float(t[finite_mask].min().item())
		vmax = float(t[finite_mask].max().item())
		return vmin, vmax

	if normalize:
		vis_levels = []
		for t in pyramid:
			lev = t.detach().cpu()
			vmin, vmax = finite_minmax(lev)
			if vmax - vmin > 1e-12:
				norm = (lev - vmin) / (vmax - vmin)
			else:
				norm = torch.zeros_like(lev)
			norm = torch.nan_to_num(norm, nan=0.0)
			vis = (norm.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
			vis_levels.append(np.transpose(vis, (1, 2, 0)))
		pyr_np = vis_levels
	else:
		pyr_np = []
		for t in pyramid:
			tt = t.detach().clamp(0, 1)
			tt = torch.nan_to_num(tt, nan=0.0)
			arr = (tt.cpu().numpy() * 255.0).astype(np.uint8)
			pyr_np.append(arr)
		pyr_np = [np.transpose(a, (1, 2, 0)) for a in pyr_np]

	heights = [p.shape[0] for p in pyr_np]
	widths = [p.shape[1] for p in pyr_np]
	max_h = max(heights)
	max_w = max(widths)
	tile_w, tile_h = max_w, max_h
	total_w = len(pyr_np) * tile_w + pad * (len(pyr_np) - 1) + 2 * border
	total_h = tile_h + 2 * border + 28
	canvas = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
	draw = ImageDraw.Draw(canvas)
	try:
		font = ImageFont.load_default()
	except Exception:
		font = None
	x = border
	for i, arr in enumerate(pyr_np):
		base = Image.fromarray(arr)
		tile = base.resize((tile_w, tile_h), resample=Image.NEAREST)
		y = border
		canvas.paste(tile, (x, y))
		label = f"{title_prefix}L{i}: {base.width}x{base.height}"
		draw.text((x, y + tile_h + 4), label, fill=(220, 220, 220), font=font)
		x += tile_w + pad
	return canvas
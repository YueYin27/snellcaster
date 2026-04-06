import argparse
import json
import math
import os
import shutil
import sys

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
	argv = sys.argv
	if "--" in argv:
		argv = argv[argv.index("--") + 1 :]
	else:
		argv = []

	parser = argparse.ArgumentParser(
		description="Load background/foreground meshes and render a JPG from camera.json"
	)
	parser.add_argument("--base_dir", required=True, help="Scene base directory containing fixed input files")
	parser.add_argument("--obj", required=True, help="Foreground object name used in mesh/mask filenames")
	parser.add_argument("--width", type=int, required=True, help="Render width in pixels")
	parser.add_argument("--height", type=int, required=True, help="Render height in pixels")
	parser.add_argument("--output", required=True, help="Output JPG path")
	parser.add_argument("--samples", type=int, default=1024, help="Cycles sample count")
	parser.add_argument(
		"--scene_mode",
		choices=["indoor", "outdoor"],
		default="indoor",
		help="Lighting mode: indoor uses point light; outdoor uses sunlight",
	)
	parser.add_argument(
		"--light_power_scale",
		type=float,
		default=1.0,
		help="Global multiplier for point-light energy (lower than 1.0 for darker renders)",
	)
	# Note: only indoor lighting mode is supported now (point-light based)
	parser.add_argument("--base_image", required=False, help="Optional base image to match lighting tone/power")
	parser.add_argument(
		"--match_to_base",
		action="store_true",
		help="Post-process render to match base-image colors in mesh regions",
	)
	parser.add_argument(
		"--match_strength",
		type=float,
		default=0.9,
		help="Blend strength for base-image color matching in [0,1]",
	)
	parser.add_argument(
		"--out_mask",
		required=False,
		help="Optional output path for FG-to-BG visibility mask (white where FG ray later hits BG)",
	)
	parser.add_argument(
		"--debug_glossy_dir",
		required=False,
		help="Optional directory to save debug glossy/reflection passes",
	)
	parser.add_argument(
		"--out_empty_mask",
		required=False,
		help="Optional output path for mask where camera rays hit no mesh (background/sky)",
	)
	parser.add_argument(
		"--apply_empty_mask",
		action="store_true",
		help="If set, apply empty-background mask to final render (blacken background); do not save mask file",
	)
	parser.add_argument(
		"--save_blend",
		required=False,
		help="Optional output .blend path to save the configured scene for debugging",
	)
	return parser.parse_args(argv)


def save_auxiliary_files(base_dir: str, obj_name: str, out_dir: str) -> None:
	"""Save base image and foreground mask to output directory with fixed names."""
	os.makedirs(out_dir, exist_ok=True)
	parent_dir = os.path.dirname(out_dir) or "."
	os.makedirs(parent_dir, exist_ok=True)

	base_src = os.path.join(base_dir, "image.jpg")
	base_dst = os.path.join(parent_dir, "base_image.jpg")
	if os.path.exists(base_src):
		try:
			shutil.copy2(base_src, base_dst)
		except Exception:
			print(f"[render_gt] Failed to copy base image: {base_src}")
	else:
		print(f"[render_gt] Base image missing, skip save: {base_src}")

	mask_src = os.path.join(base_dir, f"mask_fg_{obj_name}.png")
	mask_dst = os.path.join(out_dir, "mask_fg.jpg")
	if not os.path.exists(mask_src):
		print(f"[render_gt] FG mask missing, skip save: {mask_src}")
		return

	# Convert PNG mask to JPG when PIL is available; otherwise skip to avoid invalid extension/data mismatch.
	try:
		from PIL import Image
		Image.open(mask_src).convert("L").save(mask_dst, quality=95)
	except Exception:
		print(f"[render_gt] PIL unavailable or mask conversion failed, skip save: {mask_src}")


def clear_scene() -> None:
	bpy.ops.object.select_all(action="SELECT")
	bpy.ops.object.delete(use_global=False)


def import_mesh(mesh_path: str):
	ext = os.path.splitext(mesh_path)[1].lower()
	before = set(obj.name for obj in bpy.data.objects)

	if ext == ".obj":
		if hasattr(bpy.ops.wm, "obj_import"):
			bpy.ops.wm.obj_import(filepath=mesh_path)
		else:
			bpy.ops.import_scene.obj(filepath=mesh_path)
	elif ext in {".glb", ".gltf"}:
		bpy.ops.import_scene.gltf(filepath=mesh_path)
	elif ext == ".ply":
		bpy.ops.import_mesh.ply(filepath=mesh_path)
	elif ext == ".stl":
		bpy.ops.import_mesh.stl(filepath=mesh_path)
	elif ext == ".fbx":
		bpy.ops.import_scene.fbx(filepath=mesh_path)
	else:
		raise ValueError(f"Unsupported mesh format: {ext}")

	after = set(obj.name for obj in bpy.data.objects)
	new_names = after - before
	imported = [bpy.data.objects[name] for name in new_names]
	if not imported:
		raise RuntimeError(f"No objects imported from: {mesh_path}")
	return imported


def compute_world_aabb(objects):
	if not objects:
		return None
	mins = Vector((float("inf"),) * 3)
	maxs = Vector((float("-inf"),) * 3)
	for obj in objects:
		if obj.type != "MESH":
			continue
		for corner in obj.bound_box:
			co_world = obj.matrix_world @ Vector(corner)
			mins.x = min(mins.x, co_world.x)
			mins.y = min(mins.y, co_world.y)
			mins.z = min(mins.z, co_world.z)
			maxs.x = max(maxs.x, co_world.x)
			maxs.y = max(maxs.y, co_world.y)
			maxs.z = max(maxs.z, co_world.z)
	return mins, maxs


def point_inside_aabb(point: Vector, aabb):
	if aabb is None:
		return False
	mins, maxs = aabb
	return (mins.x <= point.x <= maxs.x) and (mins.y <= point.y <= maxs.y) and (mins.z <= point.z <= maxs.z)


def fg_distance_power_scale(light_pos: Vector, fg_objects) -> float:
	"""Return a scale in (0, 1] that reduces power when the light is close to FG."""
	fg_aabb = compute_world_aabb(fg_objects)
	if fg_aabb is None:
		return 1.0

	mins, maxs = fg_aabb
	center = (mins + maxs) * 0.5
	size = maxs - mins
	max_dim = max(size.x, size.y, size.z, 1e-4)

	d = (light_pos - center).length
	# Reference distance relative to FG size. If d < ref_dist, scale falls quadratically.
	ref_dist = max(0.75 * max_dim, 0.08)
	scale = (d / ref_dist) ** 2
	return float(_clamp(scale, 0.06, 1.0))


def add_point_light_near(fg_objects, bg_objects, energy=2000.0):
	# compute fg center and max dimension
	fg_aabb = compute_world_aabb(fg_objects)
	if fg_aabb is None:
		return None
	fg_mins, fg_maxs = fg_aabb
	fg_center = (fg_mins + fg_maxs) * 0.5
	size = fg_maxs - fg_mins
	max_dim = max(size.x, size.y, size.z, 0.001)

	# initial position: toward camera (-Y) and slightly above
	light_pos = fg_center + Vector((0.0, -0.5 * max_dim, 0.5 * max_dim))

	bg_aabb = compute_world_aabb(bg_objects)

	# if inside bg AABB, move towards camera (negative Y) until outside
	step = 0.25 * max_dim
	attempts = 0
	while point_inside_aabb(light_pos, bg_aabb) and attempts < 10:
		light_pos += Vector((0.0, -step, 0.0))
		attempts += 1

	# fallback: place above fg if still colliding
	if point_inside_aabb(light_pos, bg_aabb):
		light_pos = fg_center + Vector((0.0, 0.0, 1.5 * max_dim))

	light_data = bpy.data.lights.new(name="FG_PointLight", type="POINT")
	light_data.energy = energy
	_configure_point_light(light_data, shadow_soft_size=max_dim * 0.1)
	light_obj = bpy.data.objects.new(name="FG_PointLight", object_data=light_data)
	light_obj.location = light_pos
	_configure_light_object_visibility(light_obj)
	bpy.context.scene.collection.objects.link(light_obj)
	return light_obj


def add_point_light_for_both(fg_objects, bg_objects, energy=1200.0, light_power_scale: float = 1.0):
	# place a light that can illuminate both fg and bg roughly along camera axis (+Y)
	combined = fg_objects + bg_objects
	combined_aabb = compute_world_aabb(combined)
	if combined_aabb is None:
		return add_point_light_near(fg_objects, bg_objects, energy=energy)

	mins, maxs = combined_aabb
	center = (mins + maxs) * 0.5
	size = maxs - mins
	max_dim = max(size.x, size.y, size.z, 0.001)

	# camera at origin facing +Y. Place light between camera and objects (closer to camera)
	half_depth = (maxs.y - mins.y) * 0.5
	light_y = center.y - half_depth * 1.2
	if light_y < 0.1:
		light_y = max(0.1, center.y * 0.5)

	# place slightly above the scene to avoid occlusion
	z_offset = max_dim * 0.8
	light_pos = Vector((center.x, light_y, center.z + z_offset))

	# if this position is inside the background AABB, nudge upward/backwards
	bg_aabb = compute_world_aabb(bg_objects)
	attempts = 0
	while point_inside_aabb(light_pos, bg_aabb) and attempts < 10:
		light_pos += Vector((0.0, -0.5 * max_dim, 0.5 * max_dim))
		attempts += 1

	# final fallback: put above the combined center
	if point_inside_aabb(light_pos, bg_aabb):
		light_pos = center + Vector((0.0, 0.0, 2.0 * max_dim))

	light_data = bpy.data.lights.new(name="FG_BG_PointLight", type="POINT")
	light_data.energy = float(_clamp(energy * max(light_power_scale, 0.0), 0.0, 1e6))
	_configure_point_light(light_data, shadow_soft_size=max_dim * 0.1)

	light_obj = bpy.data.objects.new(name="FG_BG_PointLight", object_data=light_data)
	light_obj.location = light_pos
	_configure_light_object_visibility(light_obj)
	bpy.context.scene.collection.objects.link(light_obj)
	return light_obj


def _clamp(x, a, b):
	return max(a, min(b, x))


def _configure_point_light(light_data, shadow_soft_size: float = None) -> None:
	"""Apply common point-light settings used across all indoor lighting paths."""
	if shadow_soft_size is not None:
		light_data.shadow_soft_size = shadow_soft_size
	try:
		light_data.use_shadow = False
	except Exception:
		pass
	try:
		if hasattr(light_data, "cycles"):
			light_data.cycles.cast_shadow = False
	except Exception:
		pass
	# Remove visible light-source reflection on glass while keeping illumination.
	try:
		light_data.specular_factor = 0.0
	except Exception:
		pass


def _configure_light_object_visibility(light_obj) -> None:
	"""Keep light contribution but hide the emitter from camera/reflection rays."""
	try:
		light_obj.visible_camera = False
	except Exception:
		pass
	try:
		light_obj.visible_glossy = False
	except Exception:
		pass
	try:
		light_obj.visible_transmission = False
	except Exception:
		pass
	try:
		if hasattr(light_obj, "cycles_visibility"):
			light_obj.cycles_visibility.camera = False
			light_obj.cycles_visibility.glossy = False
			light_obj.cycles_visibility.transmission = False
	except Exception:
		pass


def add_sun_light(bg_objects, fg_objects, light_power_scale: float = 1.0):
	"""Add a directional sunlight for outdoor scenes."""
	combined = bg_objects + fg_objects
	combined_aabb = compute_world_aabb(combined)
	if combined_aabb is not None:
		mins, maxs = combined_aabb
		center = (mins + maxs) * 0.5
		max_dim = max((maxs - mins).x, (maxs - mins).y, (maxs - mins).z, 1.0)
	else:
		center = Vector((0.0, 0.0, 0.0))
		max_dim = 1.0

	light_data = bpy.data.lights.new(name="Outdoor_Sun", type="SUN")
	light_data.energy = float(_clamp(4.0 * max(light_power_scale, 0.0), 0.0, 1e5))
	light_data.angle = math.radians(1.0)
	try:
		light_data.use_shadow = False
	except Exception:
		pass
	try:
		if hasattr(light_data, "cycles"):
			light_data.cycles.cast_shadow = False
	except Exception:
		pass

	light_obj = bpy.data.objects.new(name="Outdoor_Sun", object_data=light_data)
	light_obj.location = center + Vector((0.0, -2.0 * max_dim, 3.0 * max_dim))
	# Downward and slightly angled sunlight direction.
	light_obj.rotation_euler = (math.radians(55.0), 0.0, math.radians(35.0))
	_configure_light_object_visibility(light_obj)
	bpy.context.scene.collection.objects.link(light_obj)
	return light_obj


def _image_mean_rgb_and_luma(base_image_path: str, border_only: bool = False):
	if not base_image_path or not os.path.exists(base_image_path):
		return None
	img = bpy.data.images.load(os.path.abspath(base_image_path))
	if img.size[0] <= 0 or img.size[1] <= 0:
		return None
	w = int(img.size[0])
	h = int(img.size[1])
	pixels = list(img.pixels[:])
	if not pixels:
		return None

	r_sum = g_sum = b_sum = 0.0
	count = 0

	def add_pixel(ix, iy):
		nonlocal r_sum, g_sum, b_sum, count
		idx = (iy * w + ix) * 4
		r_sum += pixels[idx]
		g_sum += pixels[idx + 1]
		b_sum += pixels[idx + 2]
		count += 1

	if border_only:
		# Use only a thin border so "empty" render regions match image outskirts.
		b = max(1, int(min(w, h) * 0.08))
		for y in range(h):
			for x in range(w):
				if x < b or x >= w - b or y < b or y >= h - b:
					add_pixel(x, y)
	else:
		for y in range(h):
			for x in range(w):
				add_pixel(x, y)

	if count == 0:
		return None

	r_mean = r_sum / count
	g_mean = g_sum / count
	b_mean = b_sum / count
	lum = 0.2126 * r_mean + 0.7152 * g_mean + 0.0722 * b_mean
	return Vector((r_mean, g_mean, b_mean)), float(max(lum, 1e-4))


def _brightest_patch_from_base_image(base_image_path: str):
	"""Return brightest region center (u,v in [0,1]), mean tone, and luminance from base image."""
	try:
		from PIL import Image
		import numpy as np
	except Exception:
		return None

	if not base_image_path or not os.path.exists(base_image_path):
		return None

	try:
		img = Image.open(base_image_path).convert("RGB")
		arr = np.asarray(img).astype(np.float32) / 255.0  # H, W, 3 (top-left origin)
	except Exception:
		return None

	h, w, _ = arr.shape
	if h <= 0 or w <= 0:
		return None

	lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

	# Prefer a bright cluster instead of a single noisy pixel.
	thr = float(np.percentile(lum, 99.7))
	mask = lum >= thr
	if np.count_nonzero(mask) < 8:
		y, x = np.unravel_index(np.argmax(lum), lum.shape)
	else:
		ys, xs = np.where(mask)
		weights = lum[ys, xs] + 1e-6
		x = int(np.clip(np.round(np.average(xs, weights=weights)), 0, w - 1))
		y = int(np.clip(np.round(np.average(ys, weights=weights)), 0, h - 1))

	rad = max(2, int(min(h, w) * 0.03))
	x0, x1 = max(0, x - rad), min(w, x + rad + 1)
	y0, y1 = max(0, y - rad), min(h, y + rad + 1)
	patch = arr[y0:y1, x0:x1, :]
	if patch.size == 0:
		return None

	tone = patch.reshape(-1, 3).mean(axis=0)
	patch_lum = float((0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]).mean())

	# PIL uses top-left y; convert to normalized image coordinates then to bottom-left v.
	u = (x + 0.5) / float(w)
	v = 1.0 - (y + 0.5) / float(h)

	return float(u), float(v), Vector((float(tone[0]), float(tone[1]), float(tone[2]))), float(max(patch_lum, 1e-4))


def _camera_ray_world(cam_obj, camera_info: dict, u: float, v: float):
	fov_x = math.radians(float(camera_info["fov_x"]))
	fov_y = math.radians(float(camera_info["fov_y"]))
	intr = camera_info.get("intrinsics")
	cx = float(intr[0][2]) if intr and len(intr) >= 1 and len(intr[0]) >= 3 else 0.5
	cy = float(intr[1][2]) if intr and len(intr) >= 2 and len(intr[1]) >= 3 else 0.5

	x_cam = 2.0 * (float(u) - cx) * math.tan(fov_x / 2.0)
	y_cam = 2.0 * (float(v) - cy) * math.tan(fov_y / 2.0)
	dir_cam = Vector((x_cam, y_cam, -1.0)).normalized()
	dir_world = (cam_obj.matrix_world.to_quaternion() @ dir_cam).normalized()
	return cam_obj.location.copy(), dir_world


def _refract_direction(incident: Vector, normal: Vector, n1: float, n2: float):
	"""Return refracted direction using Snell's law, or None on total internal reflection."""
	i = incident.normalized()
	n = normal.normalized()
	eta = float(n1) / float(max(n2, 1e-8))
	cos_i = -float(max(-1.0, min(1.0, i.dot(n))))
	k = 1.0 - (eta * eta) * (1.0 - cos_i * cos_i)
	if k < 0.0:
		return None
	t = eta * i + (eta * cos_i - math.sqrt(k)) * n
	return t.normalized()


def _fill_small_holes(mask_bool, region_bool, iterations: int = 2, min_neighbors: int = 6):
	"""Fill tiny black holes inside a foreground region using neighborhood voting."""
	try:
		import numpy as np
	except Exception:
		return mask_bool

	filled = mask_bool.copy()
	region = region_bool.astype(bool)
	h, w = filled.shape

	for _ in range(max(1, int(iterations))):
		p = np.pad(filled.astype(np.uint8), ((1, 1), (1, 1)), mode="constant", constant_values=0)
		nbr = (
			p[0:h, 0:w] + p[0:h, 1:w + 1] + p[0:h, 2:w + 2] +
			p[1:h + 1, 0:w] + p[1:h + 1, 2:w + 2] +
			p[2:h + 2, 0:w] + p[2:h + 2, 1:w + 1] + p[2:h + 2, 2:w + 2]
		)
		# Only fill inside FG region; this avoids bleeding into the background.
		to_fill = region & (~filled) & (nbr >= int(min_neighbors))
		if not to_fill.any():
			break
		filled[to_fill] = True

	return filled


def save_fg_bg_hit_mask(
	out_mask_path: str,
	camera_info: dict,
	cam_obj,
	fg_objects,
	bg_objects,
	width: int,
	height: int,
):
	"""Save mask: white where first hit is FG and refracted path eventually hits BG, else black."""
	if not out_mask_path:
		return

	try:
		from PIL import Image
		import numpy as np
	except Exception:
		print("[render_gt] PIL/numpy unavailable, skipping --out_mask")
		return

	fg_names = {o.name for o in fg_objects if o.type == "MESH"}
	bg_names = {o.name for o in bg_objects if o.type == "MESH"}
	if not fg_names or not bg_names:
		print("[render_gt] Missing FG/BG mesh objects, skipping --out_mask")
		return

	depsgraph = bpy.context.evaluated_depsgraph_get()
	scene = bpy.context.scene
	mask = np.zeros((int(height), int(width)), dtype=np.uint8)
	fg_sil = np.zeros((int(height), int(width)), dtype=bool)

	eps = 1e-4
	max_follow_hits = 16
	glass_ior = 1.5

	for y in range(int(height)):
		v = 1.0 - ((float(y) + 0.5) / float(height))
		for x in range(int(width)):
			u = (float(x) + 0.5) / float(width)
			origin, direction = _camera_ray_world(cam_obj, camera_info, u, v)
			hit, loc, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
			if not hit or obj is None or obj.name not in fg_names:
				continue
			fg_sil[y, x] = True

			# Ray starts in FG silhouette; trace through FG with refraction.
			curr_origin = origin
			curr_dir = direction.normalized()
			curr_ior = 1.0
			for _ in range(max_follow_hits):
				hit2, loc2, normal2, _, obj2, _ = scene.ray_cast(depsgraph, curr_origin, curr_dir)
				if not hit2 or obj2 is None:
					break
				if obj2.name in bg_names:
					mask[y, x] = 255
					break
				if obj2.name not in fg_names:
					break

				n = normal2.normalized() if normal2.length > 1e-8 else Vector((0.0, 0.0, 1.0))
				# If ray is leaving FG, flip normal to oppose incident direction.
				entering = curr_dir.dot(n) < 0.0
				if entering:
					face_n = n
					n1 = curr_ior
					n2 = glass_ior
				else:
					face_n = -n
					n1 = curr_ior
					n2 = 1.0

				refr_dir = _refract_direction(curr_dir, face_n, n1, n2)
				if refr_dir is None:
					# Total internal reflection fallback.
					refl_dir = curr_dir - 2.0 * curr_dir.dot(face_n) * face_n
					curr_dir = refl_dir.normalized()
				else:
					curr_dir = refr_dir
					curr_ior = n2

				curr_origin = loc2 + curr_dir * eps

	# Fill tiny black pinholes caused by per-pixel tracing misses.
	mask_bool = mask > 0
	mask_bool = _fill_small_holes(mask_bool, fg_sil, iterations=2, min_neighbors=6)
	mask = (mask_bool.astype(np.uint8) * 255)

	out_abs = os.path.abspath(out_mask_path)
	os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
	Image.fromarray(mask, mode="L").save(out_abs)
	print(f"[render_gt] Saved FG->BG mask to: {out_abs}")


def add_point_light_from_brightest_base(
	base_image_path: str,
	camera_info: dict,
	cam_obj,
	bg_objects,
	fg_objects,
	light_power_scale: float = 1.0,
):
	bright = _brightest_patch_from_base_image(base_image_path)
	if bright is None:
		return None
	u, v, tone, lum = bright

	bg_names = {o.name for o in bg_objects if o.type == "MESH"}
	depsgraph = bpy.context.evaluated_depsgraph_get()
	scene = bpy.context.scene

	# Search around the brightest pixel if the exact pixel does not hit bg geometry.
	off = [
		(0.0, 0.0),
		(0.015, 0.0), (-0.015, 0.0), (0.0, 0.015), (0.0, -0.015),
		(0.03, 0.0), (-0.03, 0.0), (0.0, 0.03), (0.0, -0.03),
	]

	hit_loc = None
	hit_nrm = None
	for du, dv in off:
		tu = _clamp(u + du, 0.001, 0.999)
		tv = _clamp(v + dv, 0.001, 0.999)
		origin, direction = _camera_ray_world(cam_obj, camera_info, tu, tv)
		hit, loc, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
		if hit and obj is not None and obj.name in bg_names:
			hit_loc = loc
			hit_nrm = normal.normalized() if normal.length > 1e-6 else Vector((0.0, 0.0, 1.0))
			break

	if hit_loc is None:
		return None

	all_aabb = compute_world_aabb(bg_objects + fg_objects)
	if all_aabb is not None:
		mins, maxs = all_aabb
		sz = maxs - mins
		max_dim = max(sz.x, sz.y, sz.z, 0.001)
	else:
		max_dim = 1.0

	cam_dir = (hit_loc - cam_obj.location).normalized()
	side = hit_nrm.cross(cam_dir)
	if side.length < 1e-6:
		side = Vector((1.0, 0.0, 0.0))
	else:
		side.normalize()

	offset = max(0.05, 0.12 * max_dim)
	light_pos = hit_loc + hit_nrm * (0.9 * offset) + side * (0.5 * offset)

	bg_aabb = compute_world_aabb(bg_objects)
	for _ in range(8):
		if not point_inside_aabb(light_pos, bg_aabb):
			break
		light_pos += hit_nrm * (0.6 * offset)

	light_data = bpy.data.lights.new(name="BaseImage_PointLight", type="POINT")
	_configure_point_light(light_data, shadow_soft_size=max(0.02, 0.08 * max_dim))
	# Brightest region usually indicates source: use its luminance/tone to set light.
	# Use conservative energy scaling and reduce power further when light is close to FG.
	base_energy = float(_clamp(250.0 + 3200.0 * lum, 120.0, 4200.0))
	dist_scale = fg_distance_power_scale(light_pos, fg_objects)
	light_data.energy = float(_clamp(base_energy * dist_scale * max(light_power_scale, 0.0), 20.0, 4200.0))
	tone_soft = tone.lerp(Vector((1.0, 1.0, 1.0)), 0.25)
	light_data.color = (tone_soft.x, tone_soft.y, tone_soft.z)

	light_obj = bpy.data.objects.new(name="BaseImage_PointLight", object_data=light_data)
	light_obj.location = light_pos
	_configure_light_object_visibility(light_obj)
	bpy.context.scene.collection.objects.link(light_obj)
	return light_obj


def adjust_lighting_from_image(base_image_path: str, light_obj, target_luminance: float = 0.5, adjust_light: bool = True):
	try:
		if not base_image_path or not os.path.exists(base_image_path):
			return
		full_stats = _image_mean_rgb_and_luma(base_image_path, border_only=False)
		border_stats = _image_mean_rgb_and_luma(base_image_path, border_only=True)
		if full_stats is None:
			return
		full_tone, lum = full_stats
		if border_stats is not None:
			border_tone, border_lum = border_stats
		else:
			border_tone, border_lum = full_tone, lum

		# perceived luminance
		lum = max(lum, 1e-4)

		# energy scale to map image luminance to reasonable light energy
		# higher desired target_luminance -> more energy; clamp conservatively
		energy_scale = float(target_luminance) / float(lum)
		energy_scale = _clamp(energy_scale, 0.2, 2.0)

		# tone: blend mean color with white to avoid extreme tinting
		tone = full_tone.lerp(Vector((1.0, 1.0, 1.0)), 0.45)
		bg_tone = border_tone.lerp(Vector((1.0, 1.0, 1.0)), 0.25)

		# apply to light
		if adjust_light and light_obj is not None and light_obj.type == 'LIGHT':
			try:
				light_obj.data.color = (tone.x, tone.y, tone.z)
				light_obj.data.energy = float(_clamp(light_obj.data.energy * energy_scale, 0.1, 1e6))
			except Exception:
				pass

		# also adjust world background to match overall tone and brightness (low-strength)
		try:
			world = bpy.context.scene.world
			if world is None:
				world = bpy.data.worlds.new("World")
				bpy.context.scene.world = world
			world.use_nodes = True
			bg = world.node_tree.nodes.get("Background")
			if bg:
				bg.inputs[0].default_value[0] = bg_tone.x
				bg.inputs[0].default_value[1] = bg_tone.y
				bg.inputs[0].default_value[2] = bg_tone.z
				# set strength from border brightness, not full-frame brightness; cap lower
				bg.inputs[1].default_value = _clamp(0.6 * border_lum, 0.02, 0.6)

			# conservative exposure adjustment to help match overall brightness
			try:
				scene = bpy.context.scene
				if hasattr(scene, "view_settings"):
					exp = math.log2(float(target_luminance) / float(lum))
					scene.view_settings.exposure = _clamp(exp, -2.0, 1.0)
			except Exception:
				pass
		except Exception:
			pass
	except Exception:
		return


def disable_all_shadows():
	# Disable object shadow casting (Cycles) and light shadows where possible
	for obj in bpy.context.scene.objects:
		if obj.type == "MESH":
			try:
				obj.cycles_visibility.shadow = False
			except Exception:
				pass
			try:
				obj.show_shadow = False
			except Exception:
				pass

	for obj in bpy.context.scene.objects:
		if obj.type == "LIGHT":
			ld = obj.data
			try:
				ld.use_shadow = False
			except Exception:
				pass
			try:
				if hasattr(ld, "cycles"):
					ld.cycles.cast_shadow = False
			except Exception:
				pass


def create_glass_material(ior: float = 1.5):
	mat = bpy.data.materials.new(name="FG_Glass")
	mat.use_nodes = True
	nodes = mat.node_tree.nodes
	links = mat.node_tree.links
	nodes.clear()

	out = nodes.new(type="ShaderNodeOutputMaterial")
	# Camera rays see refraction-only (no first-surface reflection), while
	# secondary rays use glass so internal reflection/TIR can still appear.
	lp = nodes.new(type="ShaderNodeLightPath")
	mix = nodes.new(type="ShaderNodeMixShader")
	refract = nodes.new(type="ShaderNodeBsdfRefraction")
	glass = nodes.new(type="ShaderNodeBsdfGlass")

	refract.inputs["IOR"].default_value = ior
	refract.inputs["Roughness"].default_value = 0.02
	refract.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

	glass.inputs["IOR"].default_value = ior
	glass.inputs["Roughness"].default_value = 0.02
	glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

	# Mix factor: 1 for camera rays -> Refraction, 0 otherwise -> Glass.
	links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
	links.new(glass.outputs["BSDF"], mix.inputs[1])
	links.new(refract.outputs["BSDF"], mix.inputs[2])
	links.new(mix.outputs["Shader"], out.inputs["Surface"])
	return mat


def apply_material_to_mesh_objects(objects, material) -> None:
	for obj in objects:
		if obj.type != "MESH":
			continue
		if obj.data.materials:
			obj.data.materials[0] = material
		else:
			obj.data.materials.append(material)


def setup_camera(camera_info: dict, width: int, height: int) -> None:
	cam_data = bpy.data.cameras.new(name="RenderCamera")
	cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
	bpy.context.scene.collection.objects.link(cam_obj)
	bpy.context.scene.camera = cam_obj

	# Blender cameras look down local -Z. Rotating +90 deg around X makes principal axis face +Y.
	cam_obj.location = (0.0, 0.0, 0.0)
	cam_obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)

	fov_x = math.radians(float(camera_info["fov_x"]))
	fov_y = math.radians(float(camera_info["fov_y"]))

	focal_mm = 50.0
	cam_data.lens = focal_mm
	cam_data.sensor_width = 2.0 * focal_mm * math.tan(fov_x / 2.0)
	cam_data.sensor_height = 2.0 * focal_mm * math.tan(fov_y / 2.0)
	cam_data.sensor_fit = "AUTO"

	intr = camera_info.get("intrinsics")
	if intr and len(intr) >= 2 and len(intr[0]) >= 3 and len(intr[1]) >= 3:
		cx = float(intr[0][2])
		cy = float(intr[1][2])
		cam_data.shift_x = 0.5 - cx
		cam_data.shift_y = cy - 0.5

	scene = bpy.context.scene
	scene.render.resolution_x = int(width)
	scene.render.resolution_y = int(height)
	scene.render.resolution_percentage = 100


def setup_render(output_path: str, samples: int) -> None:
	scene = bpy.context.scene
	scene.render.engine = "CYCLES"
	scene.cycles.samples = int(samples)
	scene.cycles.use_adaptive_sampling = True

	# Suppress glass fireflies/sparkles from high-energy paths.
	try:
		scene.cycles.sample_clamp_indirect = 1.5
	except Exception:
		pass
	try:
		scene.cycles.sample_clamp_direct = 6.0
	except Exception:
		pass
	try:
		scene.cycles.filter_glossy = 1.0
	except Exception:
		pass
	# Keep reflective paths for glass/TIR while still limiting noise.
	try:
		scene.cycles.glossy_bounces = 8
	except Exception:
		pass
	try:
		scene.cycles.caustics_reflective = False
	except Exception:
		pass
	try:
		scene.cycles.caustics_refractive = False
	except Exception:
		pass

	scene.render.image_settings.file_format = "JPEG"
	scene.render.image_settings.quality = 95
	scene.render.image_settings.color_mode = "RGB"
	scene.render.filepath = output_path

	# Ensure denoising is disabled when OpenImageDenoise is not available
	try:
		if hasattr(scene.cycles, "use_denoising"):
			scene.cycles.use_denoising = False
	except Exception:
		pass

	for vl in scene.view_layers:
		try:
			if hasattr(vl.cycles, "use_denoising"):
				vl.cycles.use_denoising = False
		except Exception:
			continue


def setup_glossy_debug_outputs(debug_dir: str) -> None:
	"""Enable glossy passes and save them as EXR files via compositor file output."""
	if not debug_dir:
		return

	debug_abs = os.path.abspath(debug_dir)
	os.makedirs(debug_abs, exist_ok=True)

	scene = bpy.context.scene
	view_layer = bpy.context.view_layer

	# Enable the reflection-related passes.
	for attr in ("use_pass_glossy_direct", "use_pass_glossy_indirect", "use_pass_glossy_color"):
		try:
			setattr(view_layer, attr, True)
		except Exception:
			pass

	scene.use_nodes = True
	nodes = scene.node_tree.nodes
	links = scene.node_tree.links

	rl = nodes.new(type="CompositorNodeRLayers")
	rl.location = (-450, 0)

	file_out = nodes.new(type="CompositorNodeOutputFile")
	file_out.location = (100, 0)
	file_out.base_path = debug_abs
	file_out.format.file_format = "OPEN_EXR"
	file_out.format.color_mode = "RGB"
	file_out.format.color_depth = "16"

	# Remove default slot and add named debug slots.
	try:
		file_out.file_slots.remove(file_out.file_slots[0])
	except Exception:
		pass

	# Blender socket names can vary by version; map robustly.
	pass_map = [
		(["GlossDir", "Glossy Direct"], "glossy_direct_"),
		(["GlossInd", "Glossy Indirect"], "glossy_indirect_"),
		(["GlossCol", "Glossy Color"], "glossy_color_"),
	]

	for socket_candidates, slot_name in pass_map:
		src_socket = None
		for socket_name in socket_candidates:
			src_socket = rl.outputs.get(socket_name)
			if src_socket is not None:
				break
		if src_socket is None:
			continue
		file_out.file_slots.new(slot_name)
		links.new(src_socket, file_out.inputs[-1])

	print(f"[render_gt] Debug glossy outputs enabled: {debug_abs}")


def ensure_world_light() -> None:
	world = bpy.context.scene.world
	if world is None:
		world = bpy.data.worlds.new("World")
		bpy.context.scene.world = world
	world.use_nodes = True
	bg = world.node_tree.nodes.get("Background")
	if bg:
		# conservative default; can be overridden by base-image adjustment.
		bg.inputs["Strength"].default_value = 0.08
	# Keep transmission and glossy visibility so reflective/TIR paths are not forced black.
	try:
		world.cycles_visibility.glossy = True
	except Exception:
		pass
	try:
		world.cycles_visibility.transmission = True
	except Exception:
		pass


# Outdoor mode removed: indoor (point-light) approach is the only supported mode now.


def match_render_to_base(render_path: str, base_image_path: str, strength: float = 0.9) -> None:
	"""Match rendered colors to base image in the mesh region using mean/std transfer."""
	try:
		from PIL import Image
		import numpy as np
	except Exception:
		print("[render_gt] PIL/numpy unavailable, skipping --match_to_base")
		return

	if not os.path.exists(render_path) or not os.path.exists(base_image_path):
		return

	strength = float(_clamp(strength, 0.0, 1.0))

	try:
		render_img = Image.open(render_path).convert("RGB")
		base_img = Image.open(base_image_path).convert("RGB").resize(render_img.size, Image.Resampling.BILINEAR)
	except Exception:
		return

	r = np.asarray(render_img).astype(np.float32) / 255.0
	b = np.asarray(base_img).astype(np.float32) / 255.0

	# Build a soft foreground mask from distance to border average color in render.
	h, w, _ = r.shape
	band = max(1, int(min(h, w) * 0.08))
	border_pixels = np.concatenate(
		[
			r[:band, :, :].reshape(-1, 3),
			r[h - band :, :, :].reshape(-1, 3),
			r[:, :band, :].reshape(-1, 3),
			r[:, w - band :, :].reshape(-1, 3),
		],
		axis=0,
	)
	bg_color = border_pixels.mean(axis=0, keepdims=True)
	delta = np.linalg.norm(r.reshape(-1, 3) - bg_color, axis=1).reshape(h, w)

	# Mesh-heavy pixels tend to be farther from empty background color.
	thresh = float(np.percentile(delta, 60.0))
	mask = delta > thresh
	if mask.sum() < 128:
		mask = np.ones((h, w), dtype=bool)

	r_mask = r[mask]
	b_mask = b[mask]

	# Per-channel affine color transfer from render->base stats on the same region.
	r_mean = r_mask.mean(axis=0)
	r_std = r_mask.std(axis=0) + 1e-6
	b_mean = b_mask.mean(axis=0)
	b_std = b_mask.std(axis=0) + 1e-6

	matched = (r - r_mean) * (b_std / r_std) + b_mean
	matched = np.clip(matched, 0.0, 1.0)

	# Apply strongly on mesh region and softly pull no-mesh background toward base image.
	alpha = np.zeros((h, w, 1), dtype=np.float32)
	alpha[mask] = strength
	out = r * (1.0 - alpha) + matched * alpha

	# For empty/background regions, blend with base image tone to avoid pure white/flat background.
	bg_blend = 0.35
	bg_alpha = np.zeros((h, w, 1), dtype=np.float32)
	bg_alpha[~mask] = bg_blend
	out = out * (1.0 - bg_alpha) + b * bg_alpha
	out = np.clip(out, 0.0, 1.0)

	try:
		Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB").save(render_path, quality=95)
		print(f"[render_gt] Applied base-image region color matching: {render_path}")
	except Exception:
		return


def main() -> None:
	args = parse_args()

	clear_scene()

	base_dir = os.path.abspath(args.base_dir)
	mesh_bg_path = os.path.join(base_dir, "mesh_bg.glb")
	mesh_fg_path = os.path.join(base_dir, f"mesh_fg_{args.obj}.glb")
	camera_json_path = os.path.join(base_dir, "camera.json")

	if not os.path.exists(mesh_bg_path):
		raise FileNotFoundError(f"Background mesh not found: {mesh_bg_path}")
	if not os.path.exists(mesh_fg_path):
		raise FileNotFoundError(f"Foreground mesh not found: {mesh_fg_path}")
	if not os.path.exists(camera_json_path):
		raise FileNotFoundError(f"Camera JSON not found: {camera_json_path}")

	bg_objects = import_mesh(mesh_bg_path)
	fg_objects = import_mesh(mesh_fg_path)

	fg_glass = create_glass_material(ior=1.5)
	apply_material_to_mesh_objects(fg_objects, fg_glass)

	with open(camera_json_path, "r", encoding="utf-8") as f:
		camera_info = json.load(f)

	setup_camera(camera_info, args.width, args.height)
	cam_obj = bpy.context.scene.camera

	# note: empty-mask application happens after render when requested

	if args.out_mask:
		try:
			save_fg_bg_hit_mask(
				out_mask_path=args.out_mask,
				camera_info=camera_info,
				cam_obj=cam_obj,
				fg_objects=fg_objects,
				bg_objects=bg_objects,
				width=args.width,
				height=args.height,
			)
		except Exception:
			pass

	# ensure foreground objects don't cast shadows
	for obj in fg_objects:
		try:
			obj.cycles_visibility.shadow = False
		except Exception:
			pass

	ensure_world_light()

	# Indoor mode uses point lights; outdoor mode uses sunlight.
	try:
		base_image_path = args.base_image or os.path.join(base_dir, "image.jpg")
		light = None
		used_brightest = False

		if args.scene_mode == "outdoor":
			light = add_sun_light(bg_objects, fg_objects, light_power_scale=args.light_power_scale)
		else:
			if os.path.exists(base_image_path):
				light = add_point_light_from_brightest_base(
					base_image_path,
					camera_info,
					cam_obj,
					bg_objects,
					fg_objects,
					light_power_scale=args.light_power_scale,
				)
				used_brightest = light is not None
			if light is None:
				# use lower-energy default fallback to avoid overbright scenes
				light = add_point_light_for_both(fg_objects, bg_objects, light_power_scale=args.light_power_scale)
		# ensure light doesn't cast shadows
		if light is not None and light.type == 'LIGHT':
			try:
				light.data.use_shadow = False
			except Exception:
				pass
			try:
				if hasattr(light.data, 'cycles'):
					light.data.cycles.cast_shadow = False
			except Exception:
				pass
		# if a base image provided, adjust light and world to match its tone/brightness
		if os.path.exists(base_image_path):
			try:
				# Keep world/background tone matching active; avoid overriding outdoor sun energy.
				adjust_lighting_from_image(
					base_image_path,
					light,
					adjust_light=(args.scene_mode == "indoor" and (not used_brightest)),
				)
			except Exception:
				pass
	except Exception:
		pass

	output_path = os.path.abspath(args.output)
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
	setup_render(output_path, args.samples)
	if args.debug_glossy_dir:
		try:
			setup_glossy_debug_outputs(args.debug_glossy_dir)
		except Exception:
			print("[render_gt] Failed to configure glossy debug outputs")

	# final defensive call to disable any shadows before render
	try:
		disable_all_shadows()
	except Exception:
		pass

	if args.save_blend:
		try:
			save_blend_path = os.path.abspath(args.save_blend)
			os.makedirs(os.path.dirname(save_blend_path) or ".", exist_ok=True)
			bpy.ops.wm.save_as_mainfile(filepath=save_blend_path)
			print(f"[render_gt] Saved debug blend to: {save_blend_path}")
		except Exception:
			print("[render_gt] Failed to save debug blend")

	bpy.ops.render.render(write_still=True)

	base_image_for_match = args.base_image or os.path.join(base_dir, "image.jpg")
	if args.match_to_base and os.path.exists(base_image_for_match):
		try:
			match_render_to_base(output_path, base_image_for_match, strength=args.match_strength)
		except Exception:
			pass

	# If an FG->BG mask was requested, apply it to the final saved render:
	# For pixels inside the FG silhouette where the mask value == 0, make them black.
	if args.out_mask:
		out_mask_abs = os.path.abspath(args.out_mask)
	else:
		out_mask_abs = None

	if out_mask_abs and os.path.exists(out_mask_abs):
		try:
			from PIL import Image
			import numpy as np
		except Exception:
			print("[render_gt] PIL/numpy unavailable, skipping applying mask to render")
		else:
			try:
				# Load rendered image and mask (nearest resampling to preserve pixel mask values)
				render_img = Image.open(output_path).convert("RGB")
				mask_img = Image.open(out_mask_abs).convert("L").resize(render_img.size, Image.Resampling.NEAREST)
				r = np.asarray(render_img).astype(np.uint8)
				m = np.asarray(mask_img).astype(np.uint8)
				h, w = m.shape
				# Compute FG silhouette by ray-casting first-hit per pixel.
				depsgraph = bpy.context.evaluated_depsgraph_get()
				scene = bpy.context.scene
				fg_names = {o.name for o in fg_objects if o.type == "MESH"}
				fg_sil = np.zeros((h, w), dtype=bool)
				for yy in range(h):
					v = 1.0 - ((float(yy) + 0.5) / float(h))
					for xx in range(w):
						u = (float(xx) + 0.5) / float(w)
						origin, direction = _camera_ray_world(cam_obj, camera_info, u, v)
						hit, loc, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
						if hit and obj is not None and obj.name in fg_names:
							fg_sil[yy, xx] = True

				# Binarize mask robustly (important when mask is saved as JPG).
				mask_white = (m >= 128)
				mask_white = _fill_small_holes(mask_white, fg_sil, iterations=2, min_neighbors=6)
				# Where pixel is in FG silhouette and mask==0 -> blacken.
				apply_mask = fg_sil & (~mask_white)
				if apply_mask.any():
					r[apply_mask] = [0, 0, 0]
				# Save back
				Image.fromarray(r).save(output_path, quality=95)
			except Exception:
				print("[render_gt] Error while applying mask to render, skipping")

		# If requested, apply empty-background mask directly (blacken pixels where camera ray hits nothing)
		if getattr(args, "apply_empty_mask", False):
			try:
				from PIL import Image
				import numpy as np
			except Exception:
				print("[render_gt] PIL/numpy unavailable, skipping --apply_empty_mask")
			else:
				try:
					render_img = Image.open(output_path).convert("RGB")
					r = np.asarray(render_img).astype(np.uint8)
					h, w, _ = r.shape
					depsgraph = bpy.context.evaluated_depsgraph_get()
					scene = bpy.context.scene
					for yy in range(h):
						v = 1.0 - ((float(yy) + 0.5) / float(h))
						for xx in range(w):
							u = (float(xx) + 0.5) / float(w)
							origin, direction = _camera_ray_world(cam_obj, camera_info, u, v)
							hit, loc, normal, _, obj, _ = scene.ray_cast(depsgraph, origin, direction)
							if not hit or obj is None:
								r[yy, xx] = [0, 0, 0]
					# Save back
					Image.fromarray(r).save(output_path, quality=95)
				except Exception:
					print("[render_gt] Error while applying empty mask to render, skipping")

	try:
		save_auxiliary_files(base_dir=base_dir, obj_name=args.obj, out_dir=os.path.dirname(output_path) or ".")
	except Exception:
		print("[render_gt] Failed to save auxiliary files to output directory")

	print(f"Saved render to: {output_path}")


if __name__ == "__main__":
	main()

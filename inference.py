import argparse
import gc
import shutil
import subprocess
import sys
import os
from pathlib import Path

import torch

from tqdm import tqdm

from utils.add_shadows import add_shadows


def run_cmd(cmd: list[str], cwd: Path) -> None:
	"""Run a subprocess command and fail fast with a clear message."""
	print("\n$ " + " ".join(cmd))
	subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
	parser = argparse.ArgumentParser(description="Run Snellcaster preprocessing pipeline.")
	parser.add_argument("prompt", type=str, help="Input scene prompt")
	parser.add_argument("--height", type=int, default=720, help="Image height (default: 720)")
	parser.add_argument("--width", type=int, default=1280, help="Image width (default: 1280)")
	parser.add_argument("--seed", type=int, default=42, help="Seed for base image generation (default: 42)")
	parser.add_argument("--out_dir", type=str, default="./results", help="Output directory (default: ./results)")
	parser.add_argument("--scene_name", type=str, default="scene", help="Scene name used in generated filenames (default: scene)")
	parser.add_argument("--alpha", type=float, default=0.5, help="Dual-view alpha blending parameter (default: 0.5)")
	parser.add_argument("--levels", type=int, default=5, help="Dual-view pyramid levels (default: 5)")
	parser.add_argument("--time_travel_repeats", type=int, default=3, help="Dual-view time-travel repeats (default: 3)")
	parser.add_argument("--blend_step_ratio", type=float, default=1.0, help="Dual-view blend step ratio (default: 1.0)")
	parser.add_argument("--num_steps", type=int, default=20, help="Dual-view denoising steps (default: 20)")
	parser.add_argument("--main_guidance_scale", type=float, default=3.5, help="Dual-view main guidance scale (default: 3.5)")
	parser.add_argument("--pano_guidance_scale", type=float, default=3.5, help="Dual-view pano guidance scale (default: 3.5)")
	parser.add_argument("--pano_seed", type=int, default=42, help="Seed for panorama generation in dual view (default: 42)")
	parser.add_argument("--num_shadow_variations", type=int, default=3, help="Number of shadow variations to generate (default: 3)")
	parser.add_argument("--obj_mesh", type=str, default=None, help="Path to an existing foreground object mesh (.glb). If provided, skips step 4 (text-to-3D generation) and uses this mesh directly.")
	parser.add_argument("--save_intermediate", action="store_true", help="Save per-step Tweedie estimates and final grid during dual-view generation")
	args = parser.parse_args()
	prompt = args.prompt
	height = args.height
	width = args.width
	seed = args.seed
	out_dir = args.out_dir
	scene_name = args.scene_name
	alpha = args.alpha
	levels = args.levels
	time_travel_repeats = args.time_travel_repeats
	blend_step_ratio = args.blend_step_ratio
	num_steps = args.num_steps
	main_guidance_scale = args.main_guidance_scale
	pano_guidance_scale = args.pano_guidance_scale
	pano_seed = args.pano_seed
	num_shadow_variations = args.num_shadow_variations
	mesh_fg_path = Path(args.obj_mesh) if args.obj_mesh is not None else None
	script_dir = Path(__file__).resolve().parent
	out_dir = Path(args.out_dir)
	if not out_dir.is_absolute():
		out_dir = (script_dir / out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	scene_dir = out_dir / f"{scene_name}_{seed}"
	intermediates_dir = scene_dir / "intermediates"
	scene_dir.mkdir(parents=True, exist_ok=True)
	intermediates_dir.mkdir(parents=True, exist_ok=True)

	# Step 1: Parse prompt into the required prompts.
	print("\n[Step 1] Prompt parsing...")
	prompts_file = intermediates_dir / "prompts.txt"
	required_keys = {"p", "p_obj", "p_minus", "p_surface", "p_pano", "p_ior"}

	def _load_prompts(path: Path) -> dict:
		saved = {}
		for line in path.read_text().splitlines():
			if "=" in line:
				key, val = line.split("=", 1)
				saved[key.strip()] = val.strip()
		return saved

	saved = _load_prompts(prompts_file) if prompts_file.exists() else {}
	if not (required_keys <= saved.keys()):
		if saved:
			print(f"Cached prompts missing keys {required_keys - saved.keys()}, re-parsing...")
		run_cmd(
			[
				sys.executable,
				"-m",
				"utils.text_parsing",
				prompt,
				"--seed", str(seed),
				"--out", str(prompts_file),
			],
			cwd=script_dir,
		)
		saved = _load_prompts(prompts_file)
	else:
		print(f"Loading cached prompts from {prompts_file}")

	p, p_obj, p_minus, p_surface, p_pano, p_ior = (
		saved["p"], saved["p_obj"], saved["p_minus"],
		saved["p_surface"], saved["p_pano"], saved["p_ior"],
	)
	print(f"p: {p}")
	print(f"p_obj: {p_obj}")
	print(f"p_minus: {p_minus}")
	print(f"p_surface: {p_surface}")
	print(f"p_pano: {p_pano}")
	print(f"p_ior: {p_ior}")


	# Step 2: Generate base image from p_minus.
	image_path = scene_dir / "base_image.jpg"
	if not image_path.exists():
		print("\n[Step 2] Generating base image with FluxPipeline...")
		run_cmd(
			[
				sys.executable,
				"generate_base.py",
				p_minus,
				"--width", str(width),
				"--height", str(height),
				"--seed", str(seed),
				"--save_path", str(image_path),
			],
			cwd=script_dir,
		)
	else:
		print(f"Base image already exists at {image_path}, skipping generation.")


	# Step 3: Run MoGe2 with maps+glb and threshold=0.1.
	print("\n[Step 3] Running MoGe2 inference...")
	mesh_bg_path = intermediates_dir / "mesh.glb"
	camera_json_path = intermediates_dir / "camera.json"
	if not mesh_bg_path.exists() or not camera_json_path.exists():
		# moge2 writes to <-o>/<image_stem>/, so point it at intermediates_dir
		# and then flatten the resulting "base_image/" subfolder up one level.
		run_cmd(
			[
				sys.executable,
				"-m",
				"utils.moge2_infer",
				"-i",
				str(image_path),
				"-o",
				str(intermediates_dir),
				"--maps",
				"--glb",
				"--threshold",
				"0.1",
			],
			cwd=script_dir,
		)
		moge2_subdir = intermediates_dir / image_path.stem
		if moge2_subdir.exists():
			for child in moge2_subdir.iterdir():
				target = intermediates_dir / child.name
				if target.exists():
					if target.is_dir():
						shutil.rmtree(target)
					else:
						target.unlink()
				shutil.move(str(child), str(target))
			moge2_subdir.rmdir()
	else:
		print(f"MoGe2 output already exists at {intermediates_dir}, skipping inference.")


	# Step 4: Text to 3D mesh
	if mesh_fg_path is None:
		print("\n[Step 4] Running text-to-3D mesh generation...")
		# TODO: Add TRELLIS inference code for text-to-3D. For now, fall back to the placeholder sphere mesh.
		mesh_fg_path = script_dir / "obj_meshes" / "mesh_sphere.glb"
		print(f"Note: Text-to-3D is not implemented yet (coming soon). Using placeholder mesh: {mesh_fg_path}")
	else:
		print(f"\n[Step 4] Skipping text-to-3D generation, using provided mesh: {mesh_fg_path}")
		if not mesh_fg_path.exists():
			raise FileNotFoundError(f"Provided foreground mesh not found: {mesh_fg_path}")

	# Step 5: Place mesh_fg on MoGe2 mesh and save in the same scene folder.
	print("\n[Step 5] Placing foreground mesh on MoGe2 mesh...")
	placed_mesh_fg_path = intermediates_dir / "mesh_fg.glb"
	if not placed_mesh_fg_path.exists():
		run_cmd(
			[
				sys.executable,
				"utils/obj_placement.py",
				str(mesh_bg_path),
				str(mesh_fg_path),
				str(placed_mesh_fg_path),
				"--camera",
				str(camera_json_path),
				"--image",
				str(image_path),
				"--prompt",
				p_surface,
				"--no-collision-check",
			],
			cwd=script_dir,
		)
	else:
		print(f"Placed foreground mesh already exists at {placed_mesh_fg_path}, skipping placement.")


	# Step 6: Render foreground mask from the updated mesh_fg.glb.
	print("\n[Step 6] Rendering foreground mask...")
	mask_fg_jpg = intermediates_dir / "mask_fg.jpg"
	mask_fg_png = intermediates_dir / "mask_fg.png"
	if mask_fg_jpg.exists():
		mask_fg_path = mask_fg_jpg
		print(f"Foreground mask already exists at {mask_fg_path}, skipping mask rendering.")
	elif mask_fg_png.exists():
		mask_fg_path = mask_fg_png
		print(f"Foreground mask already exists at {mask_fg_path}, skipping mask rendering.")
	else:
		mask_fg_path = mask_fg_jpg
		run_cmd(
			[
				sys.executable,
				"-m",
				"utils.get_mask",
				str(camera_json_path),
				str(args.width),
				str(args.height),
				str(placed_mesh_fg_path),
				str(mask_fg_path),
			],
			cwd=script_dir,
		)


	warpings_dir = intermediates_dir / "warpings"

	# Step 7: Run warping.
	print("\n[Step 7] Running warping...")
	if not warpings_dir.exists():
		run_cmd(
			[
				sys.executable,
				"-m",
				"utils.warping",
				"--camera_params",
				str(camera_json_path),
				"--image",
				str(image_path),
				"--bg_mesh",
				str(mesh_bg_path),
				"--fg_mesh",
				str(placed_mesh_fg_path),
				"--fg_mask",
				str(mask_fg_path),
				"--output_dir",
				str(warpings_dir),
				"--pano_w",
				"2048",
				"--pano_h",
				"1024",
				"--ior",
				str(p_ior),
			],
			cwd=script_dir,
		)
	else:
		print(f"Warping output already exists at {warpings_dir}, skipping warping.")


	main_path = scene_dir / "main.jpg"
	pano_path = scene_dir / "pano.jpg"

	# Step 8: Run dual-view generation.
	print("\n[Step 8] Running dual-view generation...")
	if not (main_path.exists() and pano_path.exists()):
		# Ensure no stray GPU allocations from earlier in-process steps survive
		# into the dual-view subprocess (device_map="balanced" reads free VRAM
		# at load time and silently offloads layers to CPU if it sees too little).
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		dual_view_cmd = [
			sys.executable,
			"generate_dual_view.py",
			"--main_prompt",
			prompt,
			"--pano_prompt",
			p_pano,
			"--main_clean_path",
			str(image_path),
			"--fg_mask_path",
			str(mask_fg_path),
			"--warpings_dir",
			str(warpings_dir),
			"--output_dir",
			str(intermediates_dir),
			"--alpha",
			str(alpha),
			"--levels",
			str(levels),
			"--time_travel_repeats",
			str(time_travel_repeats),
			"--blend_step_ratio",
			str(blend_step_ratio),
			"--num_steps",
			str(num_steps),
			"--main_guidance_scale",
			str(main_guidance_scale),
			"--pano_guidance_scale",
			str(pano_guidance_scale),
			"--pano_seed",
			str(pano_seed),
		]
		if args.save_intermediate:
			dual_view_cmd.append("--save_intermediate")
		run_cmd(dual_view_cmd, cwd=script_dir)
		# Promote final outputs from intermediates up to scene_dir
		shutil.move(str(intermediates_dir / "main.jpg"), str(main_path))
		shutil.move(str(intermediates_dir / "pano.jpg"), str(pano_path))
	else:
		print(f"Dual-view outputs already exist at {scene_dir}, skipping generation.")

	
	# Step 9: Add shadows
	print("\n[Step 9] Adding shadows...")
	shadow_output_path = scene_dir / "main_with_shadows.jpg"
	if not shadow_output_path.exists():
		pbar = tqdm(total=num_shadow_variations + 1, desc="[Step 9] Adding shadows", unit="step")
		def _shadow_progress(step, total, info):
			pbar.set_postfix_str(info)
			pbar.update(1)
		add_shadows(
			image_path=str(main_path),
			mask_path=str(mask_fg_path),
			output_path=str(shadow_output_path),
			obj=p_obj,
			num_variations=num_shadow_variations,
			progress_cb=_shadow_progress,
		)
		pbar.close()
	else:
		print(f"Shadow image already exists at {shadow_output_path}, skipping.")


if __name__ == "__main__":
	main()

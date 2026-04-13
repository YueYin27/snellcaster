import argparse
import subprocess
import sys
import os
from pathlib import Path

from tqdm import tqdm

from generate_base import generate_base_image
from utils.text_parsing import parse_prompt
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
	script_dir = Path(__file__).resolve().parent
	out_dir = Path(args.out_dir)
	if not out_dir.is_absolute():
		out_dir = (script_dir / out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	# Step 1: Parse prompt into the required prompts.
	print("\n[Step 1] Prompt parsing...")
	prompts_file = out_dir / f"{scene_name}_{seed}" / "prompts.txt"
	if prompts_file.exists():
		print(f"Loading cached prompts from {prompts_file}")
		saved = {}
		for line in prompts_file.read_text().splitlines():
			if "=" in line:
				key, val = line.split("=", 1)
				saved[key.strip()] = val.strip()
		required_keys = {"p", "p_obj", "p_minus", "p_surface", "p_pano"}
		if required_keys <= saved.keys():
			p, p_obj, p_minus, p_surface, p_pano = (
				saved["p"], saved["p_obj"], saved["p_minus"],
				saved["p_surface"], saved["p_pano"],
			)
		else:
			missing = required_keys - saved.keys()
			print(f"Cached prompts missing keys {missing}, re-parsing...")
			p, p_obj, p_minus, p_surface, p_pano = parse_prompt(prompt)
			prompts_file.write_text(
				f"p={p}\np_obj={p_obj}\np_minus={p_minus}\n"
				f"p_surface={p_surface}\np_pano={p_pano}\n"
			)
	else:
		p, p_obj, p_minus, p_surface, p_pano = parse_prompt(prompt)
		prompts_file.write_text(
			f"p={p}\np_obj={p_obj}\np_minus={p_minus}\n"
			f"p_surface={p_surface}\np_pano={p_pano}\n"
		)
	print(f"p: {p}")
	print(f"p_obj: {p_obj}")
	print(f"p_minus: {p_minus}")
	print(f"p_surface: {p_surface}")
	print(f"p_pano: {p_pano}")


	# Step 2: Generate base image from p_minus.
	if not (out_dir / f"{scene_name}_{seed}.jpg").exists():
		print("\n[Step 2] Generating base image with FluxPipeline...")
		base_image_path = out_dir / f"{scene_name}_{seed}.jpg"
		base_img, seed = generate_base_image(
			p_minus,
			width=width,
			height=height,
			seed=seed,
			save_path=str(base_image_path),
		)
	else:
		print(f"Base image already exists at {out_dir / f'{scene_name}_{seed}.jpg'}, skipping generation.")
		base_image_path = out_dir / f"{scene_name}_{seed}.jpg"


	# Step 3: Run MoGe2 with maps+glb and threshold=0.1.
	print("\n[Step 3] Running MoGe2 inference...")
	scene_dir = out_dir / f"{scene_name}_{seed}"
	if not scene_dir.exists():
		moge2_out_dir = out_dir
		run_cmd(
			[
				sys.executable,
				"-m",
				"utils.moge2_infer",
				"-i",
				str(base_image_path),
				"-o",
				str(moge2_out_dir),
				"--maps",
				"--glb",
				"--threshold",
				"0.1",
			],
			cwd=script_dir,
		)
	else:
		print(f"MoGe2 output already exists at {out_dir}, skipping inference.")
		moge2_out_dir = out_dir

	mesh_bg_path = scene_dir / "mesh.glb"
	camera_json_path = scene_dir / "camera.json"
	image_path = scene_dir / "image.jpg"


	# Step 4: Text to 3D mesh
	# TODO: Add TRELLIS inference code for text-to-3D.
	print("\n[Step 4] Running text-to-3D mesh generation (not implemented yet, will be added soon)...")
		

	# Step 5: Place out_dir/mesh_fg.glb on MoGe2 mesh and save in the same scene folder.
	print("\n[Step 5] Placing foreground mesh on MoGe2 mesh...")
	print("Note: The foreground mesh is currently hardcoded to be a sphere. TRELLIS inference for text-to-3D will be added soon.")
	# TODO: The foreground mesh is currently hardcoded to be a sphere. We will add the code for TRELLIS inference soon.
	mesh_fg_input = Path("obj_meshes/mesh_sphere.glb")
	if not mesh_fg_input.exists() and out_dir.name != "results":
		fallback = script_dir / "results" / "mesh_fg.glb"
		if fallback.exists():
			mesh_fg_input = fallback

	if not mesh_fg_input.exists():
		raise FileNotFoundError(
			f"Foreground mesh not found. Expected at: {out_dir / 'mesh_fg.glb'} "
			f"(or fallback: {script_dir / 'results' / 'mesh_fg.glb'})"
		)
	
	placed_mesh_fg_path = scene_dir / "mesh_fg.glb"
	if not placed_mesh_fg_path.exists():
		run_cmd(
			[
				sys.executable,
				"utils/obj_placement.py",
				str(mesh_bg_path),
				str(mesh_fg_input),
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
	mask_fg_path = scene_dir / "mask_fg.jpg"
	if not mask_fg_path.exists():
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
	else:
		print(f"Foreground mask already exists at {mask_fg_path}, skipping mask rendering.")


	# Step 7: Run warping.
	print("\n[Step 7] Running warping...")
	if not (scene_dir / "warpings").exists():
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
				str(scene_dir / "warpings"),
				"--pano_w",
				"2048",
				"--pano_h",
				"1024",
				"--ior",
				"1.5",
			],
			cwd=script_dir,
		)
	else:
		print(f"Warping output already exists at {scene_dir / 'warpings'}, skipping warping.")


	# Step 8: Run dual-view generation.
	print("\n[Step 8] Running dual-view generation...")
	if not ((scene_dir / "main_no_shadow.jpg").exists() and (scene_dir / "pano.jpg").exists()):
		run_cmd(
			[
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
				str(scene_dir / "warpings"),
				"--output_dir",
				str(scene_dir),
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
			],
			cwd=script_dir,
		)
	else:
		print(f"Dual-view outputs already exist at {scene_dir}, skipping generation.")

	
	# Step 9: Add shadows
	print("\n[Step 9] Adding shadows...")
	shadow_output_path = scene_dir / "main.jpg"
	if not shadow_output_path.exists():
		pbar = tqdm(total=num_shadow_variations + 1, desc="[Step 8] Adding shadows", unit="step")
		def _shadow_progress(step, total, info):
			pbar.set_postfix_str(info)
			pbar.update(1)
		add_shadows(
			image_path=str(scene_dir / "main_no_shadow.jpg"),
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

import argparse
import torch
import os
import random
from diffusers import FluxFillPipeline, AutoPipelineForInpainting, QwenImageControlNetModel, QwenImageControlNetInpaintPipeline
from diffusers.utils import load_image
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Run Flux fill inpainting for scenes in a directory")
    parser.add_argument('--output_dir', type=str, default='results/generated_other_shapes', help='Directory containing scene folders')
    parser.add_argument('--model', type=str, default='flux', choices=['flux', 'sdxl', 'qwen'], help='Model to use for inpainting')
    parser.add_argument('--guidance_scale', type=float, default=30.0)
    parser.add_argument('--steps', type=int, default=30)
    args = parser.parse_args()
    output_dir = args.output_dir
    model = args.model


    # GPU memory mapping like compare.py
    num_gpus = torch.cuda.device_count()
    max_memory = None
    if num_gpus > 0:
        max_memory = {}
        for idx in range(num_gpus):
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            free_gib = int(free_bytes / 1024**3)
            total_gib = int(total_bytes / 1024**3)
            cap_gib = max(min(total_gib - 4, free_gib - 2), 1)
            max_memory[idx] = f"{cap_gib}GiB"
            print(f"GPU {idx}: total={total_gib}GiB free={free_gib}GiB cap={cap_gib}GiB")
    print(f"Using {num_gpus} GPU(s) with max_memory={max_memory}")

    # Choose pipeline based on model
    print(f"Loading pipeline for model={model}...")
    if model == 'flux':
        pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, device_map="balanced", max_memory=max_memory)
        img_name = "main_flux_fill.jpg"
    elif model == 'sdxl':
        pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.bfloat16, device_map="balanced", max_memory=max_memory)
        img_name = "main_sdxl_fill.jpg"
    elif model == 'qwen':
        controlnet = QwenImageControlNetModel.from_pretrained("InstantX/Qwen-Image-ControlNet-Inpainting", torch_dtype=torch.bfloat16).to("cuda")
        pipe = QwenImageControlNetInpaintPipeline.from_pretrained("Qwen/Qwen-Image", controlnet=controlnet, torch_dtype=torch.bfloat16).to("cuda")
        img_name = "main_qwen_fill.jpg"
    else:
        raise ValueError(f"Model {model} is not supported for inpainting")

    if model == 'flux' or model == 'sdxl':
        for scene_name in sorted(os.listdir(output_dir)):
                print("\n========================================================")
                print(f"Processing scene: {scene_name}")
                print("========================================================\n")

                for obj in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
                    if not os.path.exists(os.path.join(output_dir, scene_name, obj, img_name)):
                        print(f"Processing object: {obj} in scene: {scene_name}")
                        prompt = f"a solid transparent glass {obj}"
                        scene_folder = os.path.join(output_dir, scene_name, obj)
                        clean_image_path = os.path.join(output_dir, scene_name, "base_image.jpg")
                        mask_image_path = os.path.join(output_dir, scene_name, obj, "mask_fg.jpg")
                        image = load_image(str(clean_image_path))
                        mask = load_image(str(mask_image_path))
                        width, height = image.size

                        seed = random.randint(0, 9999)
                        generator = torch.Generator(device="cuda").manual_seed(seed)

                        print("Running inpainting...")
                        out = pipe(
                            prompt=prompt,
                            image=image,
                            mask_image=mask,
                            height=height,
                            width=width,
                            guidance_scale=float(args.guidance_scale),
                            num_inference_steps=int(args.steps),
                            max_sequence_length=512,
                            generator=generator,
                        ).images[0]
                        out_path = os.path.join(output_dir, scene_name, obj, img_name)
                        out.save(out_path)
                        print(f"Saved output to {out_path}")
    
    elif model == 'qwen':
        for scene_name in sorted(os.listdir(output_dir)):
                print("\n========================================================")
                print(f"Processing scene: {scene_name}")
                print("========================================================\n")

                for obj in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
                    if not os.path.exists(os.path.join(output_dir, scene_name, obj, img_name)):
                        print(f"Processing object: {obj} in scene: {scene_name}")
                        prompt = f"a solid transparent glass {obj}"
                        scene_folder = os.path.join(output_dir, scene_name, obj)
                        clean_image_path = os.path.join(output_dir, scene_name, "base_image.jpg")
                        mask_image_path = os.path.join(output_dir, scene_name, obj, "mask_fg.jpg")
                        image = load_image(str(clean_image_path))
                        mask = load_image(str(mask_image_path))
                        width, height = image.size

                        seed = random.randint(0, 9999)
                        generator = torch.Generator(device="cuda").manual_seed(seed)

                        print("Running inpainting...")
                        out = pipe(
                            prompt=prompt,
                            control_image=image,
                            control_mask=mask,
                            height=height,
                            width=width,
                            num_inference_steps=int(args.steps),
                            true_cfg_scale=4.0,
                            generator=generator,
                        ).images[0]
                        out_path = os.path.join(output_dir, scene_name, obj, img_name)
                        out.save(out_path)
                        print(f"Saved output to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple post-processing script to refine generated images with additional prompts.
Uses FLUX.1-Kontext with relighting LoRA for lighting modifications without changing structure.

Usage:
    python add_shadows.py --image_path INPUT.png --output_dir OUTPUT_DIR --num_variations 10
"""

import argparse
import os
import random

import torch
from PIL import Image
from diffusers import DiffusionPipeline

NUM_STEPS = 20  # Number of inference steps (default for Kontext)
GUIDANCE_SCALE = 2.5  # Guidance scale (3.5 for dev model, higher = stronger prompt adherence)

# # ============ PROMPT PARAMETERS ============
# for obj in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
#     PROMPT = f"Add a very dim, soft, round shadow beneath the glass {obj}, and introduce subtle glossy specular highlights and small reflection spots on the glass surface to enhance shine, but do not change the existing pattern at all. Remove background artifacts and noise. Increase global clarity, sharpness, and brightness while preserving the original geometry, pattern, texture, and scale of the glass {obj}. Improve focus and micro-contrast so the entire scene appears crisp and clean, without altering structure."
#     # PROMPT = f"Light from the campfire creates round shaped soft shadows of the glass {obj} on the ground, but keep the shape and size of the {obj} exactly the same."
#     # PROMPT = f"Golden sunlight creates round shaped soft shadows of the glass {obj} and caustics on the ground, the {obj} is shiny and glossy."
#     # PROMPT = f"Light from the TV screen creates round shaped dim shadows of the glass {obj} on the table, but keep the shape and appearance of the {obj} exactly the same."
#     # PROMPT = f"Sunlight creates round shaped soft shadows of the glass {obj} on the table, but keep the shape and size of the {obj} exactly the same."
#     # PROMPT = f"Make the glass {obj} shiny and glossy, but keep the contents of the {obj} exactly the same."
#     # PROMPT = f"Smooth the glass {obj} surface, remove blocky effects, add reflections, remove artifacts on the background, but keep the shape and appearance of the {obj} exactly the same."
#     NEGATIVE_PROMPT = (
#         f"changing the glass {obj}, moving the {obj}, deforming the {obj}, "
#         f"altering the {obj} size, modifying the {obj} shape, changing the refracted pattern, "
#         f"distorting the glass {obj} pattern, restructuring the {obj}, warping the "
#         f"{obj} surface, changing {obj} structure"
#     )


def process_image(pipe, image_path, output_path, seed, prompt, negative_prompt):
    """Process a single image with the given seed."""
    print(f"Loading image from: {image_path}")
    input_image = Image.open(image_path).convert("RGB")
    original_width, original_height = input_image.size
    print(f"Original image size: {original_width}x{original_height}")

    target_width, target_height = 1360, 768
    print(f"Resizing to {target_width}x{target_height} for processing...")
    input_image = input_image.resize((target_width, target_height), Image.LANCZOS)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    print(f"Processing with seed {seed}")
    print(f"Prompt: '{prompt}'")
    if negative_prompt:
        print(f"Negative prompt: '{negative_prompt}'")
    print(f"Target size: {target_width}x{target_height}, Steps: {NUM_STEPS}, Guidance: {GUIDANCE_SCALE}")
    image = pipe(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        height=target_height,
        width=target_width,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        _auto_resize=False,
    ).images[0]

    if image.size != (original_width, original_height):
        print(f"Resizing from {image.size[0]}x{image.size[1]} back to {original_width}x{original_height}")
        image = image.resize((original_width, original_height), Image.LANCZOS)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving enhanced image to: {output_path}")
    image.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Add shadows to a single image.")
    parser.add_argument("--image_name", required=True, help="Name of the image to add shadows.")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--num_variations", type=int, required=True, help="Number of variations to generate.")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    img_name = args.image_name
    num_variations = args.num_variations
    if num_variations < 1:
        raise ValueError("--num_variations must be >= 1")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading FLUX.1-Kontext model...")
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )

    print("Loading relighting LoRA...")
    pipe.load_lora_weights("kontext-community/relighting-kontext-dev-lora-v3")

    print(f"\n{'#'*80}")
    print(f"# SINGLE IMAGE MODE: Generating {num_variations} variations")
    print(f"{'#'*80}")

    random_seeds = [random.randint(0, 99999999) for _ in range(num_variations)]
    print(
        f"Generating {num_variations} variations with seeds: {random_seeds[:5]}..."
        if len(random_seeds) > 5
        else f"Generating {num_variations} variations with seeds: {random_seeds}"
    )

    for scene_name in sorted(os.listdir(output_dir)):
    # for scene_name in sorted(os.listdir(output_dir), reverse=True):
        print("\n========================================================")
        print(f"Processing scene: {scene_name}")
        print("========================================================\n")
        for obj in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
            img_dir = os.path.join(output_dir, scene_name, obj, "main_shadow")
            if not os.path.exists(os.path.join(img_dir, f"main_shadow_{num_variations-1}.jpg")):
                os.makedirs(img_dir, exist_ok=True)
                print(f"Processing object: {obj} in scene: {scene_name}")
                prompt = f"Add a dim and soft shadow beneath the glass {obj}, and subtle specular highlights on the glass surface to enhance shine, but do not change the existing pattern at all. Remove background artifacts and noise."
                negative_prompt = (
                    f"changing the glass {obj}, moving the {obj}, deforming the {obj}, "
                    f"altering the {obj} size, modifying the {obj} shape, changing the refracted pattern, "
                    f"distorting the glass {obj} pattern, restructuring the {obj}, warping the "
                    f"{obj} surface, changing {obj} structure"
                )

                for i, seed in enumerate(random_seeds, 1):
                    print(f"\n{'='*60}")
                    print(f"Processing variation {i}/{num_variations} with seed {seed}")
                    print(f"{'='*60}")
                    img_path = os.path.join(output_dir, scene_name, obj, img_name)
                    output_path = os.path.join(img_dir, f"main_shadow_{i}.jpg")
                    process_image(pipe, img_path, output_path, seed, prompt, negative_prompt)

        print(f"\n{'#'*80}")
        print(f"# DONE! Generated {num_variations} variations in {img_dir}")
        print(f"{'#'*80}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Post-processing to add shadows/highlights via FLUX.1-Kontext with relighting LoRA.
Generates N variations, scores them by masked PSNR, and keeps only the best.
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline

NUM_STEPS = 20
GUIDANCE_SCALE = 3.5


def masked_psnr(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> float:
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape.")
    mask = mask.expand_as(img1)
    valid = mask == 1.0
    diff = (img1 - img2)[valid]
    if diff.numel() == 0:
        raise ValueError("Mask selects no valid pixels.")
    mse = torch.mean(diff ** 2)
    if torch.isclose(mse, torch.tensor(0.0, device=mse.device)):
        return float("inf")
    return float((10.0 * torch.log10(1.0 / mse)).item())


def compute_psnr_from_pil(variation: Image.Image, original: Image.Image, mask: Image.Image) -> float:
    var_rgb = variation.convert("RGB")
    orig_rgb = original.convert("RGB")
    mask_gray = mask.convert("L")

    if var_rgb.size != orig_rgb.size:
        orig_rgb = orig_rgb.resize(var_rgb.size, Image.LANCZOS)
    if mask_gray.size != var_rgb.size:
        mask_gray = mask_gray.resize(var_rgb.size, Image.LANCZOS)

    var_t = torch.from_numpy(np.array(var_rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    orig_t = torch.from_numpy(np.array(orig_rgb, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mask_t = (torch.from_numpy(np.array(mask_gray, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0) > 0.5).float()

    return masked_psnr(var_t, orig_t, mask_t)


def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    pipe.load_lora_weights("kontext-community/relighting-kontext-dev-lora-v3")
    return pipe


def generate_variation(pipe, image_path: str, seed: int, prompt: str, negative_prompt: str) -> Image.Image:
    input_image = Image.open(image_path).convert("RGB")
    original_width, original_height = input_image.size

    target_width, target_height = 1360, 768
    input_image = input_image.resize((target_width, target_height), Image.LANCZOS)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        image=input_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=target_height,
        width=target_width,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        _auto_resize=False,
    ).images[0]

    if image.size != (original_width, original_height):
        image = image.resize((original_width, original_height), Image.LANCZOS)

    return image


def build_prompts(obj: str) -> tuple[str, str]:
    prompt = (
        f"Add a dim and soft shadow beneath the transparent {obj}, and subtle specular "
        f"highlights on the transparent surface to enhance shine, but do not change the "
        f"existing pattern at all. Remove background artifacts and noise."
    )
    negative_prompt = (
        f"changing the transparent {obj}, moving the {obj}, deforming the {obj}, "
        f"altering the {obj} size, modifying the {obj} shape, changing the refracted pattern, "
        f"distorting the transparent {obj} pattern, restructuring the {obj}, warping the "
        f"{obj} surface, changing {obj} structure."
    )
    return prompt, negative_prompt


def add_shadows(
    image_path: str,
    mask_path: str,
    output_path: str,
    obj: str,
    num_variations: int = 5,
    pipe=None,
    progress_cb=None,
):
    """
    Generate shadow variations, pick the best by masked PSNR, and save it.

    Args:
        image_path: Path to the input image (e.g. main.jpg).
        mask_path: Path to the foreground mask (e.g. mask_fg.jpg).
        output_path: Where to save the best result (e.g. main_shadow.jpg).
        obj: Name of the transparent object.
        num_variations: How many variations to generate.
        pipe: Pre-loaded DiffusionPipeline. Loaded automatically if None.
        progress_cb: Optional callable(step: int, total: int, info: str) for progress updates.
    """
    original_image = Image.open(image_path).convert("RGB")
    mask_image = Image.open(mask_path)
    prompt, negative_prompt = build_prompts(obj)

    if pipe is None:
        pipe = load_pipeline()

    seeds = [random.randint(0, 99999999) for _ in range(num_variations)]
    total_steps = num_variations + 1  # N generations + 1 scoring pass

    variations = []
    for i, seed in enumerate(seeds):
        image = generate_variation(pipe, image_path, seed, prompt, negative_prompt)
        variations.append((seed, image))
        if progress_cb:
            progress_cb(i + 1, total_steps, f"seed={seed}")

    best_psnr = -float("inf")
    best_image = None
    best_seed = None
    for seed, image in variations:
        try:
            psnr = compute_psnr_from_pil(image, original_image, mask_image)
            if psnr > best_psnr:
                best_psnr = psnr
                best_image = image
                best_seed = seed
        except Exception:
            pass

    if progress_cb:
        progress_cb(total_steps, total_steps, f"best seed={best_seed} PSNR={best_psnr:.2f}")

    if best_image is None:
        raise RuntimeError("All variations failed PSNR computation.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    best_image.save(output_path)
    return best_seed, best_psnr

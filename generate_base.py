from diffusers import FluxPipeline
import torch
import argparse
import os
import re
import random


def generate_base_image(prompt: str, model_id: str = "black-forest-labs/FLUX.1-dev", width: int = 1280, height: int = 720, seed: int | None = None, save_path: str = "results/base_image.jpg"):
    """Generate a single base image for `prompt` using the specified Flux model.

    Returns the generated PIL Image.
    """
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # If no seed provided, generate a random 31-bit seed and use it
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    outputs = pipe(prompt, height=height, width=width, generator=generator)
    image = outputs.images[0]

    # save the image to out_dir
    base, ext = os.path.splitext(save_path)
    if ext.lower() in (".png", ".jpg", ".jpeg"):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        out_path = save_path
    else:
        os.makedirs(save_path, exist_ok=True)
        out_path = os.path.join(save_path, "base_image.jpg")
    image.save(out_path)
    print(f"Saved base image to {out_path} with seed {seed}")

    return image, seed


def main():
    parser = argparse.ArgumentParser(description="Generate a base image with FluxPipeline")
    parser.add_argument("prompt", help="Scene prompt to generate")
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev", help="Model ID to load")
    parser.add_argument("--width", type=int, default=1280, help="Output width")
    parser.add_argument("--height", type=int, default=720, help="Output height")
    parser.add_argument("--save_path", default=None, help="If provided, save the image to this file path or directory. Default: don't save")
    parser.add_argument("--seed", type=int, default=42, help="Optional random seed for generation")
    args = parser.parse_args()

    img, seed = generate_base_image(args.prompt, model_id=args.model, width=args.width, height=args.height, seed=args.seed, save_path=args.save_path)


if __name__ == "__main__":
    main()

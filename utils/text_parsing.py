from transformers import AutoProcessor, AutoModelForCausalLM, set_seed
import argparse
import gc
import os
import random

import numpy as np
import torch


def parse_prompt(prompt: str, model_id: str = "google/gemma-4-E2B-it", seed: int = 0):
    # Make every source of nondeterminism reproducible.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")

    def generate_text(user_prompt: str, max_new_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt},
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
        )
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        return response.strip()

    p = prompt
    p_obj = generate_text(
        f"Identify the phrase that describes the transparent object in the following description: {p} Extract only the object noun and remove material descriptors (e.g., if the phrase is 'a transparent solid glass sphere', output 'sphere').",
    )
    p_minus = generate_text(
        f"Rewrite this description: \"{p}\". Completely remove the transparent object mention, including the object noun itself (for example remove the whole phrase 'a solid transparent glass sphere on the table', not just 'glass'). Replace that exact span in-place with wording that the same surface is clean and empty (for example: 'the table is clean and empty, with nothing on it'). Do not append this as a new ending sentence; keep it at the original object location in the sentence flow. Keep all other scene details unchanged and natural. Self-check before output: the final sentence must not contain object words from the removed item (e.g., sphere, ball, crystal) or any transparency/material words (e.g., transparent, glass, clear). Output exactly one rewritten description and nothing else.",
    )

    p_surface = generate_text(
        f"Extract exactly one short phrase that names the TOP SURFACE where the transparent object was placed in this description: {p}. The output MUST be a single phrase in one of these formats: top surface of the <supporting-surface>, tabletop of the <supporting-surface>, or surface of the <supporting-surface>. Do NOT output only the object name (e.g. stone table) or any extra words, punctuation, or explanation. Output exactly one phrase and nothing else.",
    )

    p_pano = generate_text(
        f"Rewrite this scene description as a single equirectangular 360-degree panorama prompt captured from the exact position of the transparent object in the original scene: {p}. Remove the transparent object entirely and describe the support surface as clean and empty. Keep the same room, furniture, lighting, style, and realism as the original scene, but describe a full spherical environment view consistent with a panorama. Output exactly one concise prompt and nothing else.",
        )

    p_ior = generate_text(
        f"Given this scene description: \"{p}\", identify the transparent material that the object is made of (e.g., glass, water, ice, diamond, acrylic, quartz) and output ONLY its index of refraction (IoR) at visible wavelengths as a single decimal number (e.g., 1.5 for glass, 1.33 for water, 1.31 for ice, 2.42 for diamond, 1.49 for acrylic, 1.54 for quartz). If the material is not explicitly stated, infer the most likely transparent material from the context and output its typical IoR. Output exactly one decimal number and nothing else.",
    )

    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return p, p_obj, p_minus, p_surface, p_pano, p_ior


def main():
    parser = argparse.ArgumentParser(description="Extract object noun and rewrite scene using Gemma model")
    parser.add_argument("prompt", help="Scene description prompt")
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it", help="Model ID to load")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible generation")
    parser.add_argument("--out", default=None, help="Optional path to write KV-formatted prompts.txt")
    args = parser.parse_args()

    p, p_obj, p_minus, p_surface, p_pano, p_ior = parse_prompt(args.prompt, model_id=args.model_id, seed=args.seed)

    print(f"p: {p}")
    print(f"p_obj: {p_obj}")
    print(f"p_minus: {p_minus}")
    print(f"p_surface: {p_surface}")
    print(f"p_pano: {p_pano}")
    print(f"p_ior: {p_ior}")

    if args.out is not None:
        from pathlib import Path
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"p={p}\np_obj={p_obj}\np_minus={p_minus}\n"
            f"p_surface={p_surface}\np_pano={p_pano}\np_ior={p_ior}\n"
        )


if __name__ == "__main__":
    main()

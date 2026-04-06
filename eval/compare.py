import os
import argparse
import torch
from diffusers import DiffusionPipeline

# CLI args: object name and scene key
parser = argparse.ArgumentParser(description="Generate comparison images for scenes with a specified object")
parser.add_argument('--output_dir', type=str, default='results/generated/', help='Directory to save generated images')
parser.add_argument('--model', type=str, default='flux1', choices=['flux1', 'flux2', 'sd35', 'qwen'], help='Model to use for generation')
args = parser.parse_args()
output_dir = args.output_dir
model=args.model

os.makedirs(output_dir, exist_ok=True)
height = 720
width = 1280
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


if model == "flux1":
    ###################################### Flux ####################################
    pipe_flux1 = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map="balanced", max_memory=max_memory)
    for scene_name in sorted(os.listdir(output_dir)):
        print("\n========================================================")
        print(f"Processing scene: {scene_name}")
        print("========================================================\n")
        # get seed from scene_name format: {scene}_{seed}
        scene = scene_name.rsplit("_", 1)[0]
        seed = int(scene_name.rsplit("_", 1)[1])

        for obj in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
            if not os.path.exists(os.path.join(output_dir, scene_name, obj, "main_flux1.jpg")):
                print(f"Generating image of {obj} in {scene} using FLUX.1-dev")
                scene_prompts = {
                    "living_room": f"A cozy living room with a polished wooden coffee table close to the camera, with a solid transparent glass {obj} on top, surrounded by a beige sofa with colorful pillows, a patterned rug, green plants, bookshelves, framed wall art, and sunlight filtering through sheer curtains.",
                    "dining_room": f"A bright dining room with a wooden dining table placed in the center, with a solid transparent glass {obj} on top, surrounded by upholstered chairs, a pendant lamp above, fruit bowls and paintings in the background, and daylight streaming through tall windows with patterned curtains.",
                    "office": f"A minimalist home office with a smooth wooden desk positioned closer to the camera, with a solid transparent glass {obj} on top, surrounded by a black office chair, bookshelves with plants, framed posters, a side table with a computer monitor, and a large window with blinds letting in soft daylight.",
                    "kitchen": f"A modern kitchen with a large marble island in the center, with a solid transparent glass {obj} on top, surrounded by wooden cabinetry, bar stools, hanging lights, colorful utensils, and reflections from stainless-steel appliances under bright morning light.",
                    "artroom": f"An art classroom with a rectangular wooden worktable near the camera, with a solid transparent glass {obj} on top, surrounded by easels, color-splattered stools, sketches pinned on the walls, jars of brushes on shelves, and warm daylight pouring through wide windows.",
                    "cafe": f"A minimalist café interior with a square wooden table in the near foreground, with a solid transparent glass {obj} on top, surrounded by metal-framed chairs, green plants, hanging lights, a counter with pastries and cups in the background, and soft sunlight illuminating the colorful tiled floor.",
                    "landscape": f"A beautiful landscape with a river and mountains, viewed from a camera positioned directly in front of a stone table and chairs in the immediate foreground, filling the lower frame and very close to the lens, a solid transparent glass {obj} on the table, on the left of some colorful food on the table.",
                    "karaoke": f"A karaoke room with colorful lights, TV on the wall displaying music videos, a coffee table with a solid transparent glass {obj} on top, in front of the TV, and a sofa around the coffee table.",
                    "cave": f"A rocky cave interior lit by a bright campfire, warm flickering light casting dramatic shadows on the walls, a solid transparent glass {obj} on the ground, with camping gear scattered around—tents, sleeping bags, backpacks, lanterns, cooking pots, and a folding chair—smoke and embers in the air, cozy but high contrast.",
                    "desert": f"A high-noon desert scene with blinding sunlight and hard shadows, heat haze over sand and rocks, a solid transparent glass {obj} on the ground, and camping gear in the foreground—tent, backpacks, and a small stove.",
                }
                prompt = scene_prompts[scene]
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe_flux1(prompt=prompt, height=height, width=width, generator=generator).images[0]
                image.save(os.path.join(output_dir, scene_name, obj, "main_flux1.jpg"))
                print(f"Saved main_flux1.jpg to {os.path.join(output_dir, scene_name, obj)}/")
    del pipe_flux1
    torch.cuda.empty_cache()

elif model == "flux2":
    ###################################### Flux.2 ####################################
    pipe_flux2 = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16, device_map="balanced", max_memory=max_memory)
    for scene_name in sorted(os.listdir(output_dir)):
    # for scene_name in sorted(os.listdir(output_dir), reverse=True):
        if not os.path.isdir(os.path.join(output_dir, scene_name)):
            continue

        print("\n========================================================")
        print(f"Processing scene: {scene_name}")
        print("========================================================\n")
        # get seed from scene_name format: {scene}_{seed}
        scene = scene_name.rsplit("_", 1)[0]
        seed = int(scene_name.rsplit("_", 1)[1])

        for obj in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
            if not os.path.exists(os.path.join(output_dir, scene_name, obj, "main_flux2.jpg")):
                print(f"Generating image of {obj} in {scene} using FLUX.2-dev")
                scene_prompts = {
                    "living_room": f"A cozy living room with a polished wooden coffee table close to the camera, with a solid transparent glass {obj} on top, surrounded by a beige sofa with colorful pillows, a patterned rug, green plants, bookshelves, framed wall art, and sunlight filtering through sheer curtains.",
                    "dining_room": f"A bright dining room with a wooden dining table placed in the center, with a solid transparent glass {obj} on top, surrounded by upholstered chairs, a pendant lamp above, fruit bowls and paintings in the background, and daylight streaming through tall windows with patterned curtains.",
                    "office": f"A minimalist home office with a smooth wooden desk positioned closer to the camera, with a solid transparent glass {obj} on top, surrounded by a black office chair, bookshelves with plants, framed posters, a side table with a computer monitor, and a large window with blinds letting in soft daylight.",
                    "kitchen": f"A modern kitchen with a large marble island in the center, with a solid transparent glass {obj} on top, surrounded by wooden cabinetry, bar stools, hanging lights, colorful utensils, and reflections from stainless-steel appliances under bright morning light.",
                    "artroom": f"An art classroom with a rectangular wooden worktable near the camera, with a solid transparent glass {obj} on top, surrounded by easels, color-splattered stools, sketches pinned on the walls, jars of brushes on shelves, and warm daylight pouring through wide windows.",
                    "cafe": f"A minimalist café interior with a square wooden table in the near foreground, with a solid transparent glass {obj} on top, surrounded by metal-framed chairs, green plants, hanging lights, a counter with pastries and cups in the background, and soft sunlight illuminating the colorful tiled floor.",
                    "landscape": f"A beautiful landscape with a river and mountains, viewed from a camera positioned directly in front of a stone table and chairs in the immediate foreground, filling the lower frame and very close to the lens, a solid transparent glass {obj} on the table, on the left of some colorful food on the table.",
                    "karaoke": f"A karaoke room with colorful lights, TV on the wall displaying music videos, a coffee table with a solid transparent glass {obj} on top, in front of the TV, and a sofa around the coffee table.",
                    "cave": f"A rocky cave interior lit by a bright campfire, warm flickering light casting dramatic shadows on the walls, a solid transparent glass {obj} on the ground, with camping gear scattered around—tents, sleeping bags, backpacks, lanterns, cooking pots, and a folding chair—smoke and embers in the air, cozy but high contrast.",
                    "desert": f"A high-noon desert scene with blinding sunlight and hard shadows, heat haze over sand and rocks, a solid transparent glass {obj} on the ground, and camping gear in the foreground—tent, backpacks, and a small stove.",
                }
                prompt = scene_prompts[scene]
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe_flux2(prompt=prompt, height=height, width=width, generator=generator).images[0]
                image.save(os.path.join(output_dir, scene_name, obj, "main_flux2.jpg"))
                print(f"Saved main_flux2.jpg to {os.path.join(output_dir, scene_name, obj)}/")
    del pipe_flux2
    torch.cuda.empty_cache()

elif model == "sd35":
    ###################################### Stable Diffusion 3.5 ####################################
    pipe_sd35 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16, device_map="balanced")
    for scene_name in sorted(os.listdir(output_dir)):
        print("\n========================================================")
        print(f"Processing scene: {scene_name}")
        print("========================================================\n")
        # get seed from scene_name format: {scene}_{seed}
        scene = scene_name.rsplit("_", 1)[0]
        seed = int(scene_name.rsplit("_", 1)[1])

        for obj in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
            if not os.path.exists(os.path.join(output_dir, scene_name, obj, "main_sd35.jpg")):
                print(f"Generating image of {obj} in {scene} using Stable Diffusion 3.5")
                scene_prompts = {
                    "living_room": f"A cozy living room with a polished wooden coffee table close to the camera, with a solid transparent glass {obj} on top, surrounded by a beige sofa with colorful pillows, a patterned rug, green plants, bookshelves, framed wall art, and sunlight filtering through sheer curtains.",
                    "dining_room": f"A bright dining room with a wooden dining table placed in the center, with a solid transparent glass {obj} on top, surrounded by upholstered chairs, a pendant lamp above, fruit bowls and paintings in the background, and daylight streaming through tall windows with patterned curtains.",
                    "office": f"A minimalist home office with a smooth wooden desk positioned closer to the camera, with a solid transparent glass {obj} on top, surrounded by a black office chair, bookshelves with plants, framed posters, a side table with a computer monitor, and a large window with blinds letting in soft daylight.",
                    "kitchen": f"A modern kitchen with a large marble island in the center, with a solid transparent glass {obj} on top, surrounded by wooden cabinetry, bar stools, hanging lights, colorful utensils, and reflections from stainless-steel appliances under bright morning light.",
                    "artroom": f"An art classroom with a rectangular wooden worktable near the camera, with a solid transparent glass {obj} on top, surrounded by easels, color-splattered stools, sketches pinned on the walls, jars of brushes on shelves, and warm daylight pouring through wide windows.",
                    "cafe": f"A minimalist café interior with a square wooden table in the near foreground, with a solid transparent glass {obj} on top, surrounded by metal-framed chairs, green plants, hanging lights, a counter with pastries and cups in the background, and soft sunlight illuminating the colorful tiled floor.",
                    "landscape": f"A beautiful landscape with a river and mountains, viewed from a camera positioned directly in front of a stone table and chairs in the immediate foreground, filling the lower frame and very close to the lens, a solid transparent glass {obj} on the table, on the left of some colorful food on the table.",
                    "karaoke": f"A karaoke room with colorful lights, TV on the wall displaying music videos, a coffee table with a solid transparent glass {obj} on top, in front of the TV, and a sofa around the coffee table.",
                    "cave": f"A rocky cave interior lit by a bright campfire, warm flickering light casting dramatic shadows on the walls, a solid transparent glass {obj} on the ground, with camping gear scattered around—tents, sleeping bags, backpacks, lanterns, cooking pots, and a folding chair—smoke and embers in the air, cozy but high contrast.",
                    "desert": f"A high-noon desert scene with blinding sunlight and hard shadows, heat haze over sand and rocks, a solid transparent glass {obj} on the ground, and camping gear in the foreground—tent, backpacks, and a small stove.",
                }
                prompt = scene_prompts[scene]
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe_sd35(prompt=prompt, height=height, width=width, generator=generator).images[0]
                image.save(os.path.join(output_dir, scene_name, obj, "main_sd35.jpg"))
                print(f"Saved main_sd35.jpg to {os.path.join(output_dir, scene_name, obj)}/")
    del pipe_sd35
    torch.cuda.empty_cache()

elif model == "qwen":
    ###################################### Qwen-2521 #######################################
    pipe_qwen = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image-2512", torch_dtype=torch.bfloat16, device_map="balanced")
    for scene_name in sorted(os.listdir(output_dir)):
        print("\n========================================================")
        print(f"Processing scene: {scene_name}")
        print("========================================================\n")
        # get seed from scene_name format: {scene}_{seed}
        scene = scene_name.rsplit("_", 1)[0]
        seed = int(scene_name.rsplit("_", 1)[1])

        for obj in ["sphere", "dog", "fox", "sculpture", "pyramid", "cylinder"]:
            if not os.path.exists(os.path.join(output_dir, scene_name, obj, "main_flux_qwen.jpg")):
                print(f"Generating image of {obj} in {scene} using Qwen-2512")
                scene_prompts = {
                    "living_room": f"A cozy living room with a polished wooden coffee table close to the camera, with a solid transparent glass {obj} on top, surrounded by a beige sofa with colorful pillows, a patterned rug, green plants, bookshelves, framed wall art, and sunlight filtering through sheer curtains.",
                    "dining_room": f"A bright dining room with a wooden dining table placed in the center, with a solid transparent glass {obj} on top, surrounded by upholstered chairs, a pendant lamp above, fruit bowls and paintings in the background, and daylight streaming through tall windows with patterned curtains.",
                    "office": f"A minimalist home office with a smooth wooden desk positioned closer to the camera, with a solid transparent glass {obj} on top, surrounded by a black office chair, bookshelves with plants, framed posters, a side table with a computer monitor, and a large window with blinds letting in soft daylight.",
                    "kitchen": f"A modern kitchen with a large marble island in the center, with a solid transparent glass {obj} on top, surrounded by wooden cabinetry, bar stools, hanging lights, colorful utensils, and reflections from stainless-steel appliances under bright morning light.",
                    "artroom": f"An art classroom with a rectangular wooden worktable near the camera, with a solid transparent glass {obj} on top, surrounded by easels, color-splattered stools, sketches pinned on the walls, jars of brushes on shelves, and warm daylight pouring through wide windows.",
                    "cafe": f"A minimalist café interior with a square wooden table in the near foreground, with a solid transparent glass {obj} on top, surrounded by metal-framed chairs, green plants, hanging lights, a counter with pastries and cups in the background, and soft sunlight illuminating the colorful tiled floor.",
                    "landscape": f"A beautiful landscape with a river and mountains, viewed from a camera positioned directly in front of a stone table and chairs in the immediate foreground, filling the lower frame and very close to the lens, a solid transparent glass {obj} on the table, on the left of some colorful food on the table.",
                    "karaoke": f"A karaoke room with colorful lights, TV on the wall displaying music videos, a coffee table with a solid transparent glass {obj} on top, in front of the TV, and a sofa around the coffee table.",
                    "cave": f"A rocky cave interior lit by a bright campfire, warm flickering light casting dramatic shadows on the walls, a solid transparent glass {obj} on the ground, with camping gear scattered around—tents, sleeping bags, backpacks, lanterns, cooking pots, and a folding chair—smoke and embers in the air, cozy but high contrast.",
                    "desert": f"A high-noon desert scene with blinding sunlight and hard shadows, heat haze over sand and rocks, a solid transparent glass {obj} on the ground, and camping gear in the foreground—tent, backpacks, and a small stove.",
                }
                prompt = scene_prompts[scene]
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe_qwen(prompt=prompt, height=height, width=width, generator=generator).images[0]
                image.save(os.path.join(output_dir, scene_name, obj, "main_qwen.jpg"))
                print(f"Saved main_qwen.jpg to {os.path.join(output_dir, scene_name, obj)}/")
    del pipe_qwen
    torch.cuda.empty_cache()

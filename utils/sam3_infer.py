import argparse
import random
import torch
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# Initialize model and processor once
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Define colors: red, pink, orange, yellow, green, blue, purple
COLORS = [
    (255, 0, 0),      # red
    (255, 192, 203),  # pink
    (255, 165, 0),    # orange
    (255, 255, 0),    # yellow
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (128, 0, 128),    # purple
]


def sam3_infer(image_path: str, prompt: str = "tabletop", vis: bool = False,
               output_path: str = "image_with_masks.png"):
    """
    Run SAM3 on an image, return bounding box center coordinates, optionally save visualization.

    Args:
        image_path: Path to input image.
        prompt: Text prompt for SAM3.
        vis: If True, save image with colored translucent masks overlaid.
        output_path: Path to save visualization when vis is True.

    Returns:
        List of (x_center, y_center) tuples for each mask's bounding box center (in pixel coordinates).
    """
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    image_width, image_height = image.size  # (width, height)
    centers = []

    # Prepare base image for visualization
    if vis:
        base = image.convert("RGBA")

    for mask, box, score in zip(masks, boxes, scores):
        mask = mask.cpu().numpy()
        box = box.cpu().numpy()

        # Compute bounding box center (x, y)
        # Assuming box format is [x_min, y_min, x_max, y_max]
        if len(box) >= 4:
            x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0
            centers.append((float(x_center), float(y_center)))
        else:
            centers.append((None, None))

        # Remove batch/sequence dimensions by squeezing
        mask = np.squeeze(mask)

        # Convert mask to 0/1 binary then to 0/255 uint8
        mask = (mask > 0).astype(np.uint8) * 255

        # Ensure mask is 2D and matches image size if possible
        if mask.ndim == 1 and mask.size == image_height * image_width:
            mask = mask.reshape(image_height, image_width)
        if mask.ndim != 2:
            raise ValueError(f"Unexpected mask shape {mask.shape}, expected 2D.")

        if vis:
            mask_img = Image.fromarray(mask)

            # Create a colored, translucent overlay from the mask
            color = random.choice(COLORS)
            overlay = Image.new("RGBA", image.size, (*color, 0))
            alpha = mask_img.point(lambda p: 128 if p > 0 else 0)  # 0.5 opacity
            overlay.putalpha(alpha)

            # Composite overlay onto the base image
            base = Image.alpha_composite(base, overlay)

    if vis:
        base.save(output_path)

    return centers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAM3 inference on an image.")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--prompt", default="tabletop", help="Text prompt for SAM3")
    parser.add_argument("--vis", action="store_true", default=False, help="Save visualization with masks")
    parser.add_argument("--output", default="image_with_masks.png",
                        help="Output path for visualization when --vis is set")

    args = parser.parse_args()
    centers = sam3_infer(args.image_path, prompt=args.prompt, vis=args.vis, output_path=args.output)
    print(centers)

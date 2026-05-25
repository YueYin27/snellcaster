"""Stitch saved main-image Tweedie estimates into an MP4."""
import argparse
import re
import sys
from pathlib import Path

import imageio.v2 as imageio


def find_frames(tweedie_dir: Path, name: str) -> list[Path]:
	pattern = re.compile(rf"^{re.escape(name)}_step_(\d+)\.png$")
	frames = []
	for p in tweedie_dir.iterdir():
		m = pattern.match(p.name)
		if m:
			frames.append((int(m.group(1)), p))
	frames.sort(key=lambda x: x[0])
	return [p for _, p in frames]


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert saved Tweedie estimates to an MP4.")
	parser.add_argument("tweedie_dir", type=Path, help="Directory with main_step_###.png frames (typically <scene>/tweedie_estimates)")
	parser.add_argument("--output", type=Path, default=None, help="Output mp4 path (default: <tweedie_dir>/../<name>_tweedie.mp4)")
	parser.add_argument("--name", type=str, default="main", help="Frame prefix (default: main)")
	parser.add_argument("--fps", type=int, default=8, help="Frames per second (default: 8)")
	args = parser.parse_args()

	frames = find_frames(args.tweedie_dir, args.name)
	if not frames:
		sys.exit(f"No frames matching {args.name}_step_###.png in {args.tweedie_dir}")

	output = args.output or (args.tweedie_dir.parent / f"{args.name}_tweedie.mp4")
	output.parent.mkdir(parents=True, exist_ok=True)

	writer = imageio.get_writer(
		str(output),
		fps=args.fps,
		codec="libx264",
		pixelformat="yuv420p",
		macro_block_size=2,
	)
	try:
		for f in frames:
			writer.append_data(imageio.imread(f))
	finally:
		writer.close()

	print(f"Wrote {output} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
	main()

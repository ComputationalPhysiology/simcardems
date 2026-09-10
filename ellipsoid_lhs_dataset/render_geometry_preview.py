#!/usr/bin/env python3
"""Render sampled LV geometry parameters as an animated preflight GIF."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def scaled(value, maximum, pixels):
    return int(value / maximum * pixels)


def half_ellipse(draw, center, long_radius, short_radius, color, width=4):
    """Draw the truncated LV-like left half of an ellipse."""
    cx, cy = center
    box = (cx - long_radius, cy - short_radius, cx + long_radius, cy + short_radius)
    draw.arc(box, 90, 270, fill=color, width=width)
    draw.line((cx, cy - short_radius, cx, cy + short_radius), fill=color, width=width)


def frame(case_path, index, total):
    case = json.loads(case_path.read_text())
    geometry = case["geometry"]
    stimulus = case["stimulus"]
    image = Image.new("RGB", (900, 620), "#f7f7f4")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    title_font = ImageFont.load_default(size=27)

    draw.text((35, 25), f"LV Latin-hypercube geometry {index + 1}/{total}", fill="#171717", font=title_font)
    draw.text((35, 64), "PARAMETRIC PREFLIGHT - not a solved field", fill="#a33b20", font=font)

    max_long = max(24.0, geometry["r_long_epi"] + 1.0)
    long_outer = scaled(geometry["r_long_epi"], max_long, 300)
    long_inner = scaled(geometry["r_long_endo"], max_long, 300)
    short_outer = scaled(geometry["r_short_epi"], max_long, 300)
    short_inner = scaled(geometry["r_short_endo"], max_long, 300)
    center = (390, 335)

    half_ellipse(draw, center, long_outer, short_outer, "#1f77b4", 6)
    half_ellipse(draw, center, long_inner, short_inner, "#d62728", 6)
    depth = scaled(stimulus["depth"], max_long, 300)
    apex_x = center[0] - long_outer
    draw.rectangle((apex_x, center[1] - 9, apex_x + depth, center[1] + 9), fill="#2ca02c")

    draw.text((60, 535), "outer wall", fill="#1f77b4", font=font)
    draw.text((215, 535), "inner wall", fill="#d62728", font=font)
    draw.text((365, 535), "stimulated apex", fill="#2ca02c", font=font)

    values = [
        f"long:  {geometry['r_long_endo']:.2f} to {geometry['r_long_epi']:.2f} mm",
        f"short: {geometry['r_short_endo']:.2f} to {geometry['r_short_epi']:.2f} mm",
        f"wall thickness: {geometry['wall_thickness_long']:.2f} / {geometry['wall_thickness_short']:.2f} mm",
        f"fiber angles: {geometry['fiber_angle_endo']:.1f}° / {geometry['fiber_angle_epi']:.1f}°",
        f"stimulus depth: {stimulus['depth']:.2f} mm",
    ]
    for line_index, line in enumerate(values):
        draw.text((610, 185 + 48 * line_index), line, fill="#262626", font=font)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Dry-run or completed dataset directory")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cases = sorted(args.results.glob("sample_*/case.json"))
    if not cases:
        raise RuntimeError(f"No sample_*/case.json files found below {args.results}")
    images = [frame(path, index, len(cases)) for index, path in enumerate(cases)]
    output = args.output or args.results / "geometry_preview.gif"
    images[0].save(output, save_all=True, append_images=images[1:], duration=1200, loop=0)
    print(output)


if __name__ == "__main__":
    main()

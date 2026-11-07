#!/usr/bin/env python3
"""
Standalone image obfuscator.

Usage:
    python obfuscate.py <image_path> [output_dir] [--method balanced|blur|pixelate]
    
Examples:
    python obfuscate.py photo.jpg
    python obfuscate.py photo.jpg ./out --method pixelate
    python obfuscate.py ./scans/*.png ./thumbs --method balanced
"""

import os
import io
import sys
import glob
import argparse
import logging
import numpy as np
from PIL import Image, ImageFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# OBFUSCATION CORE (unchanged from original pipeline)
# ============================================================================

def obfuscate_thumbnail(img, method="balanced", output_size=(300, 300)):
    """Apply irreversible obfuscation to an image and return a BytesIO buffer."""
    try:
        img = img.copy()
        img.thumbnail(output_size, Image.Resampling.LANCZOS)

        if method == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=10))

        elif method == "pixelate":
            small = img.resize((40, 40), Image.Resampling.NEAREST)
            img = small.resize(img.size, Image.Resampling.NEAREST)

        elif method == "balanced":
            img = img.filter(ImageFilter.GaussianBlur(radius=0.05))
            small = img.resize((120, 120), Image.Resampling.BILINEAR)
            img = small.resize(output_size, Image.Resampling.NEAREST)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            np_img = np.array(img)
            noise = np.random.randint(0, 20, np_img.shape, dtype=np.uint8)
            np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(np_img)

        else:
            raise ValueError(f"Unknown obfuscation method: {method}")

        if img.mode != 'RGB':
            img = img.convert('RGB')

        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        buffer.seek(0)

        size = buffer.getbuffer().nbytes
        logger.info(f"Created obfuscated buffer: {size} bytes")
        if size == 0:
            logger.error("Buffer is empty!")
            return None
        return buffer

    except Exception as e:
        logger.error(f"obfuscate_thumbnail failed: {e}")
        import traceback; traceback.print_exc()
        return None


def create_obfuscated_thumbnail_from_image(original_image_path, method="balanced"):
    """Create obfuscated thumbnail from an image file path."""
    try:
        if not os.path.exists(original_image_path):
            logger.error(f"Image file does not exist: {original_image_path}")
            return None

        logger.info(f"Opening image: {original_image_path}")
        with Image.open(original_image_path) as img:
            logger.info(f"Image opened. Size={img.size}, Mode={img.mode}")
            return obfuscate_thumbnail(img, method=method)
    except Exception as e:
        logger.error(f"Error creating thumbnail from image: {e}")
        import traceback; traceback.print_exc()
        return None


# ============================================================================
# CLI
# ============================================================================

def obfuscate_file(image_path, output_dir, method="balanced", suffix="_obfuscated"):
    """Obfuscate a single file and write it to output_dir. Returns output path or None."""
    os.makedirs(output_dir, exist_ok=True)

    buffer = create_obfuscated_thumbnail_from_image(image_path, method=method)
    if buffer is None:
        return None

    base = os.path.splitext(os.path.basename(image_path))[0]
    output_name = f"{base}{suffix}.jpg"
    output_path = os.path.join(output_dir, output_name)

    with open(output_path, 'wb') as f:
        f.write(buffer.getbuffer())

    logger.info(f"Saved obfuscated image: {output_path}")
    return output_path


def expand_inputs(inputs):
    """Expand any glob patterns into a list of existing file paths."""
    resolved = []
    for item in inputs:
        if os.path.isdir(item):
            # treat as directory: grab everything image-like
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.tiff"):
                resolved.extend(sorted(glob.glob(os.path.join(item, ext))))
        elif any(ch in item for ch in "*?["):
            resolved.extend(sorted(glob.glob(item)))
        else:
            resolved.append(item)
    # De-dup while preserving order
    seen = set()
    out = []
    for p in resolved:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Obfuscate one or more image files and save the results to a directory."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image file(s), directory(ies), or glob pattern(s) to obfuscate.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="obfuscated",
        help="Directory to write obfuscated images into. Default: ./obfuscated",
    )
    parser.add_argument(
        "-m", "--method",
        choices=["balanced", "blur", "pixelate"],
        default="balanced",
        help="Obfuscation method. Default: balanced",
    )
    parser.add_argument(
        "-s", "--suffix",
        default="_obfuscated",
        help="Suffix appended to the output filename. Default: _obfuscated",
    )

    args = parser.parse_args()

    files = expand_inputs(args.inputs)
    if not files:
        logger.error("No input files matched.")
        sys.exit(1)

    logger.info(f"Found {len(files)} file(s) to process.")
    logger.info(f"Output directory: {os.path.abspath(args.output_dir)}")
    logger.info(f"Method: {args.method}")

    successes, failures = 0, 0
    for path in files:
        if not os.path.isfile(path):
            logger.warning(f"Skipping (not a file): {path}")
            failures += 1
            continue

        result = obfuscate_file(
            path,
            args.output_dir,
            method=args.method,
            suffix=args.suffix,
        )
        if result:
            successes += 1
        else:
            failures += 1

    logger.info(f"Done. Successes: {successes}, Failures: {failures}")
    sys.exit(0 if failures == 0 else 2)


if __name__ == "__main__":
    main()
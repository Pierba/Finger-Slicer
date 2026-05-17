import argparse
import config as cfg
import cv2
import numpy as np
import os
from pathlib import Path
import torch
from typing import Iterable

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Returns the appropriate (best) device (GPU, MPS, or CPU) for PyTorch operations.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="segment_objects.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "image",
        help="Input image (JPG, PNG, …)",
    )
    p.add_argument(
        "--model_type",
        choices=["yolo", "sam"],
        help="Model type to use for segmentation",
        default="yolo"
    )
    p.add_argument(
        "-i",
        "--interactive",
        help="Interactive mode (SAM2 click-to-segment); only valid with --model_type sam",
        action="store_true",
    )
    p.add_argument(
        "--output",
        default=cfg.DEFAULT_OUTPUT_DIR,
        help=f"Output directory  (default: {cfg.DEFAULT_OUTPUT_DIR})"
    )
    p.add_argument(
        "--conf",
        type=float,
        default=cfg.DEFAULT_CONF,
        help=f"[auto] confidence threshold  (default: {cfg.DEFAULT_CONF}).  \
                Flat-lay/top-down product photos score 0.05-0.15;       \
                raise to 0.25+ for standard scene photos."
    )
    p.add_argument(
        "--yolo-model",
        default=cfg.DEFAULT_YOLO_MODEL,
        help=f"[auto] YOLO26 seg weights  (default: {cfg.DEFAULT_YOLO_MODEL})\n \
                Alternatives: yolo26l-seg.pt  yolo26m-seg.pt  yolo26s-seg.pt"
    )
    p.add_argument(
        "--sam-model",
        default=cfg.DEFAULT_SAM_MODEL,
        help=f"[sam] SAM2 weights  (default: {cfg.DEFAULT_SAM_MODEL})\n \
                Higher quality: sam2.1_l.pt"
    )

    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Image / mask utilities
# ─────────────────────────────────────────────────────────────────────────────
def upscale_mask(raw: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """
    Resize a float/uint8 raw mask to *target_wh* (W, H) and re-binarise.
    Returns a uint8 mask with values 0 or 255.
    """
    resized = cv2.resize(
        (raw > 0.5).astype(np.uint8) * 255,
        target_wh,
        interpolation=cv2.INTER_LINEAR,
    )
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological clean-up:
      1. Close  – fills small holes inside the object
      2. Open   – removes isolated specks outside the object
      3. Largest component – drops disconnected fragments
    """
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_large, iterations=2)
    out = cv2.morphologyEx(out,  cv2.MORPH_OPEN,  k_small, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n_labels > 2:                          # 0 = background
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.where(labels == largest, np.uint8(255), np.uint8(0))

    return out


def mask_to_rgba_crop(img_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply *mask* as the alpha channel of *img_bgr* (transparent background),
    then crop tightly to the non-zero region.
    Returns an RGBA ndarray, or None if the mask is empty.
    """
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    cropped = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return cropped


def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    """Return a copy of *base* with a translucent coloured *mask* overlay."""
    out = base.copy()
    out[mask > 0] = (
        out[mask > 0] * (1 - alpha) + np.array(color, np.float32) * alpha
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# SAVE IMAGES [INTERACTIVE MODE]
# ─────────────────────────────────────────────────────────────────────────────
def save_interactive_images(crop_rgba: np.ndarray, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.png")
    if not cv2.imwrite(path, crop_rgba):
        raise OSError(f"Failed to write PNG: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# SAVE IMAGES [AUTO MODE]
# ─────────────────────────────────────────────────────────────────────────────
def save_auto_images(object_stream: Iterable[tuple[str, np.ndarray]], output_dir: Path):
    """
    Saves a stream of images to the specified output directory after user confirmation.
    """

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    def create_checkerboard(h: int, w: int, square_size: int = 20) -> np.ndarray:
        """
        Creates a checkerboard pattern image of the specified height and width.
        The square_size parameter controls the size of the squares in the checkerboard.
        Returns a BGR image (3 channels) with a checkerboard pattern.
        """

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        checker = ((x // square_size) + (y // square_size)) % 2
        
        # Gray background
        bg = np.full((h, w, 3), 180, dtype=np.uint8) 
        # White squares
        bg[checker == 1] = 255 
        
        return bg

    for filename, img in object_stream:
        # Get the BGR channels and the Alpha channel
        bgr = img[:, :, :3]
        # Normalize the alpha channel to be between 0 and 1 for blending
        alpha = img[:, :, 3] / 255.0
        # Adjust the alpha channel dimensions for broadcasting
        alpha_3d = np.expand_dims(alpha, axis=2)
        
        # Create a checkerboard background
        h, w = img.shape[:2]

        
        bg = create_checkerboard(h, w)
        
        # Blend the object with the checkerboard background using the alpha channel as a mask.
        display_img = (bgr * alpha_3d + bg * (1 - alpha_3d)).astype(np.uint8)

        # Show image
        cv2.imshow("Save in transparent png? (y/n)", display_img)
        
        # Wait keystroke for user input
        key = cv2.waitKey(0) & 0xFF

        # Save
        if key == ord('y'):
            cv2.imwrite(output_dir / filename, img)
            print("Image saved.")
        else:
            print("Image skipped.")
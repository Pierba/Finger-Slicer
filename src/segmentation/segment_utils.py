import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch

from config import *

# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_device() -> torch.device:
    """
    Returns the appropriate (best) device (GPU, MPS, or CPU) for PyTorch operations.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "image",
        type=Path,
        help="Input image (JPG, PNG, ...)"
    )
    p.add_argument(
        "--model-type",
        choices=["yolo", "sam"],
        help="Model type to use for segmentation",
        default="yolo"
    )
    p.add_argument(
        "-i",
        "--interactive",
        help="Interactive mode (SAM2 click-to-segment); only valid with --model-type sam",
        action="store_true"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory  (default: {DEFAULT_OUTPUT_DIR})"
    )
    p.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"[auto] confidence threshold  (default: {DEFAULT_CONF}).  \
                Flat-lay/top-down product photos score 0.05-0.15;       \
                raise to 0.25+ for standard scene photos."
    )
    p.add_argument(
        "--yolo-model",
        type=Path,
        default=DEFAULT_YOLO_MODEL,
        help=f"[auto] YOLO26 seg weights  (default: {DEFAULT_YOLO_MODEL})\n \
                Alternatives: yolo26l-seg.pt  yolo26m-seg.pt  yolo26s-seg.pt"
    )
    p.add_argument(
        "--sam-model",
        type=Path,
        default=DEFAULT_SAM_MODEL,
        help=f"[sam] SAM2 weights  (default: {DEFAULT_SAM_MODEL})\n \
                Higher quality: sam2.1_l.pt"
    )

    return p.parse_args()

# =============================================================================
# Image / mask utilities
# =============================================================================

def upscale_mask(raw: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    """
    Resize a float/uint8 raw mask to `target_wh` (W, H) and re-binarise.
    Returns a uint8 mask with values [0, 255].
    """ 

    resized = cv2.resize(
        (raw > 0.5).astype(np.uint8) * 255, # Conversion to binary array multiplied by 255 to get values in [0, 255]
        target_wh,                          # Scales up to the original image size (W, H)
        interpolation=cv2.INTER_LINEAR,     # Linear interpolation for better quality when resizing masks
    )
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY) # Re-binarise to ensure clean edges after interpolation
    
    return binary

def refine_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological clean-up:
      1. Close  - fills small holes inside the object
      2. Open   - removes isolated specks outside the object
      3. Largest component - drops disconnected fragments
    """

    # Elliptical elements created to clear the possible noise 
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # For filling holes
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # For small noise

    # Fills small holes inside objects. iterations=2 repeats the operation twice to strengthen the effect
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_large, iterations=2)
    
    # Removes small isolated foreground points
    out = cv2.morphologyEx(out,  cv2.MORPH_OPEN,  k_small, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n_labels > 2:        # 0 = background
        # Find the index of the largest component (by area) excluding the background (index 0)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

        # Build a binary mask where pixels belonging to the chosen 
        # largest label become 255 (foreground) and all others 0
        out = np.where(labels == largest, np.uint8(255), np.uint8(0))

    return out

def mask_to_rgba_crop(img_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply `mask` as the alpha channel of `img_bgr`,
    then crop tightly to the non-zero region.
    Returns an RGBA ndarray or None if the mask is empty.
    """

    # Converts the BGR image to RGBA format
    rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

    # Applies the mask to the alpha channel of the RGBA image
    rgba[:, :, 3] = mask

    # Retrieves the lists of coordinates of the non-zero pixels in the mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    
    # Crops the RGBA image within the coordinates of non-zero pixels of the mask for best precision
    cropped = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    return cropped

def overlay_mask(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    """
    Return a copy of `base` with a translucent coloured `mask` overlay.
    """
    out = base.copy()

    # Fills masked pixels with a translucent overlay of color while preserving original pixel detail.
    out[mask > 0] = (
        out[mask > 0] * (1 - alpha) + np.array(color, np.float32) * alpha
    ).astype(np.uint8)
    
    # Creates the contours of the mask and draws them on the output image to enhance visibility of object boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    
    return out

# =============================================================================
# CANVAS UTILITIES
# =============================================================================

def fit_to_canvas(crop_rgba: np.ndarray, out_w: int = SAVE_IMG_W, out_h: int = SAVE_IMG_H) -> np.ndarray:
    """
    Scale `crop_rgba` (RGBA) to fit inside an `out_w` x `out_h` canvas while
    preserving aspect ratio, centre it, and pad the remainder with full
    transparency (alpha = 0).

    Downscaling uses INTER_AREA (best quality for shrinking).
    Upscaling uses INTER_LANCZOS4 (best quality for enlarging small objects).

    Returns a (out_h, out_w, 4) uint8 RGBA array.
    """
    h, w = crop_rgba.shape[:2]
    scale   = min(out_w / w, out_h / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    interp  = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(crop_rgba, (new_w, new_h), interpolation=interp)

    canvas       = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    x_off        = (out_w - new_w) // 2
    y_off        = (out_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas

# =============================================================================
# INTERACTIVE SESSION STATE
# =============================================================================

@dataclass
class InteractiveState:
    """All mutable state for the interactive session."""
    pos_pts:      list[tuple[int, int]] = field(default_factory=list)
    neg_pts:      list[tuple[int, int]] = field(default_factory=list)
    current_mask: Optional[np.ndarray] = None
    obj_count:    int = 0
    color_idx:    int = 0
    status:       str = "Left-click an object to start  |  right-click to exclude"


# =============================================================================
# SAVE IMAGES [INTERACTIVE MODE]
# =============================================================================
def save_interactive_images(crop_rgba: np.ndarray, output_dir: Path, name: str) -> str:
    """
    Fit the crop onto a fixed canvas, then write it as a transparent PNG.
    Returns the path to the saved PNG.
    """
    canvas = fit_to_canvas(crop_rgba)
   
    path   = output_dir / f"{name}.png"
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Failed to write PNG: {path}")
    
    return path

# =============================================================================
# SAVE IMAGES [AUTO MODE]
# =============================================================================

def save_auto_images(object_stream: Iterable[tuple[str, np.ndarray]], output_dir: Path):
    """
    For each (filename, crop_rgba) pair:
      1. Fit the crop onto a fixed SAVE_IMG_W x SAVE_IMG_H canvas 
         (aspect-ratio preserving, centred, transparent padding) via fit_to_canvas().
      2. Preview the canvas blended over a checkerboard in a fixed-size window —
         no more giant preview windows regardless of the source image resolution.
      3. On Y, write the fixed-size canvas as a transparent PNG; any other key
         skips it.
    """
    def _create_checkerboard(h: int, w: int, square_size: int = 20) -> np.ndarray:
        """
        Return a (h, w, 3) BGR checkerboard used as a transparency stand-in
        when previewing RGBA crops before saving.
        """
        x, y   = np.meshgrid(np.arange(w), np.arange(h))
        checker = ((x // square_size) + (y // square_size)) % 2
        bg = np.full((h, w, 3), 180, dtype=np.uint8)
        bg[checker == 1] = 255
        return bg
    
    for filename, img in object_stream:
        # == normalise to fixed canvas ============================
        canvas = fit_to_canvas(img)           # always SAVE_IMG_H × SAVE_IMG_W × 4

        # == build checkerboard preview ===========================
        h, w    = canvas.shape[:2]
        bgr     = canvas[:, :, :3]
        alpha   = canvas[:, :, 3] / 255.0
        alpha3  = np.expand_dims(alpha, axis=2)
        bg      = _create_checkerboard(h, w)
        preview = (bgr * alpha3 + bg * (1 - alpha3)).astype(np.uint8)

        # == show & confirm ========================================
        win_title = f"Save as '{filename}'? Y = save |ESC = stop |Other keys = skip"
        cv2.imshow(win_title, preview)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(win_title)

        if key == 27:                 # ESC — stop immediately
            print("\t X  review cancelled.")
            break
        elif key == ord('y'):
            path   = output_dir / f"{filename}.png"
            cv2.imwrite(str(path), canvas)
            print(f"\t V  saved  {filename}")
        else:
            print(f"\t -  skipped  {filename}")
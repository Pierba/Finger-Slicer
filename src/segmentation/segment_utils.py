"""
Object Segmentation Utilities
======================================
This module contains utility functions and classes for object segmentation tasks, including:
    - Device detection for PyTorch (GPU, MPS, CPU)
    - Command-line argument parsing for segmentation scripts
    - Image and mask processing helpers (resizing, refining, overlaying)
    - Canvas fitting for consistent output image sizes
    - State management class for interactive segmentation sessions
    - Functions for saving segmented objects as transparent PNGs
"""
import argparse
from   config       import *
import cv2
from   dataclasses  import dataclass, field
import numpy as np
from   pathlib      import Path
import torch
from   typing       import Iterable, Optional

# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_device() -> torch.device:
    """
    Returns:
        the appropriate device (GPU, MPS, or CPU based on user hardware) to be used for model results.
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
    """
    Returns:
        the arguments passed by the user with the CLI, used for segmentation script.
    """
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
    Resizes a float/uint8 raw `mask` to `target_wh` dimensions and makes it binary.

    Args:
        raw: a float or uint8 mask (H, W) with values in [0, 1] or [0, 255].
        target_wh: desired scale for the output mask.

    Returns:
        A uint8 binary mask (H, W) with values 0 or 255, scaled to target_wh.
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
    Clean up a binary `mask` by filling holes, removing small noise, and keeping only the largest connected component.

    Args:
        mask: a binary uint8 mask (H, W) with values in [0, 255].

    Returns:
        A cleaned binary uint8 mask (H, W) with values in [0, 255].
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
    Applies a binary `mask` to a BGR image (`img_bgr`) and returns a cropped RGBA image
    where the masked area is visible while the rest is transparent.

    Args:
        img_bgr: a BGR image (H, W, 3) as a uint8 array.
        mask: a binary uint8 mask (H, W) with values in [0, 255].

    Returns:
        A cropped RGBA image (h, w, 4) where the masked area is visible and the rest is transparent.
        Returns None if the mask has no non-zero pixels.
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
    Returns a BGR image with a binary `mask` overlaid on top of a `base` image, using a given `color` and `alpha` transparency.

    Args:
        base: a BGR image (H, W, 3) as a uint8 array.
        mask: a binary uint8 mask (H, W) with values in [0, 255].
        color: a tuple of (B, G, R) values for the overlay color.
        alpha: a float in [0, 1] representing the transparency of the overlay (default: 0.45).

    Returns:
        A BGR image (H, W, 3) with the mask area overlaid in the specified color and transparency.
    """
    out = base.copy()

    # Fills masked pixels with a translucent overlay of color while preserving original pixel detail
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

def fit_to_canvas(crop_rgba: np.ndarray, out_w: int = PROJECTILE_MAX_SIZE, out_h: int = PROJECTILE_MAX_SIZE) -> np.ndarray:
    """
    Scales `crop_rgba` (RGBA) to fit inside an `out_w` x `out_h` canvas while preserving its aspect ratio.
    Downscaling uses INTER_AREA (best quality for shrinking).
    Upscaling uses INTER_LANCZOS4 (best quality for enlarging small objects).

    Args:
        crop_rgba: an RGBA image (h, w, 4) as a uint8 array.
        out_w: output width of the canvas (default: PROJECTILE_MAX_SIZE).
        out_h: output height of the canvas (default: PROJECTILE_MAX_SIZE).

    Returns:
        A RGBA image (out_h, out_w, 4) as uint8 array.
    """
    # Calculating the scaling factor
    h, w = crop_rgba.shape[:2]
    scale   = min(out_w / w, out_h / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)

    # Choosing interpolation method based on whether we are upscaling or downscaling
    interp  = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(crop_rgba, (new_w, new_h), interpolation=interp)

    # Creating a transparent canvas and centering the resized image applied to it
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
    """
    All mutable variables for the interactive session.
    """
    pos_pts:      list[tuple[int, int]] = field(default_factory=list)       # List of (x, y) coordinates for object points (positive clicks)
    neg_pts:      list[tuple[int, int]] = field(default_factory=list)       # List of (x, y) coordinates for background points (negative clicks)
    current_mask: Optional[np.ndarray] = None                               # The current mask being previewed
    obj_count:    int = 0                                                   # Counter for segmented objects
    color_idx:    int = 0                                                   # Index for cycling through overlay colors in the palette
    status:       str = "Left-click to include  |  Right-click to exclude"  # Current status message to display on the window


# =============================================================================
# SAVE IMAGES [INTERACTIVE MODE]
# =============================================================================
def save_interactive_images(crop_rgba: np.ndarray, name: str) -> str:
    """
    Fits the `crop_rgba` onto a fixed PROJECTILE_MAX_SIZE x PROJECTILE_MAX_SIZE canvas and
    saves it as a transparent PNG in the assets directory with the given `name`.

    Args:
        crop_rgba: an RGBA image (h, w, 4) as a uint8 array to be saved.
        name: the filename (without extension) for the saved PNG.

    Returns:
        The path to the saved PNG file.
    """
    canvas = fit_to_canvas(crop_rgba)

    path   = DEFAULT_OUTPUT_DIR / f"{name}.png"
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Failed to write PNG: {path}")

    return path

# =============================================================================
# SAVE IMAGES [AUTO MODE]
# =============================================================================

def save_auto_images(object_stream: Iterable[tuple[str, np.ndarray]]):
    """
    For each (filename, crop_rgba) pair in `object_stream`:
    - Fits the crop onto a fixed PROJECTILE_MAX_SIZE x PROJECTILE_MAX_SIZE canvas.
    - Shows the canvas blended over a checkerboard in a fixed-size window.
    - On Y, saves the canvas as a transparent PNG in the assets directory, else it skips it.

    Args:
        object_stream: an iterable of (filename, crop_rgba) pairs.
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
        canvas = fit_to_canvas(img)           # always PROJECTILE_MAX_SIZE × PROJECTILE_MAX_SIZE × 4

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
            path   = DEFAULT_OUTPUT_DIR / f"{filename}.png"
            cv2.imwrite(str(path), canvas)
            print(f"\t V  saved  {filename}")
        else:
            print(f"\t -  skipped  {filename}")
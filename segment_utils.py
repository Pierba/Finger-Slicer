import argparse
import config as cfg
import cv2
import numpy as np
import os
from pathlib import Path
import torch
from typing import Iterable, Optional

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
# CANVAS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def create_checkerboard(h: int, w: int, square_size: int = 20) -> np.ndarray:
    """
    Return a (h, w, 3) BGR checkerboard used as a transparency stand-in
    when previewing RGBA crops before saving.
    """
    x, y   = np.meshgrid(np.arange(w), np.arange(h))
    checker = ((x // square_size) + (y // square_size)) % 2
    bg = np.full((h, w, 3), 180, dtype=np.uint8)
    bg[checker == 1] = 255
    return bg


def fit_to_canvas(crop_rgba: np.ndarray,
                  out_w: int = cfg.SAVE_IMG_W,
                  out_h: int = cfg.SAVE_IMG_H) -> np.ndarray:
    """
    Scale *crop_rgba* (RGBA) to fit inside an *out_w* × *out_h* canvas while
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


# ─────────────────────────────────────────────────────────────────────────────
# SAVE IMAGES [INTERACTIVE MODE]
# ─────────────────────────────────────────────────────────────────────────────
def save_interactive_images(crop_rgba: np.ndarray, out_dir: str, name: str) -> str:
    """Fit the crop onto a fixed canvas, then write it as a transparent PNG."""
    canvas = fit_to_canvas(crop_rgba)
    path   = os.path.join(out_dir, f"{name}.png")
    if not cv2.imwrite(path, canvas):
        raise OSError(f"Failed to write PNG: {path}")
    return path

# ─────────────────────────────────────────────────────────────────────────────
# SAVE IMAGES [AUTO MODE]
# ─────────────────────────────────────────────────────────────────────────────
def save_auto_images(object_stream: Iterable[tuple[str, np.ndarray]], output_dir: Path):
    """
    For each (filename, crop_rgba) pair:
      1. Fit the crop onto a fixed SAVE_IMG_W × SAVE_IMG_H canvas (aspect-ratio
         preserving, centred, transparent padding) via fit_to_canvas().
      2. Preview the canvas blended over a checkerboard in a fixed-size window —
         no more giant preview windows regardless of the source image resolution.
      3. On Y, write the fixed-size canvas as a transparent PNG; any other key
         skips it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for filename, img in object_stream:
            # ── Step 1: normalise to fixed canvas ────────────────────────────
            canvas = fit_to_canvas(img)           # always SAVE_IMG_H × SAVE_IMG_W × 4

            # ── Step 2: build checkerboard preview ───────────────────────────
            h, w    = canvas.shape[:2]
            bgr     = canvas[:, :, :3]
            alpha   = canvas[:, :, 3] / 255.0
            alpha3  = np.expand_dims(alpha, axis=2)
            bg      = create_checkerboard(h, w)
            preview = (bgr * alpha3 + bg * (1 - alpha3)).astype(np.uint8)

            # ── Step 3: show & confirm ────────────────────────────────────────
            # ESC (27) exits immediately — preferred over Ctrl+C because waitKey
            # blocks in the cv2 window; ESC is caught there regardless of focus.
            # KeyboardInterrupt (Ctrl+C) is handled below as a terminal fallback
            # for when focus is on the console between windows.
            win_title = f"Save as '{filename}'? Y = save |ESC = stop |Other keys = skip"
            cv2.imshow(win_title, preview)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(win_title)

            if key == 27:                 # ESC — stop immediately
                print("  ✗  review cancelled.")
                break
            elif key == ord('y'):
                cv2.imwrite(str(output_dir / filename), canvas)
                print(f"  ✓  saved  {filename}")
            else:
                print(f"  –  skipped  {filename}")

    except KeyboardInterrupt:
        # Ctrl+C fallback: fired when the terminal has focus between windows.
        cv2.destroyAllWindows()
        print("\n  ✗  review interrupted (Ctrl+C).")
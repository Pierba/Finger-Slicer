"""
Object Segmentation & Extraction Tool
======================================
Modes
-----
  auto        YOLO26x-seg detects & segments objects automatically.
              Each object is saved as a transparent PNG via its
              pixel-precise instance mask (background fully removed).

  interactive SAM2 point-click segmentation.
              Click on any object in the window; SAM2 generates the
              mask; press Enter to save a transparent PNG.

Usage
-----
  python segment_objects.py photo.jpg                  # interactive (default)
  python segment_objects.py photo.jpg --mode auto
  python segment_objects.py photo.jpg --mode auto --conf 0.1
  python segment_objects.py photo.jpg --output my_dir
  python segment_objects.py photo.jpg --yolo-model yolo26l-seg.pt

YOLO26 notes
------------
  YOLO26 is NMS-free (end-to-end), 43 % faster on CPU than YOLO11,
  and includes dedicated instance-segmentation improvements
  (semantic segmentation loss, multi-scale proto modules).
  Weights are downloaded automatically on first run.
  Docs: https://docs.ultralytics.com/models/yolo26
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics import SAM

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_YOLO_MODEL = "yolo26x-seg.pt"
DEFAULT_SAM_MODEL  = "sam2.1_b.pt"
DEFAULT_OUTPUT_DIR = "output_objects"
DEFAULT_CONF       = 0.05   # flat-lay / product photos score lower; 0.25 misses most objects

# BGR colours for mask overlays / HUD dots
PALETTE: list[tuple[int, int, int]] = [
    ( 72, 199, 142),  # teal
    (255, 159,  64),  # orange
    (100, 149, 237),  # cornflower
    (255,  99, 132),  # pink-red
    (153, 102, 255),  # purple
    (255, 205,  86),  # yellow
]


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

# serve davvero?
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

    # qui prende la maschera al posto del box
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


def save_png(crop_rgba: np.ndarray, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.png")
    if not cv2.imwrite(path, crop_rgba):
        raise OSError(f"Failed to write PNG: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# HUD rendering
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(canvas: np.ndarray,
             lines: list[str],
             start_y: int = 20) -> None:
    """Render text lines with a black backing rectangle onto *canvas* in-place."""
    font, scale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 4
    y = start_y
    for line in lines:
        (tw, th), base = cv2.getTextSize(line, font, scale, thick)
        cv2.rectangle(canvas, (8, y - th - pad), (12 + tw, y + base + pad),
                      (0, 0, 0), -1)
        cv2.putText(canvas, line, (10, y), font, scale,
                    (220, 220, 220), thick, cv2.LINE_AA)
        y += th + base + pad + 4


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 – AUTO  (YOLO26-seg instance segmentation)
# ─────────────────────────────────────────────────────────────────────────────

def run_auto(image_path: str, out_dir: str, conf: float, model_name: str):
    """
    Detect and segment objects with YOLO26x-seg.
    For each detection the pipeline is:
      raw mask -> upscale -> binarise -> morphological refinement
      -> alpha crop (transparent background) -> save PNG
    A composite preview (_preview_auto.png) is also written.
    """
    print(f"Loading {model_name} …")
    model = YOLO(model_name)

    print(f"Segmenting '{image_path}'  (conf ≥ {conf}) …\n")
    result = model(image_path, conf=conf, verbose=False)[0]

    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot read image: {image_path}")

    H, W = img.shape[:2]
    os.makedirs(out_dir, exist_ok=True)

    if result.masks is None:
        print("No segmentation masks returned.")
        return

    # extracted the numpy arrays from result
    masks_raw = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    label_counts: dict[str, int] = {}
    preview = img.copy()

    # saved is useless because collects data never used, only used for len() ?
    # saved: list[tuple[str, float, str]] = []

    for idx, (mask_raw, box, cls) in enumerate(zip(masks_raw, boxes, classes)):
        # ── metadata ─────────────────────────────────────────────────────
        label  = result.names[int(cls)]         # get class name from model's names list
        conf_v = float(box.conf[0])             # confidence value
        color  = PALETTE[idx % len(PALETTE)]    # changing palette colour for each object

        # ── mask pipeline ────────────────────────────────────────────────
        mask = upscale_mask(mask_raw, (W, H))
        # mask = refine_mask(mask) this refine seems useless to me and also overkill for our project ?

        # crop image as we used to 
        crop = mask_to_rgba_crop(img, mask)
        if crop is None:
            print(f"  skip  {label}: empty mask after refinement")
            continue

        # ── save ─────────────────────────────────────────────────────────
        label_counts[label] = label_counts.get(label, 0) + 1
        name = f"{label}_{label_counts[label]}"
        path = save_png(crop, out_dir, name)
        # saved.append((name, conf_v, path))
        print(f"  ✓  {name:<32}  conf={conf_v:.2f}  →  {path}")

        # ── annotate preview ─────────────────────────────────────────────
        preview = overlay_mask(preview, mask, color=color, alpha=0.40)
        x1, y1 = map(int, box.xyxy[0][:2])
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(preview, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(preview, name, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    prev_path = os.path.join(out_dir, "_preview_auto.png")
    cv2.imwrite(prev_path, preview)

    # print(f"\n✅  {len(saved)} object(s) saved to '{out_dir}/'")
    print(f"   Preview → {prev_path}")

    # if not saved:
    #     print("\n⚠  Nothing detected — try lowering --conf (e.g. --conf 0.1)")

# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 – INTERACTIVE  (SAM2 click-to-segment)
# ─────────────────────────────────────────────────────────────────────────────

# class made to keep tracking the session variables
@dataclass # data class annotation used to get rid of __init__() function essentially
class _InteractiveState:
    """All mutable state for the interactive session."""
    # this field() here is used to prevent that other instances can access the same list ?
    pos_pts:      list[tuple[int, int]]   = field(default_factory=list)
    neg_pts:      list[tuple[int, int]]   = field(default_factory=list)
    current_mask: Optional[np.ndarray] = None
    obj_count:    int = 0
    color_idx:    int = 0
    status:       str = "Left-click an object to start  |  right-click to exclude"


def run_interactive(image_path: str, out_dir: str, sam_name: str):
    """
    SAM2 interactive segmentation via mouse clicks.

    Controls
    --------
    Left-click   Positive point (include in object)
    Right-click  Negative point (exclude / background)
    S            Run SAM2 segmentation with current points
    Enter        Save current mask as transparent PNG
    N            Clear points & mask - retry current object
    U            Undo last point
    Q / Esc      Quit
    """
    print(f"Loading {sam_name} …")
    model = SAM(sam_name)

    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot read image: {image_path}")

    H, W = img.shape[:2]
    os.makedirs(out_dir, exist_ok=True)

    # ── window & state ────────────────────────────────────────────────────
    state = _InteractiveState()             # could be also a dict() in my opinion ?
    WIN   = "SAM2 Interactive Segmentation" # window title
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL) # window creation with adjustable size

    # ── drawing the window ───────────────────────────────────────────────────────────
    def redraw() -> None:
        disp = img.copy()
        # Display current mask overlay if available
        if state.current_mask is not None:
            color = PALETTE[state.color_idx % len(PALETTE)]
            disp  = overlay_mask(disp, state.current_mask, color=color)
        
        # Display points
        for p in state.pos_pts:
            # Drawing green dots
            cv2.circle(disp, p, 6, (0, 230,  80), -1)
            # cv2.circle(disp, tuple(p), 8, (255, 255, 255),  2)
        for p in state.neg_pts:
            # Drawing red dots
            cv2.circle(disp, p, 6, (30,  30, 220), -1)
            # cv2.circle(disp, tuple(p), 8, (255, 255, 255),  2) useless white border for the dots, maybe not needed ? 
        
        # Display HUD actions and status string
        draw_hud(disp, [
            f"Object #{state.obj_count + 1}   saved: {state.obj_count}",
            "● green = include   ● blue = exclude",
            "[S] segment   [Enter] save   [N] clear   [U] undo   [Q] quit",
            f"→  {state.status}",
        ])

        # Show the updated image in the window
        cv2.imshow(WIN, disp)

    # ── mouse events ─────────────────────────────────────────────────────────────
    def on_mouse(event: int, x: int, y: int, _flags, _param) -> None:
        match event:
            # With left click add a positive dot
            case cv2.EVENT_LBUTTONDOWN:
                state.pos_pts.append((x, y))
                state.current_mask = None
                state.status = f"+ positive ({x},{y}) — press S to segment"
            # With right click add a negative dot
            case cv2.EVENT_RBUTTONDOWN:
                state.neg_pts.append((x, y))
                state.current_mask = None
                state.status = f"- negative ({x},{y}) — press S to segment"
        redraw()

    # Register mouse callback for the interactive window (like event listener) ?
    cv2.setMouseCallback(WIN, on_mouse)

    print("\n" + "─" * 56)
    print("  Interactive SAM2 Segmentation")
    print("─" * 56)
    print("  Left-click   mark OBJECT  (green)")
    print("  Right-click  mark BACKGROUND  (blue)")
    print("  S            segment")
    print("  Enter        save as transparent PNG")
    print("  N            clear & retry")
    print("  U            undo last point")
    print("  Q / Esc      quit")
    print("─" * 56 + "\n")
    redraw()

    # ── event loop ────────────────────────────────────────────────────────
    while True:
        # register key pressed in the window, with a small delay to allow redraw
        key = cv2.waitKey(40) & 0xFF

        # I have tried with match case but it would blow up because of ord() calls and too much indentation ?
        if key in (ord('q'), 27):   # 'q' or Esc key
            break
        elif key == ord('s'):
            _do_segment(image_path, model, state, W, H)
            redraw()
        elif key in (13, 10):       # Enter key (13 on Windows, 10 on Unix)
            _do_save(img, state, out_dir)
            redraw()
        elif key == ord('n'):       # 'n' for cleaning current selections
            state.pos_pts.clear()
            state.neg_pts.clear()
            state.current_mask = None
            state.status = "Cleared — click a new object"
            redraw()
        elif key == ord('u'):       # 'u' for undoing the last point
            if state.neg_pts:
                state.neg_pts.pop()
            elif state.pos_pts:
                state.pos_pts.pop()
            state.current_mask = None
            state.status = "Last point removed"
            redraw()

        # When the window is closed manually it ends the session
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: # This function can return values <= 0
            break

    # Close the window and end the session
    cv2.destroyAllWindows()
    print(f"\n✅  {state.obj_count} object(s) saved to '{out_dir}/'")


def _do_segment(image_path: str, model, state: _InteractiveState, W: int, H: int):
    """Run SAM2 with current points and update state.current_mask."""
    # If there are no points we cannot proceed
    if not state.pos_pts and not state.neg_pts:
        state.status = "⚠ Add at least one point first"
        return

    # updating the status string
    state.status = "Segmenting …"

    # Creates an array of all points and corresponding labels (1 for positive, 0 for negative)
    all_pts = state.pos_pts + state.neg_pts # It concats the lists of positive and negative points into a single list of all points.
    all_lbl = [1] * len(state.pos_pts) + [0] * len(state.neg_pts)

    try:
        # Runs the SAM model (points = list of coordinates, labels = list of 1/0 for positive/negative, verobse = False to suppress model output)
        res = model(image_path, points=[all_pts], labels=[all_lbl], verbose=False)
    except Exception as exc:
        state.status = f"SAM2 error: {exc}"
        return

    if not res or res[0].masks is None:
        state.status = "No mask returned — try more / different points"
        return

    masks_data = res[0].masks.data.cpu().numpy()
    if masks_data.size == 0:
        state.status = "No mask returned — try different points"
        return

    # ABSOLUTE BLACK MAGIC ?
    best       = int(np.argmax([m.sum() for m in masks_data]))
    mask       = upscale_mask(masks_data[best], (W, H))
    # state.current_mask = refine_mask(mask) same speech as before for auto mode ?
    state.current_mask = mask
    state.status = "Mask ready — Enter to SAVE,  N to retry"


def _do_save(img: np.ndarray, state: _InteractiveState, out_dir: str):
    """Crop the current mask, prompt for a name, and save the PNG."""
    if state.current_mask is None:
        state.status = "No mask yet — press S first"
        return

    suggested = f"object_{state.obj_count:02d}"
    try:
        name = input(f"  Save as (default '{suggested}'): ").strip()
    except (EOFError, KeyboardInterrupt):
        name = ""
    
    # Fall back to default if input is empty or interrupted
    name = name or suggested

    crop = mask_to_rgba_crop(img, state.current_mask)
    if crop is None:
        print("  ⚠ Empty mask — nothing saved")
        return

    path = save_png(crop, out_dir, name)
    print(f"  ✓  Saved → {path}")

    # Clears out the state object for the next iteration and tracks the count 
    state.obj_count  += 1
    state.color_idx  += 1
    state.pos_pts.clear()
    state.neg_pts.clear()
    state.current_mask = None
    state.status = "Saved!  Click the next object."


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
        help="Input image (JPG, PNG, …)"
    )
    p.add_argument(
        "--mode", 
        choices=["auto", "interactive"],           
        default="interactive",           
        help="auto = YOLO26 seg masks | interactive = SAM2  (default: interactive)"
    )
    p.add_argument(
        "--output", 
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
        default=DEFAULT_YOLO_MODEL,           
        help=f"[auto] YOLO26 seg weights  (default: {DEFAULT_YOLO_MODEL})\n \
                Alternatives: yolo26l-seg.pt  yolo26m-seg.pt  yolo26s-seg.pt"
    )
    p.add_argument(
        "--sam-model", 
        default=DEFAULT_SAM_MODEL,
        help=f"[interactive] SAM2 weights  (default: {DEFAULT_SAM_MODEL})\n \
                Higher quality: sam2.1_l.pt"
    )

    return p.parse_args()


def main() -> None:
    args = build_parser()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    match args.mode:
        case "auto":
            print("Mode: AUTO  (YOLO26-seg instance segmentation)")
            run_auto(args.image, args.output, args.conf, args.yolo_model)
        case "interactive":
            print("Mode: INTERACTIVE  (SAM2 click-to-segment)")
            run_interactive(args.image, args.output, args.sam_model)
        case _:
            sys.exit(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()

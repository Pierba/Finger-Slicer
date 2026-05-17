"""
Object Segmentation & Extraction Tool
======================================
Modes
-----
  yolo        YOLO26x-seg detects & segments objects automatically.
              Each object is saved as a transparent PNG via its
              pixel-precise instance mask (background fully removed).

  sam         SAM2 fully-automatic segmentation (no clicks needed).
              SAM2 generates all masks for the image; small objects,
              background masks and overlapping sub-parts are filtered
              out automatically. Each surviving object is saved as a
              transparent PNG.

  interactive SAM2 point-click segmentation.
              Click on any object in the window; SAM2 generates the
              mask; press Enter to save a transparent PNG.


Usage
-----
  python segment_objects.py photo.jpg                           # yolo auto (default)
  python segment_objects.py photo.jpg --model_type sam          # SAM auto
  python segment_objects.py photo.jpg --model_type sam -i       # SAM interactive
  python segment_objects.py photo.jpg --model_type yolo --conf 0.1
  python segment_objects.py photo.jpg --output my_dir
  python segment_objects.py photo.jpg --yolo-model yolo26l-seg.pt
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics import SAM
import config as cfg

from segment_utils import *

# ── ENUMS & CONSTANTS ────────────────────────────────────────────────────────
# BGR colours for mask overlays
PALETTE: list[tuple[int, int, int]] = cfg.PALETTE

# SAM auto-mode thresholds
SMALL_OBJECT_THRESHOLD = cfg.SMALL_OBJECT_THRESHOLD
BACKGROUND_THRESHOLD = cfg.BACKGROUND_THRESHOLD
SUBPART_OVERLAP_THRESHOLD = cfg.SUBPART_OVERLAP_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
# HUD rendering
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(canvas: np.ndarray, lines: list[str], start_y: int = 20):
    """Render text lines with a black backing rectangle onto *canvas* in-place."""
    font, scale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 4
    y = start_y
    for line in lines:
        (tw, th), base = cv2.getTextSize(line, font, scale, thick)
        cv2.rectangle(canvas, (8, y - th - pad), (12 + tw, y + base + pad), (0, 0, 0), -1)
        cv2.putText(canvas, line, (10, y), font, scale, (220, 220, 220), thick, cv2.LINE_AA)
        y += th + base + pad + 4

# ─────────────────────────────────────────────────────────────────────────────
# YOLO AUTO MODE  (YOLO26-seg instance segmentation)
# ─────────────────────────────────────────────────────────────────────────────
def run_yolo_auto(image_path: str, out_dir: str, conf: float, model_name: str):
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

    if result.masks is None:
        print("No segmentation masks returned.")
        return

    # Extract numpy arrays from result.
    # FIX: confs extracted separately from result.boxes.conf — iterating over
    # result.boxes.xyxy yields plain (4,) numpy rows with no .conf attribute.
    masks_raw = result.masks.data.cpu().numpy()
    boxes     = result.boxes.xyxy.cpu().numpy()   # shape (N, 4): x1, y1, x2, y2
    confs     = result.boxes.conf.cpu().numpy()   # shape (N,):   confidence scores
    classes   = result.boxes.cls.cpu().numpy()    # shape (N,):   class indices

    # label_counts: dict[str, int] = {}
    preview  = img.copy()
    # Collect (filename, crop_rgba) pairs; user will confirm each via save_images.
    detected: list[tuple[str, np.ndarray]] = []

    for idx, (mask_raw, box, conf_v, cls) in enumerate(zip(masks_raw, boxes, confs, classes)):
        # ── metadata ─────────────────────────────────────────────────────
        label  = result.names[int(cls)]         # get class name from model's names list
        color  = PALETTE[idx % len(PALETTE)]    # cycling palette colour per object

        # ── mask pipeline ────────────────────────────────────────────────
        mask = upscale_mask(mask_raw, (W, H))

        crop = mask_to_rgba_crop(img, mask)
        if crop is None:
            print(f"  skip  {label}: empty mask")
            continue

        # ── queue for user confirmation ───────────────────────────────────
        name = f"{label}_conf:{conf_v:.2f}"
        print(f"  detected  {name:<32}  conf={conf_v:.2f}")
        detected.append((f"{name}.png", crop))

        # ── annotate preview (all detections, regardless of confirmation) ─
        preview = overlay_mask(preview, mask, color=color, alpha=0.40)
        # FIX: box is already a plain (4,) numpy array [x1, y1, x2, y2];
        # the old code called box.xyxy[0][:2] which fails on a numpy row.
        x1, y1 = map(int, box[:2])
        tw, th = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.rectangle(preview, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(preview, name, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # ── interactive save: user confirms each crop with Y / any other key ──
    print(f"\nReview {len(detected)} detected object(s) — press Y to save, any other key to skip.\n")
    save_auto_images(detected, Path(out_dir))

    # ── always write the composite preview as a reference map ─────────────
    prev_path = os.path.join(out_dir, "_preview_auto.png")
    cv2.imwrite(prev_path, preview)
    print(f"\n   Preview → {prev_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SAM AUTO MODE  (SAM2 fully-automatic segmentation)
# ─────────────────────────────────────────────────────────────────────────────

def run_sam_auto(image_path: str, out_dir: str, conf: float, model_name: str):
    """
    Automatically segment all objects in *image_path* using SAM2.

    Unlike run_auto (YOLO), SAM2 has no concept of object classes: it returns
    every mask it finds.  Three filtering passes are applied to remove noise:

      1. Size filter    – drops masks whose bounding box is smaller than
                          cfg.SMALL_OBJECT_THRESHOLD in either dimension.
      2. Background     – drops masks whose bounding box covers more than
                          cfg.BACKGROUND_THRESHOLD of the total image area.
      3. Sub-part dedup – sorts survivors by area (largest first); if a
                          smaller mask overlaps a larger one by more than
                          cfg.SUBPART_OVERLAP_THRESHOLD it is treated as a
                          sub-part of that object and discarded.

    Surviving objects are saved as transparent PNGs and a composite preview
    (_preview_sam_auto.png) is written to *out_dir*.
    """
    print(f"Loading {model_name} …")
    model = SAM(model_name)

    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Cannot read image: {image_path}")

    H, W = img.shape[:2]
    total_image_area = W * H

    print(f"Generating masks with SAM2  (conf ≥ {conf}) …")
    results = model.predict(image_path, device=get_device(), conf=conf)
    result  = results[0]

    if result.masks is None:
        print("No objects detected in the image.")
        return

    # Pull raw mask tensors to CPU numpy and free the PyTorch tensors.
    masks_data = result.masks.data.cpu().numpy()
    del results

    # ── Pass 1 & 2: size filter + background filter ───────────────────────────
    # Both checks use actual mask pixel counts, NOT bounding-box area.
    # Bounding-box area over-estimates coverage (a diagonal object fills only
    # ~50 % of its box), which caused large surfaces like a desk to slip through
    # the background filter and then incorrectly absorb every object on top of
    # it during sub-part deduplication.
    print("Filtering small objects and background …")
    candidates = []
    for mask_data in masks_data:
        # SAM scales masks down internally; resize back to original dimensions.
        # Values become 0 (background) or 255 (object).
        mask_uint8 = (mask_data * 255).astype(np.uint8)
        alpha = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)

        x, y, w, h = cv2.boundingRect(alpha)
        if w == 0 or h == 0:
            continue

        # Pass 1 – minimum size: bounding-box dimensions are fine here
        # (a mask too small in either direction is noise regardless of shape).
        if w <= SMALL_OBJECT_THRESHOLD or h <= SMALL_OBJECT_THRESHOLD:
            continue

        # Pass 2 – background: count actual lit pixels, not bbox area.
        # A desk surface or wall can have a bbox covering 60-70 % of the frame
        # while its true pixel area is 50-60 %; bbox-based checks let it through.
        mask_pixel_area = int((alpha > 0).sum())
        if mask_pixel_area > total_image_area * BACKGROUND_THRESHOLD:
            continue

        candidates.append({
            "alpha": alpha,
            "x": x, "y": y, "w": w, "h": h,
            "mask_pixel_area": mask_pixel_area,
        })

    # ── Pass 3: sub-part deduplication ───────────────────────────────────────
    # Sort largest-first (by actual pixel area) so the dominant object wins
    # when two masks compete.
    candidates.sort(key=lambda c: c["mask_pixel_area"], reverse=True)

    print("Filtering overlapping sub-parts …")
    final_objects: list[dict] = []
    for cand in candidates:
        is_subpart = False
        for saved in final_objects:
            # Pixel-level intersection: count pixels that are lit in BOTH masks.
            # This is the only reliable check when objects sit on a surface —
            # their bounding boxes are entirely inside the surface's box, so a
            # bbox-based check would wrongly flag every object as a sub-part.
            pixel_intersection = int(
                np.logical_and(cand["alpha"] > 0, saved["alpha"] > 0).sum()
            )
            if pixel_intersection == 0:
                continue
            overlap_ratio = pixel_intersection / cand["mask_pixel_area"]
            if overlap_ratio > SUBPART_OVERLAP_THRESHOLD:
                is_subpart = True
                break

        if not is_subpart:
            final_objects.append(cand)

    if not final_objects:
        print("⚠  No objects survived filtering — try lowering --conf or adjusting thresholds in config.py")
        return

    # ── Build preview + collect crops for user confirmation ──────────────────
    print("Preparing objects for review …")
    preview  = img.copy()
    # Collect (filename, crop_rgba) pairs; user will confirm each via save_images.
    detected: list[tuple[str, np.ndarray]] = []

    for i, obj in enumerate(final_objects):
        color = PALETTE[i % len(PALETTE)]
        alpha = obj["alpha"]

        crop = mask_to_rgba_crop(img, alpha)
        if crop is None:
            continue

        name = f"{i:02d}_SAM"
        print(f"  detected  {name}")
        detected.append((f"{name}.png", crop))

        # Annotate composite preview (all detections, regardless of confirmation).
        preview = overlay_mask(preview, alpha, color=color, alpha=0.40)
        label_pos = (obj["x"], max(obj["y"] - 6, 10))
        cv2.putText(preview, name, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)

    # ── interactive save: user confirms each crop with Y / any other key ──
    print(f"\nReview {len(detected)} detected object(s) — press Y to save, any other key to skip.\n")
    save_auto_images(detected, Path(out_dir))

    # ── always write the composite preview as a reference map ─────────────
    prev_path = os.path.join(out_dir, "_preview_sam_auto.png")
    cv2.imwrite(prev_path, preview)
    print(f"\n✅  {len(final_objects)} object(s) reviewed, preview → {prev_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 – INTERACTIVE  (SAM2 click-to-segment)
# ─────────────────────────────────────────────────────────────────────────────

# class made to keep tracking the session variables
@dataclass
class _InteractiveState:
    """All mutable state for the interactive session."""
    pos_pts:      list[tuple[int, int]] = field(default_factory=list)
    neg_pts:      list[tuple[int, int]] = field(default_factory=list)
    current_mask: Optional[np.ndarray] = None
    obj_count:    int = 0
    color_idx:    int = 0
    status:       str = "Left-click an object to start  |  right-click to exclude"


def run_interactive(image_path: str, out_dir: str, conf: float, sam_name: str):
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
    state = _InteractiveState()
    WIN   = "SAM2 Interactive Segmentation"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # ── drawing the window ────────────────────────────────────────────────
    def redraw() -> None:
        disp = img.copy()
        if state.current_mask is not None:
            color = PALETTE[state.color_idx % len(PALETTE)]
            disp  = overlay_mask(disp, state.current_mask, color=color)

        for p in state.pos_pts:
            cv2.circle(disp, p, 6, (0, 230,  80), -1)
        for p in state.neg_pts:
            cv2.circle(disp, p, 6, (30,  30, 220), -1)

        draw_hud(disp, [
            f"Object #{state.obj_count + 1}   saved: {state.obj_count}",
            "● green = include   ● blue = exclude",
            "[S] segment   [Enter] save   [N] clear   [U] undo   [Q] quit",
            f"→  {state.status}",
        ])
        cv2.imshow(WIN, disp)

    # ── mouse events ──────────────────────────────────────────────────────
    def on_mouse(event: int, x: int, y: int, _flags, _param):
        match event:
            case cv2.EVENT_LBUTTONDOWN:
                state.pos_pts.append((x, y))
                state.current_mask = None
                state.status = f"+ positive ({x},{y}) — press S to segment"
            case cv2.EVENT_RBUTTONDOWN:
                state.neg_pts.append((x, y))
                state.current_mask = None
                state.status = f"- negative ({x},{y}) — press S to segment"
        redraw()

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
        key = cv2.waitKey(40) & 0xFF

        if key in (ord('q'), 27):       # Q or Esc
            break
        elif key == ord('s'):
            _do_segment(image_path, model, state, W, H)
            redraw()
        elif key in (13, 10):           # Enter (13 Windows, 10 Unix)
            _do_save(img, state, out_dir)
            redraw()
        elif key == ord('n'):
            state.pos_pts.clear()
            state.neg_pts.clear()
            state.current_mask = None
            state.status = "Cleared — click a new object"
            redraw()
        elif key == ord('u'):
            if state.neg_pts:
                state.neg_pts.pop()
            elif state.pos_pts:
                state.pos_pts.pop()
            state.current_mask = None
            state.status = "Last point removed"
            redraw()

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print(f"\n✅  {state.obj_count} object(s) saved to '{out_dir}/'")


def _do_segment(image_path: str, model, state: _InteractiveState, W: int, H: int):
    """Run SAM2 with current points and update state.current_mask."""
    if not state.pos_pts and not state.neg_pts:
        state.status = "⚠ Add at least one point first"
        return

    state.status = "Segmenting …"

    all_pts = state.pos_pts + state.neg_pts
    all_lbl = [1] * len(state.pos_pts) + [0] * len(state.neg_pts)

    try:
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

    best  = int(np.argmax([m.sum() for m in masks_data]))
    mask  = upscale_mask(masks_data[best], (W, H))
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

    name = name or suggested

    crop = mask_to_rgba_crop(img, state.current_mask)
    if crop is None:
        print("  ⚠ Empty mask — nothing saved")
        return

    path = save_interactive_images(crop, out_dir, name)
    print(f"  ✓  Saved → {path}")

    state.obj_count  += 1
    state.color_idx  += 1
    state.pos_pts.clear()
    state.neg_pts.clear()
    state.current_mask = None
    state.status = "Saved!  Click the next object."

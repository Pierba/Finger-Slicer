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
  python segment_objects.py photo.jpg                               # yolo auto (default)
  python segment_objects.py photo.jpg --model-type sam              # SAM auto
  python segment_objects.py photo.jpg --model-type sam -i           # SAM interactive
  python segment_objects.py photo.jpg --model-type yolo --conf 0.1
  python segment_objects.py photo.jpg --yolo-model yolo26l-seg.pt
"""
from   pathlib import Path
import sys
# Adjust the import path to include the project root
ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from   config                         import *
import cv2
import numpy                          as np
from   src.segmentation.segment_utils import *
from   ultralytics                    import SAM, YOLO
import uuid

# =============================================================================
# YOLO AUTO MODE  (YOLO26-seg instance segmentation)
# =============================================================================

def run_yolo_auto(image_path: Path, conf: float = DEFAULT_CONF, model_name: Path = DEFAULT_YOLO_MODEL):
    """
    Detects and segments objects present in `image_path` with YOLO26x-seg (default model)
    - or any other `model_name` - using the given `conf` threshold.

    For each detection the pipeline is:
    - raw mask
    - upscale
    - binarise
    - morphological refinement
    - alpha crop
    - save PNG

    Eventually the composite preview image is also saved in `PREVIEW_DIR`.

    Args:
        image_path: path to the input image file.
        conf:       confidence threshold for detection.
        model_name: path to the YOLO model file.
    """
    print(f"Loading {model_name} ...")
    model = YOLO(model_name)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read image at path '{image_path}'")
        return

    H, W = img.shape[:2]

    print(f"Segmenting '{image_path}' with conf >= {conf} ...\n")
    result = model.predict(image_path, conf=conf, device=get_device(), verbose=False)[0]
    if not result or result.masks is None:
        print("No segmentation masks returned.")
        return

    # Extract numpy arrays from result
    masks_raw = result.masks.data.cpu().numpy()   # raw masks at model's output resolution
    boxes     = result.boxes.xyxy.cpu().numpy()   # x1, y1, x2, y2
    confs     = result.boxes.conf.cpu().numpy()   # confidence scores
    classes   = result.boxes.cls.cpu().numpy()    # class indices
    names     = result.names                      # class names list from the model
    del result

    # Copying the original image to draw the preview with mask overlays and labels
    preview  = img.copy()

    # Collect (filename, crop_rgba) pairs to let user confirm each via save_images
    detected: list[tuple[str, np.ndarray]] = []

    len_palette = len(PALETTE)
    for i, (mask_raw, box, conf_v, cls) in enumerate(zip(masks_raw, boxes, confs, classes)):
        # == metadata =====================================================
        label  = names[cls]                 # get class name from model's names list with class index
        color  = PALETTE[i % len_palette]   # cycling palette color per object

        # == mask pipeline ================================================
        mask = upscale_mask(mask_raw, (W, H))
        mask = refine_mask(mask)

        # Image cropped to the mask's bounding box with transparent background
        crop = mask_to_rgba_crop(img, mask)
        if crop is None:
            print(f"skip {label}: empty mask")
            continue

        # == queue for user confirmation ===================================
        name = uuid.uuid4().hex[:8]
        print(f"detected {name:<12} class={label:<15} conf={conf_v:.2f}")
        detected.append((name, crop))

        # == annotate preview =============================================
        preview = overlay_mask(preview, mask, color=color, alpha=ALPHA_CANVAS)
        x1, y1 = map(int, box[:2])
        cv2.putText(preview, name, (max(0, x1 - 8), max(0, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # == interactive save: user confirms each crop with Y / any other key ==
    print(f"\nReview {len(detected)} detected object(s) — press Y to save, any other key to skip.\n")
    save_auto_images(detected)

    # == always write the composite preview as a reference map =============
    prev_path = PREVIEW_DIR / "_preview_yolo_auto.png"
    if cv2.imwrite(str(prev_path), preview):
        print(f"\nPreview -> {prev_path}")
    else:
        print(f"\nWarning: could not write preview to {prev_path}")


# =============================================================================
# SAM AUTO MODE  (SAM2 fully-automatic segmentation)
# =============================================================================

def run_sam_auto(image_path: Path, model_name: Path = DEFAULT_SAM_MODEL):
    """
    Automatically segments all objects in `image_path` using SAM2 (default model) or any other `model_name`.
    Unlike YOLO, SAM2 has no concept of object classes, therefore it returns every mask it finds.
    Each raw mask is first cleaned up, then it goes through
    three filtering steps to remove noise and duplicates:
    - Upscale:      upscale_mask() resizes to original dimensions with bilinear interpolation + re-threshold.

    - Refine:       refine_mask() closes holes, removes specks, and keeps only the largest connected component.

    - Size:         drops masks whose bounding box is smaller than SMALL_OBJECT_THRESHOLD in either dimension.

    - Background:   drops masks whose actual pixel count is greater than BACKGROUND_THRESHOLD of the total image pixel area.

    - Sub-part:     sorts survivors by pixel area in descending order;
                    if a smaller mask's pixel-level intersection with a larger one exceeds SUBPART_OVERLAP_THRESHOLD it is discarded.

    Surviving objects are saved as transparent PNGs and the preview image (_preview_sam_auto.png) is saved in `PREVIEW_DIR`.

    Args:
        image_path: path to the input image file.
        model_name: path to the SAM model file.
    """
    print(f"Loading {model_name}...")
    model = SAM(model_name)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read image at path '{image_path}'")
        return

    H, W = img.shape[:2]
    total_image_area = W * H

    print("Generating masks with SAM2...")
    result = model.predict(image_path, device=get_device(), verbose=False)[0]
    if not result or result.masks is None:
        print("No objects detected in the image.")
        return

    # Pull raw mask tensors to CPU numpy and free the PyTorch tensors
    masks_data = result.masks.data.cpu().numpy()
    del result

    # == size filter + background filter ======================================
    print("Filtering small objects and background ...")
    candidates = []
    for mask_data in masks_data:
        # == mask pipeline ====================================================
        alpha = upscale_mask(mask_data, (W, H))
        alpha = refine_mask(alpha)

        # == minimum size check ===============================================
        x, y, w, h = cv2.boundingRect(alpha)
        if w <= SMALL_OBJECT_THRESHOLD or h <= SMALL_OBJECT_THRESHOLD:
            continue

        # == background rejection check =======================================
        # Use actual pixel count, not bbox area, to allow large objects with irregular shapes
        mask_pixel_area = int((alpha > 0).sum())
        if mask_pixel_area > total_image_area * BACKGROUND_THRESHOLD:
            continue

        # == adding candidate for next pass ===================================
        candidates.append({
            "alpha": alpha,
            "x": x, "y": y, "w": w, "h": h,
            "mask_pixel_area": mask_pixel_area,
        })

    # == sub-part deduplication ===============================================
    # Sort largest-first (by actual pixel area) so the dominant object wins when two masks compete
    candidates.sort(key=lambda c: c["mask_pixel_area"], reverse=True)

    print("Filtering overlapping sub-parts ...")
    final_objects: list[dict] = []
    for cand in candidates:
        is_subpart = False

        for saved in final_objects:
            # Shared pixels can only lie inside the overlap of the two bounding boxes,
            # so skip the costly mask and when the boxes are disjoint and crop to it otherwise
            ix0 = max(cand["x"], saved["x"])
            iy0 = max(cand["y"], saved["y"])
            ix1 = min(cand["x"] + cand["w"], saved["x"] + saved["w"])
            iy1 = min(cand["y"] + cand["h"], saved["y"] + saved["h"])
            if ix1 <= ix0 or iy1 <= iy0:
                continue

            # Count pixels lit in both masks, within that overlap rectangle only
            a = cand["alpha"][iy0:iy1, ix0:ix1] > 0
            b = saved["alpha"][iy0:iy1, ix0:ix1] > 0
            pixel_intersection = int(np.logical_and(a, b).sum())
            if pixel_intersection == 0:
                continue

            overlap_ratio = pixel_intersection / cand["mask_pixel_area"]
            if overlap_ratio > SUBPART_OVERLAP_THRESHOLD:
                is_subpart = True
                break

        if not is_subpart:
            final_objects.append(cand)

    if not final_objects:
        print("! No objects survived filtering — try lowering --conf or adjusting thresholds in config.")
        return

    # == Build preview + collect crops for user confirmation ==================
    print("Preparing objects for review ...")
    preview  = img.copy()

    # Collect (filename, crop_rgba) pairs; user will confirm each via save_images
    detected: list[tuple[str, np.ndarray]] = []

    len_palette = len(PALETTE)
    for i, obj in enumerate(final_objects):
        color = PALETTE[i % len_palette]
        alpha = obj["alpha"]

        crop = mask_to_rgba_crop(img, alpha)
        if crop is None:
            continue

        name = uuid.uuid4().hex[:8]
        print(f"detected {name}")
        detected.append((name, crop))

        # Annotate composite preview with mask overlay and label
        preview = overlay_mask(preview, alpha, color=color, alpha=ALPHA_CANVAS)
        label_pos = (obj["x"], max(obj["y"] - 6, 10))
        cv2.putText(preview, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # == interactive save: user confirms each crop with Y / any other key ==
    print(f"\nReview {len(detected)} detected object(s) — press Y to save, any other key to skip.\n")
    save_auto_images(detected)

    # == always write the composite preview as a reference map =============
    prev_path = PREVIEW_DIR / "_preview_sam_auto.png"
    if cv2.imwrite(str(prev_path), preview):
        print(f"\n{len(final_objects)} object(s) reviewed, preview -> {prev_path}")
    else:
        print(f"\nWarning: could not write preview to {prev_path}")


# =============================================================================
# SAM INTERACTIVE MODE  (SAM2 click-to-segment)
# =============================================================================

def run_interactive(image_path: Path, model_name: Path = DEFAULT_SAM_MODEL):
    """
    SAM2 interactive segmentation via mouse clicks and keyboard actions.

    Controls:
    - Left-click   Positive point (include in object)
    - Right-click  Negative point (exclude / background)
    - S            Run SAM2 segmentation with current points
    - Enter        Save current mask as transparent PNG
    - C            Clear points & mask - retry current object
    - Z            Undo last point
    - Q / Esc      Quit
    """
    print(f"Loading {model_name} ...")
    model = SAM(model_name)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Cannot read image at path '{image_path}'")
        return

    H, W = img.shape[:2]

    # == display scale =====================================================
    MAX_DISP_W, MAX_DISP_H = 1280, 980
    scale  = min(MAX_DISP_W / W, MAX_DISP_H / H, 1.0)
    disp_w, disp_h = int(W * scale), int(H * scale)

    # == window & state ====================================================
    state = InteractiveState()
    WIN   = "SAM2 Interactive Segmentation"
    # WINDOW_AUTOSIZE locks the window to exactly the image dimensions we send,
    # preventing any OS-level stretching
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    # == HUD rendering ====================================================
    def draw_hud(canvas: np.ndarray, lines: list[str]):
        """
        Renders text `lines` with a black background onto `canvas` in-place.

        Args:
            canvas: the image to draw on.
            lines:  list of text lines to render.
        """
        font, scale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1, 4  # HUD text style params
        y = 20  # initial y position for the first line
        for line in lines:
            # Creating black background containing line text
            (tw, th), base = cv2.getTextSize(line, font, scale, thick)
            cv2.rectangle(canvas, (8, y - th - pad), (12 + tw, y + base + pad), (0, 0, 0), -1)
            cv2.putText(canvas, line, (10, y), font, scale, (220, 220, 220), thick, cv2.LINE_AA)

            y += th + base + pad + 4

    # == drawing the window ================================================
    def redraw():
        """
        Renders: current state, current mask and click points onto the display window.
        """
        # Compose overlays at full resolution for maximum mask quality
        canvas = img.copy()
        if state.current_mask is not None:
            color = PALETTE[state.color_idx % len(PALETTE)]
            canvas = overlay_mask(canvas, state.current_mask, color=color, alpha=ALPHA_CANVAS)

        # Downscale to display size
        disp = cv2.resize(canvas, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        # Draw click dots and HUD `after` downscaling so they are always rendered at a fixed pixel size regardless of image resolution
        for (px, py), label in state.clicks:
            color = (0, 230, 80) if label == 1 else (30, 30, 220)
            cv2.circle(disp, (int(px * scale), int(py * scale)), 7, color, -1)

        draw_hud(disp, [
            f"Object #{state.obj_count + 1}   saved: {state.obj_count}",
            "green = include   red = exclude",
            "[S] segment   [Enter] save   [C] clear   [Z] undo   [Q] quit",
            f"->  {state.status}",
        ])
        cv2.imshow(WIN, disp)

    # == mouse events ======================================================
    def on_mouse(event: int, x: int, y: int, _flags, _param):
        # x, y arrives in display space — convert back to original image space before storing so that SAM receives the correct coordinates
        ix, iy = int(x / scale), int(y / scale)
        match event:
            case cv2.EVENT_LBUTTONDOWN:
                state.clicks.append(((ix, iy), 1))
                state.current_mask = None
                state.status = f"+ positive ({ix},{iy}) — press S to segment"
            case cv2.EVENT_RBUTTONDOWN:
                state.clicks.append(((ix, iy), 0))
                state.current_mask = None
                state.status = f"- negative ({ix},{iy}) — press S to segment"
        redraw()

    cv2.setMouseCallback(WIN, on_mouse)

    # == segmentation logic ================================================
    def _do_segment():
        """
        Runs SAM2 with current points and updates the current mask.
        """
        if not state.clicks:
            state.status = "Add at least one point first !!!"
            return

        state.status = "Segmenting ..."

        # SAM2 expects lists of points and labels, so we combine positive and negative points into single lists
        # The model will treat points with label 1 as "include in object" and points with label 0 as "exclude / background"
        all_pts, all_lbl = [], []
        for pt, lbl in state.clicks:
            all_pts.append(pt)
            all_lbl.append(lbl)

        try:
            result = model.predict(image_path, points=[all_pts], labels=[all_lbl], device=get_device(), verbose=False)
        except Exception as exc:
            state.status = f"SAM2 error: {exc}"
            return

        if not result or result[0].masks is None:
            state.status = "No mask returned — try more / different points"
            return

        masks_data = result[0].masks.data.cpu().numpy()
        del result
        if masks_data.size == 0:
            state.status = "No mask returned — try different points"
            return

        # If multiple masks are returned, we select the one with the largest pixel area
        best  = int(np.argmax([m.sum() for m in masks_data]))
        mask  = upscale_mask(masks_data[best], (W, H))
        state.current_mask = refine_mask(mask)
        state.status = "Mask ready — Enter to SAVE"

    # == save logic ================================================
    def _do_save():
        """
        Crops the current mask and saves the PNG as 'uuid.png'.
        """
        if state.current_mask is None:
            state.status = "No mask yet — press S first"
            return

        # The mask crop is done at full resolution to preserve the pixel-precise quality of SAM's output
        crop = mask_to_rgba_crop(img, state.current_mask)
        if crop is None:
            print("! Empty mask — nothing saved")
            return

        # Saving the image with a unique name to avoid overwriting issues
        name = uuid.uuid4().hex[:8]
        path = save_interactive_images(crop, name)
        print(f"Saved -> {path}")

        # Updating the state for the next segmentation
        state.obj_count  += 1
        state.color_idx  += 1
        state.clicks.clear()
        state.current_mask = None
        state.status = "Saved! Click the next object."

    # == initial instructions =================================================
    print(
        f"\n{'=' * 56}\n"
        "  Interactive SAM2 Segmentation\n"
        f"{'=' * 56}\n"
        "  Left-click   mark OBJECT  (green)\n"
        "  Right-click  mark BACKGROUND  (red)\n"
        "  S            segment\n"
        "  Enter        save as transparent PNG\n"
        "  C            clear & retry\n"
        "  Z            undo last point\n"
        "  Q / Esc      quit\n"
        f"{'=' * 56}\n"
    )

    # Initial render before any interaction
    redraw()

    # == event loop ========================================================
    while True:
        key = cv2.waitKey(40) & 0xFF    # Getting the ASCII code of the pressed key

        if key in (ord('q'), 27):       # Q or Esc to quit
            break
        elif key == ord('s'):           # S to segment
            _do_segment()
            redraw()
        elif key in (13, 10):           # Enter (13 Windows, 10 Unix) for saving the current mask
            _do_save()
            redraw()
        elif key == ord('c'):           # C to clear
            state.clicks.clear()
            state.current_mask = None
            state.status = "Cleared — click a new object"
            redraw()
        elif key == ord('z'):           # Z to undo
            if state.clicks:
                state.clicks.pop()
            state.current_mask = None
            state.status = "Last point removed"
            redraw()

        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print(f"\n{state.obj_count} object(s) saved to '{DEFAULT_OUTPUT_DIR}/'")

def main():
    """
    Main function to parse command-line arguments and run the segmentation modes based on user input.
    """
    # Parse arguments
    args = build_parser()

    image: Path         = args.image
    model_type: str     = args.model_type
    interactive: bool   = args.interactive
    conf: float         = args.conf
    yolo_model: Path    = args.yolo_model
    sam_model: Path     = args.sam_model

    # Validate input image path
    if not image.is_file():
        print(f"Error: Image not found at path '{image}'")
        return

    # Ensure the weights, assets and previews directories exist
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Branch the execution based on the selected model type and mode
    match model_type:
        case "yolo":
            print("Mode: YOLO AUTO  (YOLO26-seg instance segmentation)")
            run_yolo_auto(image, conf, yolo_model)
        case "sam":
            if interactive:
                print("Mode: INTERACTIVE  (SAM2 click-to-segment)")
                run_interactive(image, sam_model)
            else:
                print("Mode: SAM AUTO  (SAM2 fully-automatic segmentation)")
                run_sam_auto(image, sam_model)

if __name__ == "__main__":
    main()
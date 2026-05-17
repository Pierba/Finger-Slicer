from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ASSETS = Path("assets")
DEFAULT_YOLO_MODEL = "yolo26x-seg.pt"
DEFAULT_SAM_MODEL  = "sam2.1_b.pt"
DEFAULT_OUTPUT_DIR = "assets"
DEFAULT_CONF       = 0.05   # flat-lay / product photos score lower; 0.25 misses most objects

# ─────────────────────────────────────────────────────────────────────────────
# SAM auto-mode thresholds
# ─────────────────────────────────────────────────────────────────────────────

# Minimum bounding-box side length (px) for a mask to be kept.
# Masks whose width OR height is below this are treated as noise and dropped.
SMALL_OBJECT_THRESHOLD    = 50

# A mask whose actual pixel count exceeds this fraction of the total image
# pixel area is assumed to be the background (e.g. a desk surface or wall)
# and discarded.  Uses real mask pixel area, NOT bounding-box area.
# 0.50 catches large surfaces while keeping objects up to half the frame.
BACKGROUND_THRESHOLD      = 0.50

# When two candidate masks overlap, the smaller one is considered a sub-part
# of the larger one — and dropped — if their actual pixel-level intersection
# covers more than this fraction of the smaller mask's pixel area.
# Pixel-level comparison prevents objects *on* a surface from being wrongly
# discarded just because their bounding boxes sit inside the surface's box.
SUBPART_OVERLAP_THRESHOLD = 0.80

# BGR colours for mask overlays
PALETTE: list[tuple[int, int, int]] = [
    ( 72, 199, 142),  # teal
    (255, 159,  64),  # orange
    (100, 149, 237),  # cornflower
    (255,  99, 132),  # pink-red
    (153, 102, 255),  # purple
    (255, 205,  86),  # yellow
]

# ─────────────────────────────────────────────────────────────────────────────
# Saved-image canvas
# ─────────────────────────────────────────────────────────────────────────────

# Every extracted object is fitted (aspect-ratio preserving, centred) onto a
# transparent canvas of exactly these dimensions before being written to disk.
# Both auto and interactive modes use the same canvas, so all output PNGs are
# uniformly sized and compositable without further processing.
SAVE_IMG_W = 512
SAVE_IMG_H = 512

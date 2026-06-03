"""
Configuration file for the Finger Slicer project.
Contains constants and parameters for both segmentation and gameplay scripts.
"""
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================

# Paths for output images
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "assets"
PREVIEW_DIR      = Path(__file__).parent / "previews"

# YOLO / SAM model weights
DEFAULT_YOLO_MODEL = Path(__file__).parent / ".temp" / "yolo26x-seg.pt"
DEFAULT_SAM_MODEL  = Path(__file__).parent / ".temp" / "sam2.1_b.pt"

# =============================================================================
# YOLO AUTO-MODE THRESHOLDS
# =============================================================================

DEFAULT_CONF       = 0.05   # confidence threshold for yolo model; 0.25 misses most objects

# =============================================================================
# SAM AUTO-MODE THRESHOLDS
# =============================================================================

# Minimum bounding-box side length in pixels for a mask to be kept
# Masks whose width OR height is below this are treated as noise and dropped
SMALL_OBJECT_THRESHOLD    = 50

# A mask whose pixel count exceeds this percentage of the total image pixel area is treated as background and discarded
# Uses real mask pixel area instead bounding-box one
BACKGROUND_THRESHOLD      = 0.50

# When two candidate masks overlap, the smaller one is considered a sub-part of the larger one,
# therefore it's dropped if their actual pixel-level intersection covers more than this percentage
# of the smaller mask's pixel area
SUBPART_OVERLAP_THRESHOLD = 0.80

# =============================================================================
# SAVED IMAGES CANVAS SIZE
# =============================================================================

# Every extracted object is fitted onto a transparent canvas
# of these dimensions before being saved
SAVE_IMG_W = 512
SAVE_IMG_H = 512

# =============================================================================
# MASK OVERLAYS CONFIGS
# =============================================================================

PALETTE: list[tuple[int, int, int]] = [
    ( 72, 199, 142),  # teal
    (255, 159,  64),  # orange
    (100, 149, 237),  # cornflower
    (255,  99, 132),  # pink-red
    (153, 102, 255),  # purple
    (255, 205,  86),  # yellow
]

ALPHA_CANVAS = 0.45  # Alpha value for mask overlays

# =============================================================================
# HAND TRACKING CONFIGS
# =============================================================================

# Pretrained MediaPipe hand landmark model
# Downloaded on first run
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = Path(__file__).parent / ".temp" / "hand_landmarker.task"

# Landmark index of the index-finger tip (the point we want to highlight)
INDEX_FINGERTIP = 8

# Minimum confidence for the hand detection to be considered successful
HAND_DETECT_CONFIDENCE = 0.6

# Minimum confidence for the hand landmarks to be considered tracked successfully
HAND_TRACK_CONFIDENCE  = 0.6

# Requested webcam capture resolution and frame rate
# OpenCV will silently fall back to the nearest mode the camera supports
CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 60

# =============================================================================
# GAMEPLAY CONFIGS
# =============================================================================

# Folder we pull RGBA projectile sprites from (same place segment_objects.py saves to)
ASSETS_DIR            = DEFAULT_OUTPUT_DIR

# Longest side of a projectile in pixels after trimming + downscaling the source PNG
# Sources are saved at 512x512 with transparent padding — too big to throw around at game scale
PROJECTILE_MAX_SIZE   = 180

# Physics (units: pixels per frame at CAM_FPS)
GRAVITY               = 0.6
LAUNCH_VX_RANGE       = (2.0, 6.0)      # absolute; sign is chosen at spawn to arc toward center
LAUNCH_VY_RANGE       = (-30.0, -25.0)  # negative = upward kick; sized so apex lands in the upper third of the frame
SPIN_RANGE            = (-6.0, 6.0)     # degrees per frame

# A new projectile spawns every N frames (~1.2s at 30fps)
# Lower = harder
SPAWN_INTERVAL_FRAMES = 35

# Slicing
TRAIL_LEN             = 6       # fingertip positions kept for the "blade" polyline
MIN_SLICE_SPEED       = 18      # px between two samples; resting your finger should not slice
SLICE_HITBOX_SHRINK   = 0.6     # shrink the sprite bbox to this fraction for hit testing
SLICE_KICK            = 6.0     # horizontal split velocity given to each half on slice
SLICE_SPIN_BOOST      = 8.0     # extra angular velocity given to each half

# Game over after this many unsliced projectiles fall off-screen
MAX_MISSES            = 3

# Red "X" mark drawn where a projectile leaves the screen unsliced
MISS_MARK_LIFETIME    = 30      # frames the mark stays visible (~1s at 30fps)
MISS_MARK_SIZE        = 28      # half-length of each diagonal stroke, in pixels
MISS_MARK_THICKNESS   = 5       # stroke thickness
MISS_MARK_COLOR       = (0, 0, 255)  # BGR — red

# Bombs: occasionally a spawn is a "bomb" (red outline)
# Slicing one ends the game
BOMB_SPAWN_CHANCE     = 0.15    # probability a given spawn is a bomb
BOMB_OUTLINE_COLOR    = (0, 0, 255)  # BGR — red
BOMB_OUTLINE_THICKNESS = 4

# Combo: occasionally a spawn is a "combo" (yellow outline)
# Each hit scores and refreshes a slow-motion effect
COMBO_SPAWN_CHANCE     = 0.10
COMBO_OUTLINE_COLOR    = (0, 255, 255)  # BGR — yellow
COMBO_OUTLINE_THICKNESS = 4
COMBO_SLOWMO_DURATION  = 20     # frames of slow-motion granted per combo hit
COMBO_SLOWMO_FACTOR    = 0.35   # physics time-scale while slow-motion is active
COMBO_MAX_HITS         = 10      # hits before the combo finally splits like normal fruit
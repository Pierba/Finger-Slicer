from pathlib import Path

# Directory used by default to save the extracted objects
ASSETS = Path("assets")

# From 0.0 to 1.0
YOLO_CONF = 0.3
SAM_CONF = 0.4

# ============
#  SAM PARAMS
# ============
# Pixels. Objects smaller than this threshold will be ignored
SMALL_OBJECT_THRESHOLD = 40
# If the bounding box of the detected object covers more than this percentage of the entire image, it's considered background and will be ignored.
BACKGROUND_THRESHOLD = 0.60
# If a detected object is more than this percentage overlapped by a larger object, it's considered a sub-part and will be ignored.
SUBPART_OVERLAP_THRESHOLD = 0.3
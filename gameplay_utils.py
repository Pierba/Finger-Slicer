from   config import *

import time
import urllib.request
from   pathlib                import Path

import cv2
import mediapipe              as mp
from   mediapipe.tasks        import python as mp_python
from   mediapipe.tasks.python import vision as mp_vision

def load_model() -> Path:
    """
    Download the hand landmark model on first run and return its cached path.

    Returns:
        Path: Absolute path to the local `hand_landmarker.task` file.
    """
    if not HAND_MODEL_PATH.exists():
        # Create directory if it doesn't exist
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand_landmarker model to {HAND_MODEL_PATH} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    return HAND_MODEL_PATH
import cv2
import numpy as np
from pathlib import Path
import torch
from typing import Iterable

def get_device() -> torch.device:
    """
    Returns the appropriate (best) device (GPU, MPS, or CPU) for PyTorch operations.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def save_images(object_stream: Iterable[tuple[str, np.ndarray]], output_dir: Path) -> None:
    """
    Saves a stream of images to the specified output directory after user confirmation.
    """

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, img in object_stream:
        # Show image
        cv2.imshow("Save in transparent png? (y/n)", img)
        
        # Wait keystroke for user input
        key = cv2.waitKey(0) & 0xFF

        # Save
        if key == ord('y'):
            cv2.imwrite(output_dir / filename, img)
            print("Image saved.")
        else:
            print("Image skipped.")
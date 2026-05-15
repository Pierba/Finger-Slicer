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

def create_checkerboard(h: int, w: int, square_size: int = 20) -> np.ndarray:
    """
    Creates a checkerboard pattern image of the specified height and width.
    The square_size parameter controls the size of the squares in the checkerboard.
    Returns a BGR image (3 channels) with a checkerboard pattern.
    """

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    checker = ((x // square_size) + (y // square_size)) % 2
    
    # Gray background
    bg = np.full((h, w, 3), 180, dtype=np.uint8) 
    # White squares
    bg[checker == 1] = 255 
    
    return bg

def save_images(object_stream: Iterable[tuple[str, np.ndarray]], output_dir: Path) -> None:
    """
    Saves a stream of images to the specified output directory after user confirmation.
    """

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, img in object_stream:
        # Get the BGR channels and the Alpha channel
        bgr = img[:, :, :3]
        # Normalize the alpha channel to be between 0 and 1 for blending
        alpha = img[:, :, 3] / 255.0
        # Adjust the alpha channel dimensions for broadcasting
        alpha_3d = np.expand_dims(alpha, axis=2)
        
        # Create a checkerboard background
        h, w = img.shape[:2]
        bg = create_checkerboard(h, w)
        
        # Blend the object with the checkerboard background using the alpha channel as a mask.
        display_img = (bgr * alpha_3d + bg * (1 - alpha_3d)).astype(np.uint8)

        # Show image
        cv2.imshow("Save in transparent png? (y/n)", display_img)
        
        # Wait keystroke for user input
        key = cv2.waitKey(0) & 0xFF

        # Save
        if key == ord('y'):
            cv2.imwrite(output_dir / filename, img)
            print("Image saved.")
        else:
            print("Image skipped.")
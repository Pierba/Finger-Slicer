import config as cfg
import cv2
import numpy as np
from pathlib import Path
from typing import Generator
from ultralytics import YOLO
from utils import get_device

def extract_objects(image_path: Path) -> Generator[tuple[str, np.ndarray], None, None]:
    """
    Detects objects in an image, extracts them using pixel-perfect masks.
    Returns a generator of tuples (filename, image_array) for each detected object.
    The image_array is in RGBA format, where the Alpha channel represents the mask.
    """

    # Read the image using OpenCV
    print(f"Loading '{image_path}'...")
    img = cv2.imread(image_path)

    if img is None:
        print(f"ERROR: loading '{image_path}'.")
        return
    print(f"Image '{image_path}' loaded successfully.")
    
    # Image attributes
    H, W, _ = img.shape

    # Load a pre-trained YOLOv8 Segmentation model
    # (It will download automatically the first time you run it)
    print("Loading YOLOv8-seg model...")
    # 'n' stands for nano (fastest)
    model = YOLO('yolov8m-seg.pt')
    print("Model loaded successfully.")

    # Run inference
    print("Detecting objects...")
    results = model(image_path, device=get_device(), conf=cfg.YOLO_CONF)
    result = results[0] # Get the results for the first (and only) image
    
    if result.masks is None:
        print("No objects detected in the image.")
        return

    # Extract bounding boxes, masks, and class IDs
    # Bounding box coordinates [x1, y1, x2, y2]
    boxes = result.boxes.xyxy.cpu().numpy()
    # Raw mask tensors
    masks = result.masks.data.cpu().numpy()

    # Free pyTorch tensors from memory
    del results

    # Process each detected object
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        # We use the bounding box to crop tightly around the object.
        x1, y1, x2, y2 = map(int, box)        
        # Boundary checks to prevent slicing errors
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # YOLO scales masks down for processing. We must resize it back to the original image dimensions.
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Get only the cropped area of the mask and convert it to an alpha channel
        # Because we are using instance segmentation, the mask already excludes pixels of objects overlapping in front of it. 
        # We turn the mask into an Alpha (transparency) channel. 
        # 255 = fully visible object pixel, 0 = transparent background pixel.
        cropped_alpha = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
        
        # Crop the original image to get the BGR channels of the object
        cropped_bgr = img[y1:y2, x1:x2]

        # Merge the cropped BGR channels with the cropped Alpha channel to get a RGBA image of the object
        rgba_img = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
        rgba_img[:, :, 3] = cropped_alpha
    
        # Save
        yield f"{i}_YOLO.png", rgba_img

    print("Object extraction completed.")
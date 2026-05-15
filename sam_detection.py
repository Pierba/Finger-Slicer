import config as cfg
import cv2
import numpy as np
from pathlib import Path
from typing import Generator
from ultralytics import SAM
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
    total_image_area = W * H

    # Load SAM model
    # (It will download automatically the first time you run it)
    print("Loading SAM2 model...")
    # 'b' stands for base
    model = SAM("sam2_b.pt")
    print("Model loaded successfully.")

    # Run inference
    print("Generating masks with SAM...")
    results = model.predict(image_path, device=get_device(), conf=cfg.SAM_CONF)
    # Get the results for the first (and only) image
    if results[0].masks is None:
        print("No objects detected in the image.")
        return

    # Raw mask tensors
    masks_data = results[0].masks.data.cpu().numpy()

    # Free pyTorch tensors from memory
    del results

    # Process each detected object
    print("Filtering small objects and background...")
    candidates = []
    for mask_data in masks_data:
        # SAM scales masks down for processing. We must resize it back to the original image dimensions.
        # We turn the mask into an Alpha (transparency) channel. 
        # 255 = fully visible object pixel, 0 = transparent background pixel.
        mask_uint8 = (mask_data * 255).astype(np.uint8)
        alpha = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)

        # Find bounding box of the mask
        x, y, w, h = cv2.boundingRect(alpha)
        # If the mask is empty skip it
        if w == 0 or h == 0:
            continue
        mask_area = w * h
        
        # Ignores small objects or objects that are too big (background)
        if w > cfg.SMALL_OBJECT_THRESHOLD and h > cfg.SMALL_OBJECT_THRESHOLD:
            if mask_area > total_image_area * cfg.BACKGROUND_THRESHOLD:
                continue
            
            # Crop the image and the alpha channel
            cropped_bgr = img[y:y+h, x:x+w]
            cropped_alpha = alpha[y:y+h, x:x+w]
            
            # Merge the B, G, R channels with our new Alpha channel
            rgba = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = cropped_alpha 

            candidates.append({
                'rgba': rgba,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': mask_area
            })

    # Sort candidates by area in descending order (largest first)
    candidates.sort(key=lambda c: c['area'], reverse=True)

    # Filter out sub-parts
    print("Filtering overlapping sub-parts...")
    final_objects = []
    for cand in candidates:
        is_subpart = False
        for saved in final_objects:
            x_left = max(cand['x'], saved['x'])
            y_top = max(cand['y'], saved['y'])
            x_right = min(cand['x'] + cand['w'], saved['x'] + saved['w'])
            y_bottom = min(cand['y'] + cand['h'], saved['y'] + saved['h'])

            # Check if there is an overlap
            if x_right > x_left and y_bottom > y_top:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)
                overlap_ratio = intersection_area / cand['area']
                
                # If the overlap ratio is greater than the threshold, consider it a sub-part
                if overlap_ratio > cfg.SUBPART_OVERLAP_THRESHOLD:
                    is_subpart = True
                    break
        
        if not is_subpart:
            final_objects.append(cand)

    print("Object extraction completed.")

    # Save
    for i, obj in enumerate(final_objects):
        yield f"{i}.png", obj['rgba']
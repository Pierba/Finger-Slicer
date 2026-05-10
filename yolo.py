import cv2
import numpy as np
import os
from ultralytics import YOLO

def extract_objects(image_path, output_dir):
    """
    Detects objects in an image, extracts them using pixel-perfect masks, 
    and saves them as transparent PNGs.
    """
    # 1. Load a pre-trained YOLOv8 Segmentation model
    # (It will download automatically the first time you run it)
    print("Loading YOLOv8-seg model...")
    model = YOLO('yolov8n-seg.pt') # 'n' stands for nano (fastest)

    # 2. Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find or load image at {image_path}")
        return

    H, W, _ = img.shape
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 3. Run Inference
    print("Detecting objects...")
    results = model(image_path)
    result = results[0] # Get the results for the first (and only) image

    if result.masks is None:
        print("No objects detected in the image.")
        return

    # Extract bounding boxes, masks, and class IDs
    boxes = result.boxes.xyxy.cpu().numpy() # Bounding box coordinates [x1, y1, x2, y2]
    masks = result.masks.data.cpu().numpy() # Raw mask tensors
    classes = result.boxes.cls.cpu().numpy() # Class IDs

    print(f"Found {len(masks)} objects. Extracting...")

    # 4. Process each detected object
    for i, (mask, box, cls) in enumerate(zip(masks, boxes, classes)):
        
        # YOLO scales masks down for processing. We must resize it back to the original image dimensions.
        mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # --- HANDLING OVERRIDING/OVERLAPPING ---
        # Because we are using instance segmentation, the mask already excludes 
        # pixels of objects overlapping in front of it. 
        # We turn the mask into an Alpha (transparency) channel. 
        # 255 = fully visible object pixel, 0 = transparent background pixel.
        alpha_channel = (mask_resized * 255).astype(np.uint8)

        # Split the original image into Blue, Green, and Red channels
        b, g, r = cv2.split(img)

        # Merge the B, G, R channels with our new Alpha channel
        rgba_image = cv2.merge([b, g, r, alpha_channel])

        # We don't want a full-screen image with one tiny fruit in the corner.
        # We use the bounding box to crop tightly around the object.
        x1, y1, x2, y2 = map(int, box)
        
        # Boundary checks to prevent slicing errors
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Crop the RGBA image
        cropped_object = rgba_image[y1:y2, x1:x2]

        # 5. Save the image
        class_name = model.names[int(cls)]
        filename = os.path.join(output_dir, f"{class_name}_{i}.png")
        
        # OpenCV imwrite handles RGBA to PNG conversion automatically
        cv2.imwrite(filename, cropped_object)
        print(f"Saved: {filename}")

    print(f"Phase 1 Complete! Check the '{output_dir}' folder.")

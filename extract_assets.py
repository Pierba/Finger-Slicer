import cv2
import config as cfg
from pathlib import Path
from ultralytics import YOLO
import numpy as np

def extract_assets(image_path: Path, assets_path: Path = cfg.ASSETS) -> list | None:
    # Implementation for extracting assets
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return None

    # Create the output directory if it doesn't exist
    assets_path.mkdir(parents=True, exist_ok=True)
     
    # Load a pretrained YOLO segmentation model
    # The 'n' stands for nano (fastest). The '-seg' is required for pixel masks.
    model = YOLO("yolo11n-seg.pt") 
    
    # Run inference on the input image
    results = model.predict(source=image_path)
    
    for result in results:
        img = result.orig_img
        
        # Check if the model actually found anything
        if result.masks is None:
            print("No objects detected.")
            continue
            
        # Iterate over each detected object's mask, bounding box, and class label
        for i, (mask_obj, box_obj, cls) in enumerate(zip(result.masks, result.boxes, result.boxes.cls)):
            
            # Get the human-readable name of the object (e.g., "car", "dog")
            label = model.names[int(cls)]
            
            # 1. Create an empty black mask of the same size as the original image
            b_mask = np.zeros(img.shape[:2], np.uint8)
            
            # 2. Extract the polygon contour and draw it onto the mask in white (255)
            contour = mask_obj.xy[0].astype(np.int32).reshape(-1, 1, 2)
            cv2.drawContours(b_mask, [contour], -1, (255), cv2.FILLED)
            
            # 3. Create transparency: Stack the original image (BGR) with the binary mask (Alpha)
            # This creates a 4-channel image where everything outside the white mask becomes transparent
            transparent_img = np.dstack([img, b_mask])
            
            # 4. Get the bounding box coordinates to crop the image tightly around the object
            x1, y1, x2, y2 = box_obj.xyxy[0].cpu().numpy().astype(np.int32)
            cropped_object = transparent_img[y1:y2, x1:x2]
            
            # 5. Save as PNG (JPEG does not support transparency)
            output_path = f"{assets_path}/{label}_{i}.png"
            cv2.imwrite(output_path, cropped_object)
            print(f"Saved: {output_path}")

    
    # Add your asset extraction logic here
    return None
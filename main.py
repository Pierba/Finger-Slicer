from pathlib import Path
from segment_utils   import build_parser

from segment_objects import run_yolo_auto, run_sam_auto, run_interactive
# add finger_track.py

def main():
    # Parse arguments
    args = build_parser()

    image: Path         = args.image
    model_type: str     = args.model_type
    interactive: bool   = args.interactive
    output_dir: Path    = args.output
    conf: float         = args.conf
    yolo_model: Path    = args.yolo_model
    sam_model: Path     = args.sam_model

    # Validate input image path
    if not image.is_file():
        print(f"Error: Image not found at path '{image}'")
        return

    # Branch the execution based on the selected model type and mode
    match model_type:
        case "yolo":
            print("Mode: YOLO AUTO  (YOLO26-seg instance segmentation)")
            run_yolo_auto(image, output_dir, conf, yolo_model)
        case "sam":
            if interactive:
                print("Mode: INTERACTIVE  (SAM2 click-to-segment)")
                run_interactive(image, output_dir, sam_model)
            else:
                print("Mode: SAM AUTO  (SAM2 fully-automatic segmentation)")
                run_sam_auto(image, output_dir, sam_model)

if __name__ == "__main__":
    main()
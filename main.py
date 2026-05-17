import os
# from pathlib import Path
import sys
from segment_objects import run_yolo_auto, run_sam_auto, run_interactive
from segment_utils import build_parser

def main():
    args = build_parser()

    image: str          = args.image
    model_type: str     = args.model_type
    interactive: bool   = args.interactive
    output_dir: str     = args.output
    conf: float         = args.conf
    yolo_model: str     = args.yolo_model
    sam_model: str      = args.sam_model

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    match model_type:
        case "yolo":
            print("Mode: YOLO AUTO  (YOLO26-seg instance segmentation)")
            run_yolo_auto(image, output_dir, conf, yolo_model)
        case "sam":
            if interactive:
                print("Mode: INTERACTIVE  (SAM2 click-to-segment)")
                run_interactive(image, output_dir, conf, sam_model)
            else:
                print("Mode: SAM AUTO  (SAM2 fully-automatic segmentation)")
                run_sam_auto(image, output_dir, conf, sam_model)
        case _:
            sys.exit(f"Invalid mode: {model_type}")


if __name__ == "__main__":
    main()

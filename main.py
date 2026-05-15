import argparse
import config as cfg
from pathlib import Path
import utils
import yolo_detection as yl

def args_parser() -> argparse.Namespace:
    # Init parser
    parser = argparse.ArgumentParser(description="")

    # Flags
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image."
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=cfg.ASSETS,
        help="Path to the output image."
    )
    
    # Read arguments
    return parser.parse_args()

def main():
    args = args_parser()

    image_path: Path = args.image
    if not image_path.exists():
        print(f"ERROR: '{image_path}' does not exist!")
        return
    assets_path: Path = args.assets
    
    #device = utils.get_device()
    #if device.type == "cuda" or device.type == "mps":
    #    pass
    #else:
    #    pass
    stream = yl.extract_objects(image_path)

    # Save
    utils.save_images(stream, assets_path)

if __name__ == "__main__":
    main()
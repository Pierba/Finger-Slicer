import argparse
from pathlib import Path
import config as cfg

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
    assets_path: Path = args.assets



if __name__ == "__main__":
    main()
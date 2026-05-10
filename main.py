import argparse
from pathlib import Path
import config as cfg
from extract_assets import extract_assets

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
    assets = extract_assets(image_path, assets_path)



if __name__ == "__main__":
    main()
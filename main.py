import argparse
from pathlib import Path
import config as cfg


def main():
    # init parser
    parser = argparse.ArgumentParser(description='')

    # flags
    parser.add_argument('--image', type=Path, required=True, help='Path to the input image.')
    parser.add_argument('--assets', type=Path, default=cfg.ASSETS, help='Path to the output image.')
    # read arguments
    args = parser.parse_args()

    # create the output dir if it doesn't exist
    # exist_ok = True means that if the dir already exists, it won't raise an error
    args.assets.mkdir(exist_ok = True)

if __name__ == "__main__":
    main()
from pathlib import Path
import sam_detection as sam
import utils
import yolo_detection as yl

def main():
    args = utils.args_parser()

    image_path: Path = Path(args.image)
    assets_path: Path = Path(args.assets)
    
    # CON UN TEST VELOCE YOLO > SAM
    device = utils.get_device()
    print(f"Using '{device.type}' for detection.")
    
    match device.type:
        case "cuda" | "mps":
            stream = sam.extract_objects(image_path)
        case _:
            stream = yl.extract_objects(image_path)
    
    # Save
    utils.save_images(stream, assets_path)

if __name__ == "__main__":
    main()
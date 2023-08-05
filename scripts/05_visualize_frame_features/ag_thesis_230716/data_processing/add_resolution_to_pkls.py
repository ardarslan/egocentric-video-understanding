# %%

import argparse
import os
from os.path import basename, dirname, join
import pickle
from PIL import Image
import shutil
import sys
import zipfile

sys.path.append(dirname(dirname(__file__)))
from utils.args import arg_dict_to_list
from utils.globals import *
from data_handling.specific.ek100 import *
from utils.io import read_pkl


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--auto_full_size", action="store_true")
    parser.add_argument("--auto_first_entry_if_only", action="store_true")
    args = parser.parse_args(arg_dict_to_list(arg_dict))
    if not args.auto_full_size and args.width is None and args.height is None:
        raise argparse.ArgumentError()

    num_processed = 0
    thumbnail_size_cache = {}
    video_list = get_video_list()
    for root, dirs, files in os.walk(args.dir_path):
        if args.auto_full_size:
            # try to find video
            video_id = None
            for component in root.split(os.sep):
                if component in video_list:
                    video_id = component
                    break
            
            if video_id is None:
                continue

            reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id))
            video_width = reader.video_width
            video_height = reader.video_height

            if args.width is not None or args.height is not None:
                key = (video_width, video_height)
                if key in thumbnail_size_cache:
                    width_to_use, height_to_use = thumbnail_size_cache[key]
                else:
                    img = Image.new("L", (video_width, video_height))
                    img.thumbnail(args.width or LARGE, args.height or LARGE)
                    width_to_use, height_to_use = (img.width, img.height)
                    thumbnail_size_cache[key] = (img.width, img.height)
            else:
                width_to_use = video_width
                height_to_use = video_height
        else:
            width_to_use = args.width
            height_to_use = args.height

        for fn in files:
            fn_lower = fn.lower()
            if not fn_lower.endswith(".pkl") and not fn_lower.endswith(".zip"):
                continue

            path = join(root, fn)
            pkl_data = read_pkl(path)
            first_entry = False
            if not isinstance(pkl_data, dict):
                if isinstance(pkl_data, list) and len(pkl_data) == 1 and args.auto_first_entry_if_only:
                    first_entry = True 
                else:
                    print(f"Not a dict: {path}")
                    continue

            if first_entry: 
                pkl_data[0]["image_width"] = width_to_use
                pkl_data[0]["image_height"] = height_to_use
            else:
                pkl_data["image_width"] = width_to_use
                pkl_data["image_height"] = height_to_use

            if fn_lower.endswith(".zip"):
                # "path" already ends with ".zip":
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, False) as zip_file:
                    zip_file.writestr(basename(path), pickle.dumps(pkl_data))
            else:
                with open(path, "wb") as f:
                    pickle.dump(pkl_data, f)
            
            num_processed += 1
            print(f"Processed {path} (#{num_processed})")
    print("Done")


if __name__ == "__main__":
    main()

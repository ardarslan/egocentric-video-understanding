# %%

# partly from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join, dirname
import pickle
from PIL import Image
from scipy.sparse import csr_matrix
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
import time
import torch
from tqdm import tqdm
import zipfile

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import get_action_recognition_frame_gen
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.imaging import superimpose_colored_mask
from utils.io import read_pkl


def main(arg_dict=None):
    start_time = int(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument(
        "--input_dir", type=str, default=join(ROOT_PATH, "data", "EK_sam_mask_outputs")
    )
    parser.add_argument(
        "--output_dir", type=str, default=join(ROOT_PATH, "data", "EK_sam_mask_vis")
    )
    parser.add_argument(
        "-f", "--f", help="Dummy argument to make ipython work", default=""
    )
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    gen = get_action_recognition_frame_gen(
        subsets=["val"],
        videos=[s.strip() for v in args.generator_videos for s in v.split(",")]
        if args.generator_videos is not None
        else None,
    )

    frame_idx = 0
    for data in tqdm(gen):
        im = data["image"]
        pil_img = Image.fromarray(im).convert("RGBA")
        frame_id = data["frame_id"]

        zip_path = join(args.input_dir, data["video_id"], f"{frame_id}.pkl.zip")

        if not os.path.isfile(zip_path):
            continue

        print(f"Found: {zip_path}")
        pkl = read_pkl(zip_path)

        out_dir = join(args.output_dir, data["video_id"])
        os.makedirs(out_dir, exist_ok=True)

        for frame_box_idx, frame_box in enumerate(pkl):
            box = frame_box["box"]
            masks = frame_box["masks"]
            # crop frame to box

            pil_img_cropped = pil_img.crop(box)

            filled_mask = Image.new(
                "RGBA",
                (pil_img_cropped.width, pil_img_cropped.height),
                (0, 0, 255, 128),
            )

            for mask_idx, mask in enumerate(masks):
                seg_mask = mask["segmentation"].toarray()
                im_composite = superimpose_colored_mask(
                    pil_img_cropped, seg_mask, color=(0, 0, 255)
                )
                im_composite.save(
                    join(out_dir, f"{frame_id}_box={frame_box_idx}_mask={mask_idx}.jpg")
                )

        frame_idx += 1


if __name__ == "__main__":
    main()

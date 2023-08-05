# %%

import argparse
import cv2
import numpy as np
import os
from os.path import join, dirname
import pandas as pd
import pickle
import sys
import time
from tqdm import tqdm
from PIL import Image

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import get_action_recognition_frame_gen
from utils.args import arg_dict_to_list
from utils.globals import *


def main(arg_dict=None):
    start_time = int(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=f"./data/ek_variance_of_laplacian_{start_time}.pkl")
    parser.add_argument("--hand_boxes_only", action="store_true", default=False)
    args = parser.parse_args(arg_dict_to_list(arg_dict))

    gen = get_action_recognition_frame_gen(subsets=["val"],
                                           videos=[s.strip() for v in args.generator_videos for s in v.split(",")]
                                                   if args.generator_videos is not None else None)

    if args.resume_from not in ["", None]:
        with open(args.resume_from, "rb") as f:
            out_dict = pickle.load(f)
    else:
        out_dict = {}

    frame_idx = 0
    for data in (progress_bar := tqdm(gen)):
        frame_id = data["frame_id"]
        video_id = data["video_id"]
        progress_bar.set_description(frame_id + f" (#{frame_idx})")
        if frame_id in out_dict:
            continue

        im = data["image"]
        pil_img = Image.fromarray(im)
        has_output_path = args.output_path not in ["", None]
        if args.hand_boxes_only:
            hand_bbox_data_path = join(HAND_BBOX_DATA_DIR, video_id, f"{frame_id}.pkl")
            if not os.path.isfile(hand_bbox_data_path):
                continue
            
            with open(hand_bbox_data_path, "rb") as f:
                hand_bbox_data = pickle.load(f)
            frame_out_dict = {"hand_bbox_data_path": hand_bbox_data_path, "results": {}}
            for hand, hand_bbox in hand_bbox_data[2][0].items():
                if hand_bbox is None or hand not in ["left_hand", "right_hand"]:
                    continue

                x1 = hand_bbox[0]
                y1 = hand_bbox[1]
                x2 = x1 + hand_bbox[2]
                y2 = y1 + hand_bbox[3]
                im_cropped_resized = pil_img.crop([x1, y1, x2, y2])
                grayscale = cv2.cvtColor(np.array(im_cropped_resized), cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(grayscale, cv2.CV_64F).var()
                frame_out_dict["results"][hand] = laplacian_var

            if has_output_path:
                out_dict[frame_id] = frame_out_dict
            else:
                hand_bbox_data[2][0].update({k + "_laplacian_var": v for k, v in frame_out_dict["results"].items()})

                with open(hand_bbox_data_path, "wb") as f:
                    pickle.dump(hand_bbox_data, f)
        else:
            im_cropped_resized = pil_img.resize((pil_img.width // 3, pil_img.height // 3), resample=Image.BILINEAR)
            grayscale = cv2.cvtColor(np.array(im_cropped_resized), cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(grayscale, cv2.CV_64F).var()
            out_dict[frame_id] = float(laplacian_var)
        
        if has_output_path and frame_idx % 500 == 0:
            with open(args.output_path, "wb") as f:
                pickle.dump(out_dict, f)
        frame_idx += 1

        if frame_idx % 500 == 0:
            progress_bar.set_description(progress_bar.desc + " (sleeping 3s...)")
            time.sleep(3)


    if has_output_path:
        with open(args.output_path, "wb") as f:
            pickle.dump(out_dict, f)


if __name__ == "__main__":
    main()

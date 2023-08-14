# %%

import argparse
import numpy as np
import os
from os.path import dirname, join
import pickle
from PIL import Image, ImageDraw
import sys
from tqdm import tqdm

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import (
    get_action_recognition_frame_gen,
    get_video_list,
)
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.imaging import vstack_np_images


def main(arg_dict=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--f", type=str, default=None)
    parser.add_argument(
        "--input_dir", type=str, default=join(ROOT_PATH, "data", "EK_hand_bboxes")
    )
    parser.add_argument("--max_frames_per_video", type=int, default=10)
    parser.add_argument("--max_frames_per_bucket", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=1000)  # 200
    parser.add_argument("--thumbnail_size", type=int, default=480)
    parser.add_argument("--bucket_size", type=int, default=10)
    parser.add_argument(
        "--output_dir", type=str, default=join(ROOT_PATH, "data", "EK_hand_bbox_vis")
    )
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [
            s.strip() for v in args.generator_videos for s in v.split(",")
        ]

    buckets_images = {}
    videos_frames_processed = {}
    frames_processed = 0
    for video in args.generator_videos:
        gen = get_action_recognition_frame_gen(subsets=["val"], videos=[video])
        for frame_data in tqdm(gen):
            video_id = frame_data["video_id"]
            frame_id = frame_data["frame_id"]
            print(frame_id)

            video_frames_processed = videos_frames_processed.get(video_id, 0)
            if video_frames_processed >= args.max_frames_per_video:
                break
            videos_frames_processed[video_id] = video_frames_processed + 1
            image = Image.fromarray(frame_data["image"])
            draw = ImageDraw.Draw(image)

            pkl_path = join(args.input_dir, video_id, f"{frame_id}.pkl")
            if not os.path.isfile(pkl_path):
                continue

            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)

            if isinstance(pkl_data, tuple):
                pkl_data = list(pkl_data)

            if not isinstance(pkl_data, list) or len(pkl_data) < 2:
                continue

            hand_data = pkl_data[2]
            if isinstance(hand_data, list):
                hand_data = hand_data[0]

            if not isinstance(hand_data, dict):
                continue

            for hand, color in [("left_hand", "blue"), ("right_hand", "red")]:
                if hand in hand_data and hand_data[hand] is not None:
                    d = hand_data[hand]
                    draw.rectangle(
                        (d[0], d[1], d[0] + d[2], d[1] + d[3]), outline=color, width=2
                    )

            output_path = join(args.output_dir, video_id, f"{frame_id}.jpg")
            os.makedirs(dirname(output_path), exist_ok=True)

            thumbnail_img = image.copy()
            thumbnail_img.thumbnail((args.thumbnail_size, args.thumbnail_size))

            thumbnail_img.save(output_path)

            frames_processed += 1
            if frames_processed >= args.max_frames:
                break

        if frames_processed >= args.max_frames:
            break


if __name__ == "__main__":
    main()

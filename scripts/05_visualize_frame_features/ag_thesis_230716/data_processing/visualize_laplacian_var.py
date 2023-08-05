# %%

import argparse
import numpy as np
import os
from os.path import dirname, join
import pickle
from PIL import Image
import sys

sys.path.append(dirname(dirname(__file__)))

from data_handling.specific.ek100 import get_action_recognition_frame_gen, get_video_list
from utils.args import arg_dict_to_list
from utils.globals import *
from utils.imaging import vstack_np_images


def main(arg_dict=None):
    parser = argparse.ArgumentParser()    
    parser.add_argument("-f", "--f", type=str, default=None)
    parser.add_argument("--focus_path", type=str,
                        default="/mnt/scratch/agavryushin/Thesis/data/ek_variance_of_laplacian_1686167060.pkl.bak")
    parser.add_argument("--max_frames_per_video", type=int, default=10)
    parser.add_argument("--max_frames_per_bucket", type=int, default=10)
    parser.add_argument("--max_frames", type=int, default=1000) # 200
    parser.add_argument("--thumbnail_size", type=int, default=480)
    parser.add_argument("--bucket_size", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default=join(ROOT_PATH, "data", "laplacian_var_vis"))
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = [s.strip() for v in args.generator_videos for s in v.split(",")]

    with open(args.focus_path, "rb") as f:
        focus_dict = pickle.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    buckets_images = {}
    videos_frames_processed = {}
    frames_processed = 0
    for video in args.generator_videos:
        gen = get_action_recognition_frame_gen(subsets=["val"],
                                               videos=[video])
        for frame_data in gen:
            video_id = frame_data["video_id"]
            frame_id = frame_data["frame_id"]
            if frame_id not in focus_dict:
                continue
            video_frames_processed = videos_frames_processed.get(video_id, 0)
            if video_frames_processed >= args.max_frames_per_video:
                break
            videos_frames_processed[video_id] = video_frames_processed + 1
            focus_value = focus_dict[frame_id]
            image = frame_data["image"]

            bucket = (focus_value // args.bucket_size) * args.bucket_size
            if bucket not in buckets_images:
                buckets_images[bucket] = []
            elif len(buckets_images[bucket]) >= args.max_frames_per_bucket:
                continue
            thumbnail_img = Image.fromarray(frame_data["image"])
            thumbnail_img.thumbnail((args.thumbnail_size, args.thumbnail_size))
            buckets_images[bucket].append(np.array(thumbnail_img))
            frames_processed += 1
            print(f"{frames_processed=}, {focus_value=}, {bucket=}")
            if frames_processed >= args.max_frames:
                break
        
        if frames_processed >= args.max_frames:
            break

    for bucket_val, images in buckets_images.items():
        if len(images) == 0:
            continue

        stacked_img = Image.fromarray(vstack_np_images(images))
        stacked_img.save(join(args.output_dir, f"{bucket_val}.jpg"))


if __name__ == "__main__":
    main()

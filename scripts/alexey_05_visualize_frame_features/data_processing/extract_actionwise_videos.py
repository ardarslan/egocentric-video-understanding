# %%

import argparse
import cv2
import moviepy.video.io.ImageSequenceClip
import numpy as np
import os
from os.path import join, dirname, isdir, isfile
import pandas as pd
import pickle
import shutil
import sys
import tempfile
import time
from tqdm import tqdm
from PIL import Image

sys.path.append(dirname(dirname(__file__)))

from data_handling.video_reader import VideoReader
from data_handling.specific.ek100 import *
from utils.args import arg_dict_to_list
from utils.globals import *


def main(arg_dict=None):
    start_time = int(time.time())
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=ACTIONWISE_VIDEO_DATA_DIR)
    args = parser.parse_args(arg_dict_to_list(arg_dict))

    gen = get_action_recognition_frame_gen(
        subsets=["val"],
        videos=[s.strip() for v in args.generator_videos for s in v.split(",")]
        if args.generator_videos is not None
        else None,
        action_frames_only=True,
    )

    last_video_id = None
    last_initial_original_frame_idx = None
    last_initial_verb = None
    last_initial_noun = None
    last_original_frame_idx = None
    current_full_reader = None
    video_fn = None
    frame_paths = []
    current_dir = tempfile.mkdtemp()

    output_paths = []
    video_ids = []

    def clear_video():
        nonlocal current_dir, frame_paths, current_full_reader
        shutil.rmtree(current_dir)
        frame_paths.clear()
        current_full_reader = None

    def write_video(video_id):
        nonlocal current_dir, frame_paths, current_full_reader
        if not isdir(current_dir) or len(frame_paths) == 0:
            frame_paths.clear()
            current_full_reader = None
            return

        os.makedirs(join(args.output_path, video_id), exist_ok=True)
        output_clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
            frame_paths, fps=current_full_reader.fps
        )
        output_path = join(args.output_path, video_id, video_fn)
        output_clip.write_videofile(output_path)
        output_paths.append(output_path)
        video_ids.append(video_id)
        print(f"Wrote {output_path}")

        clear_video()

    for frame_data in tqdm(gen):
        original_frame_idx = frame_data["original_frame_idx"]
        frame_idx = frame_data["frame_idx"]
        video_id = frame_data["video_id"]

        if (
            original_frame_idx - 1 != last_original_frame_idx
            or last_video_id != video_id
        ):
            if video_fn is not None:
                output_path = join(args.output_path, last_video_id, video_fn)
                if isfile(output_path):
                    last_original_frame_idx = original_frame_idx
                    last_video_id = video_id
                    output_paths.append(output_path)
                    clear_video()
                    continue

            if len(frame_paths) > 0:
                write_video(last_video_id)
            else:
                clear_video()

        if current_full_reader is None:
            # create reader
            video_path = get_video_path(video_id)
            frame_dir_path = get_extracted_frame_dir_path(video_id)
            current_full_reader = VideoReader(
                video_path, frame_dir_path, assumed_fps=-1
            )
            last_initial_original_frame_idx = original_frame_idx
            last_initial_verb = frame_data["activity_verb"]
            last_initial_noun = frame_data["activity_noun"]
            video_fn = f"{video_id}_OS{'%07d' % original_frame_idx}_{last_initial_verb.replace(':', '-').replace(' ', '-')}_{last_initial_noun.replace(':', '-').replace(' ', '-')}.mp4"
            print(f"{video_fn=}")
            current_dir = tempfile.mkdtemp()

        output_path = join(args.output_path, video_id, video_fn)
        if not isfile(output_path):
            img_pil = Image.fromarray(frame_data["image"])
            frame_path = join(current_dir, f"{original_frame_idx}.jpg")
            img_pil.save(frame_path, quality=100)
            frame_paths.append(frame_path)

        last_original_frame_idx = original_frame_idx
        last_video_id = video_id

    if not isfile(join(args.output_path, last_video_id, video_fn)):
        write_video(last_video_id)

    if isdir(current_dir):
        shutil.rmtree(current_dir)

    return video_ids, output_paths


if __name__ == "__main__":
    main()

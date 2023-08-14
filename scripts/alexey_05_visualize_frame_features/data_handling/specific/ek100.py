# %%

import json
import numpy as np
import os
from os.path import join, dirname
import pandas as pd
import sys
import time


sys.path.append(dirname(dirname(dirname(__file__))))

from data_handling.dataset import construct_gen_item
from data_handling.video_reader import VideoReader
from utils.exceptions import ToggleableException
from utils.globals import *


EK100_DATASET_VIDEO_LIST_PATH = "/mnt/scratch/agavryushin/Thesis/data/ek_val_video_list.json"
EK100_DATASET_NAME = "ek100"
EK100_PATH =  join(DATA_ROOT_PATH, "EK")


def fmt_frame(video_id, frame_idx):
    return f"{video_id}_{'%07i' % frame_idx}"


def get_video_path(video_id, use_original=False):
    return join(EK100_PATH, "videos_original" if use_original else "videos_converted", f"{video_id}.MP4")


def get_extracted_frame_dir_path(video_id, use_original=False):
    return join(EK100_PATH, "videos_extracted_original" if use_original else "videos_extracted", video_id)


def get_video_list():
    with open(EK100_DATASET_VIDEO_LIST_PATH, "r") as f:
        return json.load(f)


def extend_video_list(video_list):
    new_videos = []
    for video_item in video_list:
        for video in video_item.split(","):
            video = video.strip()
            if len(video) == 0:
                continue
            elif "/" in video:
                all_videos = get_video_list()
                num = int(video.split("/")[0])
                den = int(video.split("/")[1])
                split = len(all_videos) // den
                if num == den:
                    extend_with = all_videos[split * (num-1) :]
                else:
                    extend_with = all_videos[split * (num-1) : split * num]
                new_videos.extend(extend_with)
                print(f'Extended "{video}" to {extend_with}')
            else:
                new_videos.append(video)
    return new_videos


def get_dataset_csv(subset):
    if subset == "train":
        return pd.read_csv(join(EK100_PATH, "EPIC_100_train.csv")).sort_values(by=["video_id", "start_frame"])
    elif subset == "val":
        return pd.read_csv(join(EK100_PATH, "EPIC_100_validation.csv")).sort_values(by=["video_id", "start_frame"])
    else:
        raise NotImplementedError()


def get_action_recognition_frame_gen(subsets=["train", "val"], videos=None, add_deltas=[], max_width=None, max_height=None, action_frames_only=True, use_original=False):
    train_csv = get_dataset_csv("train")
    val_csv = get_dataset_csv("val")

    if videos is not None:
        videos = extend_video_list(videos)

    if action_frames_only:
        for subset_name in subsets:
            subset = {"train": train_csv, "val": val_csv, "evaluation": val_csv}[subset_name]
            current_reader = None
            current_video_id = None
            for _, row in subset.iterrows():
                video_id = row["video_id"]
                if videos is not None and video_id not in videos:
                    continue
                
                if video_id != current_video_id:
                    del current_reader
                    current_reader = VideoReader(get_video_path(video_id, use_original=use_original),
                                                 get_extracted_frame_dir_path(video_id, use_original=use_original),
                                                 max_width, max_height, assumed_fps=EK_ASSUMED_FPS)
                    current_video_id = video_id

                range_obj = range(row["start_frame"], row["stop_frame"])
                for frame_idx in range_obj:
                    frame_id = fmt_frame(video_id, frame_idx)
                    try:
                        frame_img, original_frame_idx = current_reader.get_frame(frame_idx, return_real_frame_idx=True)
                    except ToggleableException as ex:
                        print(f"Error reading frame {frame_idx} from video {video_id}:", ex)
                        continue
                    
                    delta_imgs = {delta: (frame_img if delta == 0
                                        else current_reader.get_frame(frame_idx + delta))
                                    for delta in sorted(list(set((add_deltas or []) + [0])))}
                    yield construct_gen_item(EK100_DATASET_NAME, video_id, frame_idx, frame_id, frame_img, delta_imgs, original_frame_idx,
                                             activity_verb=row["verb"], activity_noun=row["noun"])
    else:
        current_reader = None
        for video_id in videos:
            del current_reader
            current_reader = VideoReader(get_video_path(video_id, use_original=use_original),
                                         get_extracted_frame_dir_path(video_id, use_original=use_original),
                                         max_width, max_height, assumed_fps=EK_ASSUMED_FPS)

            for virtual_frame_idx in range(current_reader.get_virtual_frame_count()):
                frame_id = fmt_frame(video_id, virtual_frame_idx)
                frame_img, original_frame_idx = current_reader.get_frame(virtual_frame_idx, return_real_frame_idx=True)
                delta_imgs = {delta: (frame_img if delta == 0
                                    else current_reader.get_frame(virtual_frame_idx + delta))
                                for delta in sorted(list(set((add_deltas or []) + [0])))}
                yield construct_gen_item(EK100_DATASET_NAME, video_id, virtual_frame_idx, frame_id, frame_img,
                                         delta_imgs, original_frame_idx)
                

if __name__ == "__main__":
    # for compatibility with Python 3.7, do not use {=}
    num_frames = 0
    gen = get_action_recognition_frame_gen(["val"])
    missing_video_ids = set()
    for frame_data in gen:
        print(f"Read frame_data['frame_id']={frame_data['frame_id']}")
        #vid = frame_data["video_id"]
        #if vid not in missing_video_ids and not os.path.isfile(f"/mnt/scratch/agavryushin/Datasets/EK/videos_original/{vid}.MP4"):
        #    print(f"wget https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test/{vid.split('_')[0]}/{vid}.MP4")
        #    missing_video_ids.add(vid)
        #num_frames += 1
        #pass
    print(f"num_frames={num_frames}")
    print(f"missing_video_ids={missing_video_ids}")
    
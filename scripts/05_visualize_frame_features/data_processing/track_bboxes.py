# %%

import argparse
import motpy
from motpy.core import Box, Detection, Track, Vector
from motpy.tracker import MultiObjectTracker
import multiprocessing as mp
from multiprocessing.pool import Pool
import numpy as np
import os
from os.path import dirname, isfile, join
import pickle
from PIL import Image
import sys
import time
import ssl
import sys
import torch
from tqdm import tqdm
import zipfile

sys.path.append(dirname(dirname(__file__)))

from data_handling.reflection import (get_available_hos_versions,
                                      get_available_tracking_bbox_versions,
                                      get_available_segmentation_mask_versions,
                                      get_available_object_bbox_versions)
from data_handling.specific.ek100 import *
from data_handling.video_reader import VideoReader
from utils.globals import *
from utils.args import arg_dict_to_list
from utils.io import read_pkl
from utils.exceptions import NeverThrownException


ToggleableException = NeverThrownException


def process(process_idx, data):
    args, timestamp, video_ids = data
    data_types = list(args.data_types)
    if "hand_bbox" in data_types:
        data_types = [*[d for d in data_types if d != "hand_bbox"], "hand_bbox_left", "hand_bbox_right"]
    
    available_hos_versions = get_available_hos_versions()
    available_tracking_bbox_versions = get_available_tracking_bbox_versions()
    available_segmentation_mask_versions = get_available_segmentation_mask_versions()
    available_object_bbox_versions = get_available_object_bbox_versions()

    if "hos_hands" in data_types:
        data_types = [*[d for d in data_types if d != "hos_hands"],
                      *[f"hos_left_hand_{v}" for v in available_hos_versions],
                      *[f"hos_right_hand_{v}" for v in available_hos_versions]]

    if "hos" in data_types:
        data_types = [*[d for d in data_types if d != "hos"],
                      *[f"hos_left_hand_{v}" for v in available_hos_versions],
                      *[f"hos_right_hand_{v}" for v in available_hos_versions],
                      *[f"hos_object_{v}" for v in available_hos_versions]]
        
    if "segmentation_mask" in data_types:
        data_types = [*[d for d in data_types if d != "hos"],
                      *[f"segmentation_mask_{v}" for v in available_segmentation_mask_versions]]
        
    if "object_bbox" in data_types:
        data_types = [*[d for d in data_types if d != "object_bbox"],
                      *[f"object_bbox_{v}" for v in available_object_bbox_versions]]

    data_types = list(set(data_types))

    def get_motpy_box(box):
        return np.array(box)
    
    def init_dict():
        return {"session_timestamp": int(timestamp), "args": args, "video_id": video_id, "assumed_fps": EK_ASSUMED_FPS, "data_types": data_types,
                "tracks": {data_type: {} for data_type in data_types},
                "tracks_segmented": {"gt": {data_type: {} for data_type in data_types}},
                "frame_original_box_idxs_to_tracks": {data_type: {} for data_type in data_types},
                "frame_original_box_idxs_to_tracks_segmented": {"gt": {data_type: {} for data_type in data_types}}}

    csv = get_dataset_csv("val")

    num_processed_frames = 0
    for video_idx, video_id in tqdm(enumerate(video_ids)):
        data_dict = {**init_dict(),
                     "exception": None}
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = join(args.output_dir, f"{video_id}.pkl")

        def save_pkl():
            with open(out_path, "wb") as f:
                pickle.dump(data_dict, f)

        reader = VideoReader(get_video_path(video_id), get_extracted_frame_dir_path(video_id),
                             assumed_fps=EK_ASSUMED_FPS)
        
        video_section_rows = csv[csv["video_id"] == video_id]

        try:
            trackers = {data_type: MultiObjectTracker(dt=1.0/reader.fps) for data_type in data_types}
            output_tracks = {}

            range_obj = range(reader.get_virtual_frame_count())
            
            for frame_idx in range_obj:
                start_frame = -1
                last_stop_frame = -1
                for row in video_section_rows.iterrows():
                    if row[1].loc["start_frame"] <= frame_idx <= row[1].loc["stop_frame"]:
                        start_frame = row[1]["start_frame"]
                    elif last_stop_frame < frame_idx < row[1].loc["start_frame"]:
                        start_frame = last_stop_frame + 1
                    last_stop_frame = row[1].loc["stop_frame"]
                if start_frame == -1:
                    start_frame = row[1].loc["stop_frame"] + 1
                suffix = f"-ffffff{'%010d' % start_frame}"

                frame_id = fmt_frame(video_id, frame_idx)
                frame_dict = init_dict()

                frame_dict_out_path = join(args.output_dir, video_id, f"{frame_id}.pkl")

                # check which data exists
                for data_type in data_types:
                    frame_data_type_detections = []
                    frame_boxes_idxs = []

                    data_dict["frame_original_box_idxs_to_tracks"][data_type][frame_id] = {}
                    data_dict["frame_original_box_idxs_to_tracks_segmented"]["gt"][data_type][frame_id] = {}

                    if "hand_bbox" in data_type:
                        path_func = CHANNEL_FRAME_PATH_FUNCTS["hand_bbox" if "hand_bbox" in data_type
                                                                else "hos" if data_type.startswith("hos_")
                                                                else data_type]
                        in_path = path_func(video_id, frame_idx, frame_id)
                    elif data_type.startswith("hos"):
                        path_func = CHANNEL_FRAME_PATH_FUNCTS["hos"]
                        if "threshold" in data_type:
                            threshold = float(data_type.replace("_", "=").split("=")[-1])
                            version = f"threshold={threshold}"
                        elif "egohos" in data_type:
                            version = "egohos"
                        else:
                            version = DEFAULT_HOS_VERSION
                        in_path = path_func(video_id, frame_idx, frame_id, version)
                    elif data_type.startswith("segmentation_mask"):
                        path_func = CHANNEL_FRAME_PATH_FUNCTS["segmentation_mask"]
                        if "threshold" in data_type:
                            threshold = float(data_type.replace("_", "=").split("=")[-1])
                            version = f"threshold={threshold}"
                        elif "egohos" in data_type:
                            version = "egohos"
                        elif "full_image" in data_type:
                            version = "full_image"
                        else:
                            version = DEFAULT_HOS_VERSION
                        in_path = path_func(video_id, frame_idx, frame_id, version)
                    elif data_type.startswith("object_bbox"):
                        path_func = CHANNEL_FRAME_PATH_FUNCTS["object_bbox"]
                        if data_type.startswith("object_bbox_"):
                            version = "_".join(data_type.split("_")[2:])
                        else:
                            version = DEFAULT_OBJECT_BBOX_VERSION
                        in_path = path_func(video_id, frame_idx, frame_id, version)
                    else:
                        path_func = CHANNEL_FRAME_PATH_FUNCTS[data_type]
                        in_path = path_func(video_id, frame_idx, frame_id)
                    
                    if not any((in_path.lower().endswith(ext) for ext in [".pkl", ".zip"])):
                        continue
                    
                    if not isfile(in_path):
                        continue
                    
                    #print(f"Found file: {in_path=}, {data_type=}")

                    external_data = read_pkl(in_path)

                    if data_type.startswith("segmentation_mask"):
                        print(f"{in_path=}")
                        for frame_box_idx, frame_box in enumerate(external_data):
                            outer_bbox = frame_box["box"]
                            masks = frame_box["masks"]
                            for mask_idx, mask in enumerate(masks):
                                mask_bbox = mask["bbox"]
                                mask_bbox[2] += mask_bbox[0]
                                mask_bbox[3] += mask_bbox[1]
                                mask_bbox_global = [outer_bbox[0] + mask_bbox[0], outer_bbox[1] + mask_bbox[1],
                                                    outer_bbox[0] + mask_bbox[2], outer_bbox[1] + mask_bbox[3]]

                                frame_boxes_idxs.append([mask_bbox_global, (frame_box_idx, mask_idx)])
                                frame_data_type_detections.append(Detection(box=get_motpy_box(mask_bbox_global)))
                    elif "hand_bbox" in data_type:
                        if isinstance(external_data, tuple):
                            external_data = list(external_data)

                        if not isinstance(external_data, list) or len(external_data) < 2:
                            continue
                        
                        hand_data = external_data[2]
                        if isinstance(hand_data, list):
                            hand_data = hand_data[0]
                        
                        if not isinstance(hand_data, dict):
                            continue
                        
                        hand = "left_hand" if "left" in data_type else "right_hand"
                        if hand in hand_data and hand_data[hand] is not None:
                            d = hand_data[hand]
                            box = np.array([d[0], d[1], d[0] + d[2], d[1] + d[3]])

                            frame_boxes_idxs.append([box, hand])
                            frame_data_type_detections.append(Detection(box=get_motpy_box(box)))
                    elif data_type.startswith("object_bbox"):
                        for outer_idx in range(len(external_data["classes"])):
                            inner_idx = 0
                            for cls, box, score in zip(external_data["classes"][outer_idx],
                                                        external_data["boxes"][outer_idx],
                                                        external_data["scores"][outer_idx]):
                                frame_boxes_idxs.append([box, (outer_idx, inner_idx)])
                                frame_data_type_detections.append(Detection(box=get_motpy_box(box)))
                                inner_idx += 1
                    elif data_type.startswith("hos_"):
                        idx = -1
                        if "instances" not in external_data:
                            print(f'"instances" not in external_data: {in_path}')
                            continue

                        if not hasattr(external_data["instances"], "pred_handsides"):
                            print(f'not hasattr(external_data["instances"], "pred_handsides"): {in_path}')
                            continue

                        for cls, handside, mask, box_tensor in zip(external_data["instances"].pred_classes,
                                                                   external_data["instances"].pred_handsides,
                                                                   external_data["instances"].pred_masks,
                                                                   external_data["instances"].pred_boxes):
                            idx += 1
                            box = box_tensor.numpy()
                            if cls == 0:  # 0: hand
                                # 0: left; 1: right
                                hand_side = handside.argmax().item()
                                if ["left", "right"][hand_side] not in data_type:
                                    continue
                                frame_boxes_idxs.append([box, idx])
                                frame_data_type_detections.append(Detection(box=get_motpy_box(box)))
                            else:  # 1: object
                                if "obj" not in data_type:
                                    continue

                                frame_boxes_idxs.append([box, idx])
                                frame_data_type_detections.append(Detection(box=get_motpy_box(box)))
                    else:
                        raise NotImplementedError()

                    trackers[data_type].step(detections=frame_data_type_detections)
                    tracks = trackers[data_type].active_tracks()
                    #print(f"{data_type=} {len(frame_data_type_detections)=} {len(tracks)=}")

                    for detection_idx, track_id in enumerate(trackers[data_type].detections_matched_ids):
                        track = next((t for t in tracks if t.id == track_id), None)
                        if track is None:
                            continue

                        track_id = track.id
                            
                        # id box score class_id
                        actual_original_box = frame_boxes_idxs[detection_idx][0]
                        original_box_idxs = []

                        #print(f"{data_type=} {detection_idx=} {track_id=}")
                        
                        for candidate_box, original_box_idx in frame_boxes_idxs:
                            if np.sum(np.abs(np.array(actual_original_box) - np.array(candidate_box))) < 1e-6:
                                original_box_idxs.append(original_box_idx)
                                if original_box_idx not in data_dict["frame_original_box_idxs_to_tracks"][data_type][frame_id]:
                                    data_dict["frame_original_box_idxs_to_tracks"][data_type][frame_id][original_box_idx] = [track_id]
                                    frame_dict["frame_original_box_idxs_to_tracks"][data_type][original_box_idx] = [track_id]

                                    data_dict["frame_original_box_idxs_to_tracks_segmented"]["gt"][data_type][frame_id][original_box_idx] = [track_id + suffix]
                                    frame_dict["frame_original_box_idxs_to_tracks_segmented"]["gt"][data_type][original_box_idx] = [track_id + suffix]
                                else:
                                    data_dict["frame_original_box_idxs_to_tracks"][data_type][frame_id][original_box_idx].append(track_id)
                                    frame_dict["frame_original_box_idxs_to_tracks"][data_type][original_box_idx].append(track_id)
                                    
                                    data_dict["frame_original_box_idxs_to_tracks_segmented"]["gt"][data_type][frame_id][original_box_idx].append([track_id + suffix])
                                    frame_dict["frame_original_box_idxs_to_tracks_segmented"]["gt"][data_type][original_box_idx].append([track_id + suffix])

                        if track_id not in data_dict["tracks"][data_type]:
                            data_dict["tracks"][data_type][track_id] = {}
                        if track_id + suffix not in data_dict["tracks_segmented"]["gt"][data_type]:
                            data_dict["tracks_segmented"]["gt"][data_type][track_id + suffix] = {}
                        
                        data_dict["tracks"][data_type][track_id][frame_id] = {"box": track.box, "score": track.score, "original_box_idxs": original_box_idxs}
                        frame_dict["tracks"][data_type][track_id] = {"box": track.box, "score": track.score, "original_box_idxs": original_box_idxs}
                        
                        data_dict["tracks_segmented"]["gt"][data_type][track_id + suffix][frame_id] = {"box": track.box, "score": track.score, "original_box_idxs": original_box_idxs}
                        frame_dict["tracks_segmented"]["gt"][data_type][track_id + suffix] = {"box": track.box, "score": track.score, "original_box_idxs": original_box_idxs}

                if isfile(frame_dict_out_path) and not args.force_overwrite:
                    continue

                frame_dict_out_dir = dirname(frame_dict_out_path)
                os.makedirs(frame_dict_out_dir, exist_ok=True)
                with open(frame_dict_out_path, "wb") as f:
                    pickle.dump(frame_dict, f)

                num_processed_frames += 1
                if num_processed_frames % 1000 == 0:
                    print(f"Process {process_idx}: at {frame_id} ({num_processed_frames} frames , {video_idx}/{len(video_ids)} videos)")
                    save_pkl()
        except ToggleableException as ex:
            print(f"Error processing {video_id}:", ex)
            data_dict["exception"] = ex
            save_pkl()
            continue
        
        save_pkl()
        

def main(arg_dict=None):
    mp.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=join(ROOT_PATH, "data", "EK_bbox_tracks"))
    parser.add_argument("--force_overwrite", action="store_true", default=False)
    parser.add_argument("--generator_videos", action="append", type=str, default=None)
    parser.add_argument("--num_jobs", type=int, default=5)
    parser.add_argument("--data_types", action="append", type=str, default=["hos,hand_bbox,object_bbox,segmentation_mask"])
    args, _ = parser.parse_known_args(arg_dict_to_list(arg_dict))

    timestamp = time.time()

    if args.generator_videos is None:
        args.generator_videos = get_video_list()
    else:
        args.generator_videos = extend_video_list([s.strip() for v in args.generator_videos for s in v.split(",")])

    args.num_jobs = min(args.num_jobs, len(args.generator_videos))
    args.data_types = [s.strip() for v in args.data_types for s in v.split(",")]

    if args.num_jobs == 1:
        process(0, (args, timestamp, args.generator_videos))
    else:
        with Pool(processes=args.num_jobs) as pool:
            paths_split = list(map(lambda a: list(map(str, a)), np.array_split(args.generator_videos, args.num_jobs)))
            pool.starmap(process, enumerate(zip([args] * args.num_jobs, [timestamp] * args.num_jobs, paths_split)))


if __name__ == "__main__":
    main()

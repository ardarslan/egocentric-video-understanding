import os
import gc
import cv2
import json
import argparse

import torch
torch.cuda.empty_cache()
from tqdm import tqdm
from utils import (
    get_frame_feature_extractor,
    get_output_file_name,
    get_error_file_name,
)
from frame_feature_extractor import FrameFeatureExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument(
        "--frame_feature_name",
        type=str,
        choices=[
            "blip2_vqa",
            "video_blip"
        ],
        default="video_blip"
    )
    parser.add_argument(
        "--split", type=str, choices=["train"] # CHANGE HERE
    )
    parser.add_argument(
        "--quarter_index", type=int, choices=[0, 1, 2, 3, 4, 5], default=0 # CHANGE HERE
    )
    parser.add_argument("--num_devices", type=int, default=torch.cuda.device_count())
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--annotations_json_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/08_reproduce_mq_experiments/data/ego4d",
            "ego4d_clip_annotations_v3.json",
        ),
    )
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/clips",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/frame_features",
    )
    parser.add_argument(
        "--error_folder_path",
        type=str,
        default=f"{os.environ['CODE']}/error_files",
    )

    args = parser.parse_args()

    os.makedirs(args.error_folder_path, exist_ok=True)

    frame_feature_extractor = get_frame_feature_extractor(args=args)
    column_names = frame_feature_extractor.column_names
    output_file_name = get_output_file_name(args=args)
    error_file_name = get_error_file_name(args=args)

    with open(args.annotations_json_file_path, "r") as annotations_json_file:
        annotations_dict = json.load(annotations_json_file)
        all_clip_uids = sorted(list(annotations_dict.keys()))
        clip_uids = [clip_uid for clip_uid in all_clip_uids if annotations_dict[clip_uid]["subset"] == args.split]

    if args.quarter_index == 0:
        clip_uids = clip_uids[: int(len(clip_uids) / 12)]
    elif args.quarter_index == 1:
        clip_uids = clip_uids[int(len(clip_uids) / 12) : int(2 * len(clip_uids) / 12)]
    elif args.quarter_index == 2:
        clip_uids = clip_uids[int(2 * len(clip_uids) / 12) : int(3 * len(clip_uids) / 12)]
    elif args.quarter_index == 3:
        clip_uids = clip_uids[int(3 * len(clip_uids) / 12) : int(4 * len(clip_uids) / 12)]
    elif args.quarter_index == 4:
        clip_uids = clip_uids[int(4 * len(clip_uids) / 12) : int(5 * len(clip_uids) / 12)]
    elif args.quarter_index == 5:
        clip_uids = clip_uids[int(5 * len(clip_uids) / 12) : int(6 * len(clip_uids) / 12)]
    elif args.quarter_index == 6:
        clip_uids = clip_uids[int(6 * len(clip_uids) / 12) : int(7 * len(clip_uids) / 12)]
    elif args.quarter_index == 7:
        clip_uids = clip_uids[int(7 * len(clip_uids) / 12) : int(8 * len(clip_uids) / 12)]
    elif args.quarter_index == 8:
        clip_uids = clip_uids[int(8 * len(clip_uids) / 12) : int(9 * len(clip_uids) / 12)]
    elif args.quarter_index == 9:
        clip_uids = clip_uids[int(9 * len(clip_uids) / 12) : int(10 * len(clip_uids) / 12)]
    elif args.quarter_index == 10:
        clip_uids = clip_uids[int(10 * len(clip_uids) / 12) : int(11 * len(clip_uids) / 12)]
    elif args.quarter_index == 11:
        clip_uids = clip_uids[int(11 * len(clip_uids) / 12) : ]
    else:
        raise Exception(f"{args.quarter_index} is not a valid quarter index.")

    for clip_uid in tqdm(clip_uids):
        if os.path.exists(
            os.path.join(args.output_folder_path, clip_uid, output_file_name)
        ):
            continue
        input_video_file_path = os.path.join(args.input_folder_path, clip_uid + ".mp4")
        cap = cv2.VideoCapture(input_video_file_path)

        results_list = []

        current_input_start_frame_index = 0
        while True:
            current_input_start_frame_index, current_input = frame_feature_extractor.get_new_input(
                current_input_start_frame_index=current_input_start_frame_index, cap=cap
            )

            if current_input is None:
                break
            current_result = frame_feature_extractor.predictor_function(**current_input)
            results_list.append(current_result)
            del current_input
            gc.collect()

        FrameFeatureExtractor.save_results(
            input_video_file_path=input_video_file_path,
            results_list=results_list,
            output_folder_path=args.output_folder_path,
            column_names=column_names,
            output_file_name=output_file_name,
        )
        del results_list

        cap.release()
        gc.collect()

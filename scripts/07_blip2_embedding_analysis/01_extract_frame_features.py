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
    get_output_file_name_wo_ext,
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
        default="blip2_vqa"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val"] # CHANGE HERE
    )
    parser.add_argument(
        "--quarter_index", type=int, choices=[0, 1, 2, 3, 4, 5], default=0 # CHANGE HERE
    )
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
    output_file_name_wo_ext = get_output_file_name_wo_ext(args=args)
    error_file_name = get_error_file_name(args=args)

    with open(args.annotations_json_file_path, "r") as annotations_json_file:
        annotations_dict = json.load(annotations_json_file)
        all_clip_uids = sorted(list(annotations_dict.keys()))
        clip_uids = [clip_uid for clip_uid in all_clip_uids if annotations_dict[clip_uid]["subset"] == args.split]
        clip_uids = clip_uids[int(args.quarter_index * len(clip_uids) / 12) : int((args.quarter_index + 1) * len(clip_uids) / 12)]

    for clip_uid in tqdm(clip_uids):
        input_video_file_path = os.path.join(args.input_folder_path, clip_uid + ".mp4")
        cap = cv2.VideoCapture(input_video_file_path)

        results_list = []

        file_name_counter = 0

        current_input_start_frame_index = 0
        while True:
            if os.path.exists(os.path.join(args.output_folder_path, clip_uid, output_file_name_wo_ext + "_" + str(file_name_counter).zfill(6) + ".tsv")):
                current_input_start_frame_index += 100 * frame_feature_extractor.window_center_frame_stride
                results_list = []
                file_name_counter += 1
                continue

            current_input_start_frame_index, current_input = frame_feature_extractor.get_new_input(current_input_start_frame_index=current_input_start_frame_index, cap=cap)

            if current_input is None:
                break

            current_result = frame_feature_extractor.predictor_function(**current_input)
            results_list.append(current_result)
            del current_input
            gc.collect()

            if len(results_list) == 100:
                FrameFeatureExtractor.save_results(
                    clip_uid=clip_uid,
                    results_list=results_list,
                    output_folder_path=args.output_folder_path,
                    column_names=column_names,
                    output_file_name=output_file_name_wo_ext + "_" + str(file_name_counter).zfill(6) + ".tsv",
                )
                results_list = []
                file_name_counter += 1

        if len(results_list) > 0:
            FrameFeatureExtractor.save_results(
                clip_uid=clip_uid,
                results_list=results_list,
                output_folder_path=args.output_folder_path,
                column_names=column_names,
                output_file_name=output_file_name_wo_ext + "_" + str(file_name_counter).zfill(6) + ".tsv",
            )

        cap.release()
        gc.collect()

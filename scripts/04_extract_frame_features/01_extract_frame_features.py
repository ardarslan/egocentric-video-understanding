import os
import gc
import json
import argparse

import ray
import torch
from tqdm import tqdm
import cv2
from utils import (
    get_frame_feature_extractor,
    get_column_names,
    get_output_file_name,
    get_error_file_name,
    GlobalFrameIndex,
)
from frame_feature_extractor import FrameFeatureExtractor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument(
        "--frame_feature_name",
        type=str,
        choices=[
            "blip2_vqa",
        ],
        default="blip2_vqa",
    )
    parser.add_argument(
        "--quarter_index", type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7], default=0
    )
    parser.add_argument("--num_devices", type=int, default=torch.cuda.device_count())
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_feature_extraction_stride", type=int, default=6)
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
        "--blip2_model_folder_path",
        type=str,
        default=f"{os.environ['SCRATCH']}/mq_libs/blip2",
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
        default=f"{os.environ['CODE']}/error_files/",
    )

    args = parser.parse_args()

    os.makedirs(args.error_folder_path, exist_ok=True)

    ray.init(num_gpus=args.num_devices, num_cpus=args.num_devices)

    frame_feature_extractor_pool = ray.util.ActorPool(
        [get_frame_feature_extractor(args=args) for _ in range(args.num_devices)]
    )
    column_names = get_column_names(args=args)
    output_file_name = get_output_file_name(args=args)
    error_file_name = get_error_file_name(args=args)

    with open(args.annotations_json_file_path, "r") as annotations_json_file:
        annotations_dict = json.load(annotations_json_file)
        clip_uids = sorted(list(annotations_dict.keys()))

    if args.quarter_index == 0:
        clip_uids = clip_uids[: int(len(clip_uids) / 8)]
    elif args.quarter_index == 1:
        clip_uids = clip_uids[int(len(clip_uids) / 8) : int(2 * len(clip_uids) / 8)]
    elif args.quarter_index == 2:
        clip_uids = clip_uids[int(2 * len(clip_uids) / 8) : int(3 * len(clip_uids) / 8)]
    elif args.quarter_index == 3:
        clip_uids = clip_uids[int(3 * len(clip_uids) / 8) : int(4 * len(clip_uids) / 8)]
    elif args.quarter_index == 4:
        clip_uids = clip_uids[int(4 * len(clip_uids) / 8) : int(5 * len(clip_uids) / 8)]
    elif args.quarter_index == 5:
        clip_uids = clip_uids[int(5 * len(clip_uids) / 8) : int(6 * len(clip_uids) / 8)]
    elif args.quarter_index == 6:
        clip_uids = clip_uids[int(6 * len(clip_uids) / 8) : int(7 * len(clip_uids) / 8)]
    elif args.quarter_index == 7:
        clip_uids = clip_uids[int(7 * len(clip_uids) / 8) : len(clip_uids)]
    else:
        raise Exception(f"{args.quarter_index} is not a valid quarter index.")

    for clip_uid in tqdm(clip_uids):
        if os.path.exists(
            os.path.join(args.output_folder_path, clip_uid, output_file_name)
        ):
            continue

        input_video_file_path = os.path.join(args.input_folder_path, clip_uid + ".mp4")
        output_subfolder_path = os.path.join(args.output_folder_path, clip_uid)

        cap = cv2.VideoCapture(input_video_file_path)

        global_frame_index = GlobalFrameIndex()

        inputs = FrameFeatureExtractor.get_inputs(
            cap=cap,
            batch_size=args.batch_size,
            frame_feature_name=args.frame_feature_name,
            output_subfolder_path=output_subfolder_path,
            frame_feature_extraction_stride=args.frame_feature_extraction_stride,
            global_frame_index=global_frame_index,
        )
        results_list = frame_feature_extractor_pool.map(
            lambda frame_feature_extractor, current_input: frame_feature_extractor.predictor_function.remote(
                *current_input
            ),
            inputs,
        )
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
        cap.release()

        FrameFeatureExtractor.save_results(
            input_video_file_path=input_video_file_path,
            results_list=results_list,
            output_folder_path=args.output_folder_path,
            column_names=column_names,
            output_file_name=output_file_name,
        )
        del results_list
        gc.collect()

    ray.shutdown()

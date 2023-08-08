import os
import gc
import json
import math
import argparse
import traceback
from datetime import datetime

import cv2
import ray
import torch
from tqdm import tqdm
from imutils.video import FileVideoStream

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
            "unidet",
            "visor_hos",
            "ego_hos",
            "gsam",
            "ofa",
            "blip_captioning",
            "blip_vqa",
        ],
        required=True,
    )
    parser.add_argument("--num_devices", type=int, required=True)
    parser.add_argument("--quarter_index", type=int, required=True)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_frames_per_processing_split", type=int, default=3000)
    parser.add_argument(
        "--annotations_json_file_path",
        type=str,
        default="/home/aarslan/mq/scripts/07_reproduce_baseline_results/ego4d_asl/data/ego4d/ego4d_clip_annotations_v3.json",
    )

    parser.add_argument("--unidet_confidence_threshold", type=float, default=0.4)
    parser.add_argument(
        "--unidet_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/unidet/Unified_learned_OCIM_R50_6x+2x.pth",
    )
    parser.add_argument(
        "--unidet_config_file_path",
        type=str,
        default="/home/aarslan/mq/scripts/04_extract_frame_features/unidet/configs/Unified_learned_OCIM_R50_6x+2x.yaml",
    )

    parser.add_argument("--visor_hos_confidence_threshold", type=float, default=0.4)
    parser.add_argument(
        "--visor_hos_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/visor_hos/model_final_hos.pth",
    )
    parser.add_argument(
        "--visor_hos_config_file_path",
        type=str,
        default="/home/aarslan/mq/scripts/04_extract_frame_features/visor_hos/configs/hos_pointrend_rcnn_R_50_FPN_1x.yaml",
    )

    parser.add_argument(
        "--ofa_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/ofa",
    )

    parser.add_argument(
        "--blip_captioning_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/blip/model_base_capfilt_large.pth",
    )
    parser.add_argument(
        "--blip_vqa_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/blip/model_base_vqa_capfilt_large.pth",
    )

    parser.add_argument(
        "--gsam_grounding_config_file_path",
        type=str,
        default="/home/aarslan/mq/scripts/04_extract_frame_features/gsam/gsam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to config file",
    )
    parser.add_argument(
        "--gsam_grounding_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/gsam/groundingdino_swint_ogc.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--gsam_ram_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/gsam/ram_swin_large_14m.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--gsam_box_threshold", type=float, default=0.25, help="box threshold"
    )
    parser.add_argument(
        "--gsam_text_threshold", type=float, default=0.2, help="text threshold"
    )
    parser.add_argument(
        "--gsam_iou_threshold", type=float, default=0.5, help="iou threshold"
    )

    parser.add_argument(
        "--ego_hos_seg_twohands_config_file_path",
        type=str,
        default="/home/aarslan/mq/scripts/04_extract_frame_features/ego_hos/configs/seg_twohands_ccda.py",
    )
    parser.add_argument(
        "--ego_hos_seg_twohands_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/ego_hos/seg_twohands_ccda/best_mIoU_iter_56000.pth",
    )
    parser.add_argument(
        "--ego_hos_twohands_to_cb_config_file_path",
        type=str,
        default="/home/aarslan/mq/scripts/04_extract_frame_features/ego_hos/configs/twohands_to_cb_ccda.py",
    )
    parser.add_argument(
        "--ego_hos_twohands_to_cb_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/ego_hos/twohands_to_cb_ccda/best_mIoU_iter_76000.pth",
    )
    parser.add_argument(
        "--ego_hos_twohands_cb_to_obj2_config_file_path",
        type=str,
        default="/home/aarslan/mq/scripts/04_extract_frame_features/ego_hos/configs/twohands_cb_to_obj2_ccda.py",
    )
    parser.add_argument(
        "--ego_hos_twohands_cb_to_obj2_model_file_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/datamq_libs/ego_hos/twohands_cb_to_obj2_ccda/best_mIoU_iter_32000.pth",
    )

    parser.add_argument(
        "--input_folder_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/data/ego4d_data/v2/clips",
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default="/srv/beegfs02/scratch/aarslan_data/data/ego4d_data/v2/frame_features",
    )
    parser.add_argument(
        "--error_folder_path",
        type=str,
        default="/home/aarslan/mq/error_files/",
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
        clip_uids = clip_uids[: int(len(clip_uids) / 4)]
    elif args.quarter_index == 1:
        clip_uids = clip_uids[int(len(clip_uids) / 4) : int(len(clip_uids) / 2)]
    elif args.quarter_index == 2:
        clip_uids = clip_uids[int(len(clip_uids) / 2) : int(3 * len(clip_uids) / 4)]
    elif args.quarter_index == 3:
        clip_uids = clip_uids[int(3 * len(clip_uids) / 4) :]

    for clip_uid in tqdm(clip_uids):
        try:
            if os.path.exists(
                os.path.join(args.output_folder_path, clip_uid, output_file_name)
            ):
                continue

            input_video_file_path = os.path.join(
                args.input_folder_path, clip_uid + ".mp4"
            )
            output_subfolder_path = os.path.join(args.output_folder_path, clip_uid)

            cap = FileVideoStream(input_video_file_path).start()

            global_frame_index = GlobalFrameIndex()

            results_list = []
            for _ in range(
                int(
                    math.ceil(
                        cap.stream.get(cv2.CAP_PROP_FRAME_COUNT)
                        / float(args.num_frames_per_processing_split)
                    )
                )
            ):
                inputs = FrameFeatureExtractor.get_inputs(
                    cap=cap,
                    batch_size=args.batch_size,
                    frame_feature_name=args.frame_feature_name,
                    output_subfolder_path=output_subfolder_path,
                    num_frames_per_processing_split=args.num_frames_per_processing_split,
                    global_frame_index=global_frame_index,
                )
                current_results_list = frame_feature_extractor_pool.map(
                    lambda frame_feature_extractor, current_input: frame_feature_extractor.predictor_function.remote(
                        *current_input
                    ),
                    inputs,
                )
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                results_list.extend(current_results_list)
            cap.stop()

            FrameFeatureExtractor.save_results(
                input_video_file_path=input_video_file_path,
                results_list=results_list,
                output_folder_path=args.output_folder_path,
                column_names=column_names,
                output_file_name=output_file_name,
            )
        except Exception as e:
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S%z} | Error with file {clip_uid}.mp4: \n"
            )
            print(traceback.format_exc())
            e = "-" * 100
            print(e)
            with open(
                os.path.join(args.error_folder_path, error_file_name), "a"
            ) as error_file_writer:
                error_file_writer.write(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S%z} | Error with file {clip_uid}.mp4: \n"
                )
                error_file_writer.write(traceback.format_exc())
                error_file_writer.write(e)

    ray.shutdown()

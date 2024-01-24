import os
import gc
import cv2
import json
import torch

import argparse
import numpy as np

torch.cuda.empty_cache()
from tqdm import tqdm
from utils import (
    get_frame_feature_extractor,
    get_output_file_name_wo_ext,
    get_error_file_name,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")

    parser.add_argument(
        "--frame_feature_name",
        type=str,
        choices=["blip2_vqa", "video_blip"],
        required=True,
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], required=True
    )
    parser.add_argument(
        "--quarter_index",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
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
    output_file_name_wo_ext = get_output_file_name_wo_ext(args=args)
    error_file_name = get_error_file_name(args=args)

    with open(args.annotations_json_file_path, "r") as annotations_json_file:
        annotations_dict = json.load(annotations_json_file)
        all_clip_uids = sorted(list(annotations_dict.keys()))
        clip_uids = [
            clip_uid
            for clip_uid in all_clip_uids
            if annotations_dict[clip_uid]["subset"] == args.split
        ]
        clip_uids = clip_uids[
            int(args.quarter_index * len(clip_uids) / 12) : int(
                (args.quarter_index + 1) * len(clip_uids) / 12
            )
        ]

    for clip_uid in tqdm(clip_uids):
        input_video_file_path = os.path.join(args.input_folder_path, clip_uid + ".mp4")
        cap = cv2.VideoCapture(input_video_file_path)

        caption_sbert_embeddings = []
        encoder_outputs = []

        current_embedding_index = 0
        while True:
            (
                current_embedding_index,
                current_input,
            ) = frame_feature_extractor.get_new_input(
                current_embedding_index=current_embedding_index, cap=cap
            )

            if current_input is None:
                break

            (
                caption_sbert_embedding,
                encoder_output,
            ) = frame_feature_extractor.predictor_function(**current_input)

            caption_sbert_embeddings.append(caption_sbert_embedding)
            encoder_outputs.append(encoder_output)

            del current_input
            gc.collect()

        caption_sbert_embeddings = torch.tensor(np.vstack(caption_sbert_embeddings))
        encoder_outputs = torch.tensor(np.vstack(encoder_outputs))

        frame_feature_extractor.save_results(
            caption_sbert_embeddings=caption_sbert_embeddings,
            encoder_outputs=encoder_outputs,
            output_folder_path=args.output_folder_path,
            clip_uid=clip_uid,
        )

        cap.release()
        gc.collect()

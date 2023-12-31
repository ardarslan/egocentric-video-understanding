import os
import pandas as pd
from tqdm import tqdm
import argparse
import gc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quarter_index",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/frame_features"),
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/postprocessed_frame_features"
        ),
    )
    args = parser.parse_args()

    clip_ids = sorted(os.listdir(args.input_folder_path))
    clip_ids = clip_ids[
        int(len(clip_ids) * args.quarter_index / 10.0) : int(
            len(clip_ids) * (args.quarter_index + 1) / 10.0
        )
    ]

    for clip_id in tqdm(clip_ids):
        os.makedirs(
            os.path.join(args.output_folder_path, clip_id, "encoder_output"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(args.output_folder_path, clip_id, "caption_sbert_embedding"),
            exist_ok=True,
        )
        file_names = os.listdir(os.path.join(args.input_folder_path, clip_id))
        for file_name in file_names:
            if os.path.exists(
                os.path.join(
                    args.output_folder_path,
                    clip_id,
                    "encoder_output",
                    file_name,
                )
            ) and os.path.exists(
                os.path.join(
                    args.output_folder_path,
                    clip_id,
                    "caption_sbert_embedding",
                    file_name,
                )
            ):
                continue
            current_df = pd.read_csv(
                os.path.join(args.input_folder_path, clip_id, file_name),
                sep="\t",
                usecols=[
                    "frame_index",
                    "encoder_output",
                    "caption_sbert_embedding",
                ],
            )
            current_df[["frame_index", "encoder_output"]].to_csv(
                os.path.join(
                    args.output_folder_path,
                    clip_id,
                    "encoder_output",
                    file_name,
                ),
                sep="\t",
                index=False,
            )
            current_df[["frame_index", "caption_sbert_embedding"]].to_csv(
                os.path.join(
                    args.output_folder_path,
                    clip_id,
                    "caption_sbert_embedding",
                    file_name,
                ),
                sep="\t",
                index=False,
            )
            del current_df
            gc.collect()

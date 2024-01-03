import os
import pandas as pd
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    clip_ids = os.listdir(args.input_folder_path)
    for clip_id in tqdm(clip_ids):
        file_names = os.listdir(args.input_folder_path, clip_id)
        for file_name in file_names:
            os.makedirs(
                os.path.join(
                    args.output_folder_path, clip_id, "blip2_llm_encoder_output"
                ),
                exist_ok=True,
            )
            os.makedirs(
                os.path.join(
                    args.output_folder_path, clip_id, "blip2_caption_sbert_embedding"
                ),
                exist_ok=True,
            )
            current_df = pd.read_csv(
                os.path.join(args.input_folder_path, clip_id, file_name),
                sep="\t",
                usecols=[
                    "frame_index",
                    "blip2_llm_encoder_output",
                    "blip2_caption_sbert_embedding",
                ],
            )
            current_df[["frame_index", "blip2_llm_encoder_output"]].to_csv(
                os.path.join(
                    args.output_folder_path,
                    clip_id,
                    "blip2_llm_encoder_output",
                    file_name,
                ),
                sep="\t",
            )
            current_df[["frame_index", "blip2_caption_sbert_embedding"]].to_csv(
                os.path.join(
                    args.output_folder_path,
                    clip_id,
                    "blip2_caption_sbert_embedding",
                    file_name,
                ),
                sep="\t",
            )
            del current_df

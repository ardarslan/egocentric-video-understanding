import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
from tqdm import tqdm


from sklearn.decomposition import IncrementalPCA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/postprocessed_frame_features"
        ),
    )
    parser.add_argument(
        "--annotations_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/08_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json",
        ),
    )
    args = parser.parse_args()

    clip_ids = os.listdir(args.input_folder_path)

    train_val_clip_ids = []
    with open(args.annotations_file_path, "r") as reader:
        annotations = json.load(reader)

    inc_pca = IncrementalPCA(n_components=1000)
    for clip_id in tqdm(clip_ids):
        if annotations[clip_id]["subset"] not in ["train", "val"]:
            continue
        current_file_names = os.listdir(
            os.path.join(args.input_folder_path, clip_id, "encoder_output")
        )
        for current_file_name in current_file_names:
            current_file_path = os.path.join(
                args.input_folder_path, clip_id, "encoder_output", current_file_name
            )
            current_df = pd.read_csv(current_file_path, sep="\t")
            current_embeddings = np.array(
                [
                    literal_eval(current_embedding)
                    for current_embedding in current_df["encoder_output"].values
                ]
            )
            print(current_embeddings.shape)
            inc_pca.partial_fit(current_embeddings)

    cumsum = np.cumsum(inc_pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum >= 0.95) + 1
    print(f"Selected number of components is {n_components}.")

    inc_pca = IncrementalPCA(n_components=n_components)
    for clip_id in tqdm(clip_ids):
        if annotations[clip_id]["subset"] not in ["train", "val"]:
            continue
        current_file_names = os.listdir(
            os.path.join(args.input_folder_path, clip_id, "encoder_output")
        )
        for current_file_name in current_file_names:
            current_file_path = os.path.join(
                args.input_folder_path, clip_id, "encoder_output", current_file_name
            )
            current_df = pd.read_csv(current_file_path, sep="\t")
            current_embeddings = np.array(
                [
                    literal_eval(current_embedding)
                    for current_embedding in current_df["encoder_output"]
                ]
            )
            inc_pca.partial_fit(current_embeddings)

    for clip_id in clip_ids:
        current_file_names = os.listdir(
            os.path.join(args.input_folder_path, clip_id, "encoder_output")
        )
        for current_file_name in current_file_names:
            current_file_path = os.path.join(
                args.input_folder_path, clip_id, "encoder_output", current_file_name
            )
            current_df = pd.read_csv(current_file_path, sep="\t")
            current_embeddings = np.array(
                [
                    literal_eval(current_embedding)
                    for current_embedding in current_df["encoder_output"]
                ]
            )
            current_embeddings = inc_pca.transform(current_embeddings)
            frame_indices = current_df["frame_index"].tolist()
            del current_df
            gc.collect()
            current_df = pd.DataFrame.from_dict(
                {
                    "frame_index": frame_indices,
                    "encoder_output_pca": [
                        current_embedding for current_embedding in current_embeddings
                    ],
                }
            )
            os.makedirs(
                os.path.join(args.input_folder_path, clip_id, "encoder_output_pca"),
                exist_ok=True,
            )
            current_df.to_csv(
                os.path.join(
                    args.input_folder_path,
                    clip_id,
                    "encoder_output_pca",
                    current_file_name,
                ),
                sep="\t",
                index=False,
            )

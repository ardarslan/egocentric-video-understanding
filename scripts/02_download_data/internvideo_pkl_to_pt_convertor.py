import os
import pickle
import argparse

from tqdm import tqdm
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--internvideo_pkl_clip_features_folder_path", type=str)
    parser.add_argument("--internvideo_pt_clip_features_folder_path", type=str)
    args = parser.parse_args()

    os.makedirs(args.internvideo_pt_clip_features_folder_path, exist_ok=True)

    for internvideo_pkl_clip_features_file_name in tqdm(
        os.listdir(args.internvideo_pkl_clip_features_folder_path)
    ):
        internvideo_pkl_clip_features_file_path = os.path.join(
            args.internvideo_pkl_clip_features_folder_path,
            internvideo_pkl_clip_features_file_name,
        )
        internvideo_pt_clip_features_file_path = os.path.join(
            args.internvideo_pt_clip_features_folder_path,
            internvideo_pkl_clip_features_file_name.replace(".pkl", ".pt"),
        )
        with open(
            internvideo_pkl_clip_features_file_path, "rb"
        ) as internvideo_pkl_clip_features_file:
            internvideo_clip_features_np = pickle.load(
                internvideo_pkl_clip_features_file
            )
            internvideo_clip_features_pt = torch.tensor(internvideo_clip_features_np)
            torch.save(
                internvideo_clip_features_pt, internvideo_pt_clip_features_file_path
            )

import os
import json
import torch
import argparse
from tqdm import tqdm


clip_size = 32
stride = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--annotations_folder_path",
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/annotations",
        type=str,
    )
    parser.add_argument(
        "--all_clip_names_file_path",
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/all_clip_names.txt",
        type=str,
    )
    parser.add_argument(
        "--slowfast_video_features_folder_path",
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/slowfast8x8_r101_k400",
        type=str,
    )
    parser.add_argument(
        "--omnivore_video_features_folder_path",
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/omnivore_video_swinl",
        type=str,
    )
    parser.add_argument(
        "--slowfast_clip_features_folder_path",
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/slowfast_clip",
        type=str,
    )
    parser.add_argument(
        "--omnivore_clip_features_folder_path",
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/omnivore_clip",
        type=str,
    )
    args = parser.parse_args()

    os.makedirs(args.slowfast_clip_features_folder_path, exist_ok=True)
    os.makedirs(args.omnivore_clip_features_folder_path, exist_ok=True)

    with open(
        os.path.join(args.annotations_folder_path, "moments_train.json"), "r"
    ) as moments_train_json_file:
        moments_train_json_dict = json.load(moments_train_json_file)
    with open(
        os.path.join(args.annotations_folder_path, "moments_val.json"), "r"
    ) as moments_val_json_file:
        moments_val_json_dict = json.load(moments_val_json_file)
    with open(
        os.path.join(args.annotations_folder_path, "moments_test_unannotated.json"), "r"
    ) as moments_test_json_file:
        moments_test_json_dict = json.load(moments_test_json_file)

    with open(args.all_clip_names_file_path, "r") as reader:
        all_clip_names = reader.read().strip().split(" ")

    for json_dict in [
        moments_train_json_dict,
        moments_val_json_dict,
        moments_test_json_dict,
    ]:
        for video_dict in tqdm(json_dict["videos"]):
            video_uid = video_dict["video_uid"]
            slowfast_video_features = None
            omnivore_video_features = None
            for clip_dict in video_dict["clips"]:
                clip_uid = clip_dict["clip_uid"]
                if clip_uid in all_clip_names:
                    if slowfast_video_features is None:
                        try:
                            slowfast_video_features = torch.load(
                                os.path.join(
                                    args.slowfast_video_features_folder_path,
                                    video_uid + ".pt",
                                )
                            ).numpy()
                        except Exception as e:
                            e = ""
                            print(
                                f"Slowfast features are not available for video with video uid: {video_uid}."
                                + e
                            )
                            continue

                    if omnivore_video_features is None:
                        try:
                            omnivore_video_features = torch.load(
                                os.path.join(
                                    args.omnivore_video_features_folder_path,
                                    video_uid + ".pt",
                                )
                            ).numpy()
                        except Exception as e:
                            e = ""
                            print(
                                f"Omnivore features are not available for video with video uid: {video_uid}."
                                + e
                            )
                            continue

                    ss = max(float(clip_dict["video_start_sec"]), 0)
                    es = float(clip_dict["video_end_sec"])
                    sf = max(int(clip_dict["video_start_frame"]), 0)
                    ef = int(clip_dict["video_end_frame"])
                    duration = es - ss
                    frames = ef - sf
                    fps = frames / duration
                    if fps < 10 or fps > 100:
                        continue

                    prepend_frames = sf % stride
                    prepend_sec = prepend_frames / fps
                    duration += prepend_sec
                    frames += prepend_frames

                    append_frames = append_sec = 0
                    if (frames - clip_size) % stride:
                        append_frames = stride - (frames - clip_size) % stride
                        append_sec = append_frames / fps
                        duration += append_sec
                        frames += append_frames

                    # save clip features
                    si = (sf - prepend_frames) // stride
                    ei = (ef + append_frames - clip_size) // stride

                    if ei > len(slowfast_video_features):
                        raise ValueError("end index exceeds slowfast feature length")
                    if ei > len(omnivore_video_features):
                        raise ValueError("end index exceeds omnivore feature length")

                    slowfast_clip_features = slowfast_video_features[si:ei]
                    omnivore_clip_features = omnivore_video_features[si:ei]

                    torch.save(
                        torch.tensor(slowfast_clip_features),
                        os.path.join(
                            args.slowfast_clip_features_folder_path, clip_uid + ".pt"
                        ),
                    )
                    torch.save(
                        torch.tensor(omnivore_clip_features),
                        os.path.join(
                            args.omnivore_clip_features_folder_path, clip_uid + ".pt"
                        ),
                    )

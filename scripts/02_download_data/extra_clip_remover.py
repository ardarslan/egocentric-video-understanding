import os
import json
import argparse


def get_clip_uids(json_file_path):
    with open(json_file_path, "r") as json_file:
        json_dict = json.load(json_file)
    clip_uids = []
    for video in json_dict["videos"]:
        for clip in video["clips"]:
            clip_uids.append(clip["clip_uid"])
    return clip_uids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--input_json_folder_path", type=str, required=True)
    parser.add_argument("--clips_folder_path", type=str, required=True)
    args = parser.parse_args()

    train_clip_uids = get_clip_uids(os.path.join(args.input_json_folder_path, "moments_train.json"))
    val_clip_uids = get_clip_uids(os.path.join(args.input_json_folder_path, "moments_val.json"))
    test_clip_uids = get_clip_uids(os.path.join(args.input_json_folder_path, "moments_test_unannotated.json"))
    train_val_test_clip_uids = set(train_clip_uids).union(val_clip_uids).union(test_clip_uids)

    for clip_file_name in os.listdir(args.clips_folder_path):
        if clip_file_name[-4:] == ".mp4" and clip_file_name[:-4] not in train_val_test_clip_uids:
            os.remove(os.path.join(args.clips_folder_path, clip_file_name))

import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--slowfast_video_features_folder_path", type=str, required=True
    )
    parser.add_argument("--annotations_folder_path", type=str, required=True)
    parser.add_argument(
        "--missing_slowfast_clip_names_file_path", type=str, required=True
    )
    args = parser.parse_args()

    train_annotations_json_file_path = os.path.join(
        args.annotations_folder_path, "moments_train.json"
    )
    val_annotations_json_file_path = os.path.join(
        args.annotations_folder_path, "moments_val.json"
    )
    test_annotations_json_file_path = os.path.join(
        args.annotations_folder_path, "moments_test_unannotated.json"
    )

    with open(train_annotations_json_file_path, "r") as train_annotations_json_file:
        train_annotations_json_dict = json.load(train_annotations_json_file)

    with open(val_annotations_json_file_path, "r") as val_annotations_json_file:
        val_annotations_json_dict = json.load(val_annotations_json_file)

    with open(test_annotations_json_file_path, "r") as test_annotations_json_file:
        test_annotations_json_dict = json.load(test_annotations_json_file)

    all_video_uids = []

    for annotations_json_dict in [
        train_annotations_json_dict,
        val_annotations_json_dict,
        test_annotations_json_dict,
    ]:
        for video in annotations_json_dict["videos"]:
            all_video_uids.append(video["video_uid"])

    existing_slowfast_video_uids = []
    for file_name in os.listdir(args.slowfast_video_features_folder_path):
        if file_name.split(".")[-1] != ".pt":
            continue
        existing_slowfast_video_uids.append(file_name.split(".")[0])

    missing_slowfast_video_uids = frozenset(
        set(all_video_uids) - set(existing_slowfast_video_uids)
    )
    missing_slowfast_clip_uids = []
    for annotations_json_dict in [
        train_annotations_json_dict,
        val_annotations_json_dict,
        test_annotations_json_dict,
    ]:
        for video in annotations_json_dict["videos"]:
            video_uid = video["video_uid"]
            if video_uid in missing_slowfast_video_uids:
                missing_slowfast_clip_uids.extend(
                    [clip["clip_uid"] for clip in video["clips"]]
                )
    with open(args.missing_slowfast_clip_names_file_path, "w") as writer:
        writer.write(" ".join(missing_slowfast_clip_uids) + "\n")

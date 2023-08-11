import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--annotations_folder_path", type=str, required=True)
    parser.add_argument("--all_clip_names_file_path", type=str, required=True)
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

    clip_uids = []

    for annotations_json_dict in [
        train_annotations_json_dict,
        val_annotations_json_dict,
        test_annotations_json_dict,
    ]:
        for video in annotations_json_dict["videos"]:
            for clip in video["clips"]:
                clip_uids.append(clip["clip_uid"])

    with open(args.all_clip_names_file_path, "w") as writer:
        writer.write(" ".join(clip_uids) + "\n")

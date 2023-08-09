import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--annotations_json_file_path", type=str, required=True)
    parser.add_argument("--updated_annotations_json_file_path", type=str, required=True)
    parser.add_argument("--clips_folder_path", type=str, required=True)
    parser.add_argument("--egovlp_egonce_features_folder_path", type=str, required=True)
    parser.add_argument("--internvideo_features_folder_path", type=str, required=True)
    parser.add_argument("--omnivore_features_folder_path", type=str, required=True)
    parser.add_argument("--slowfast_features_folder_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.annotations_json_file_path, "r") as annotations_json_file:
        annotations_json_dict = json.load(annotations_json_file)
        annotations_clip_uids = frozenset(annotations_json_dict.keys())
    
    slowfast_clip_uids = [feature_file_name.split(".")[0] for feature_file_name in os.listdir(args.slowfast_features_folder_path)]
    clip_uids = frozenset(set(annotations_clip_uids).intersection(slowfast_clip_uids))

    for folder_path in [args.clips_folder_path, args.egovlp_egonce_features_folder_path, args.internvideo_features_folder_path, args.omnivore_features_folder_path, args.slowfast_features_folder_path]:
        for file_name in os.listdir(folder_path):
            if file_name.split(".")[0] not in clip_uids:
                os.remove(os.path.join(folder_path, file_name))
            else:
                continue
    
    with open(args.updated_annotations_json_file_path, "w") as updated_annotations_json_file:
        updated_annotations_json_dict = dict([(clip_uid, annotations_json_dict[clip_uid]) for clip_uid in clip_uids])
        json.dump(updated_annotations_json_dict, updated_annotations_json_file)

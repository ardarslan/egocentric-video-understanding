import os
import cv2
import json
import argparse
import pickle
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default=os.path.join(os.environ["CODE"], "scripts/08_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json"))
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"], "scripts/06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/label_verb_noun_tool_mapping.json"
        ),
    )
    parser.add_argument("--output_file_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/ground_truth_labels", "ground_truth_labels.pickle"))
    args = parser.parse_args()

    clip_id_frame_id_ground_truth_label_indices_mapping = dict()

    with open(
        args.input_file_path,
        "r",
    ) as reader:
        ground_truth_labels = json.load(reader)

    with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)

    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    os.makedirs(Path(args.output_file_path).parent, exist_ok=True)

    for clip_id in ground_truth_labels.keys():
        clip_id_frame_id_ground_truth_label_indices_mapping[clip_id] = dict()
        cap = cv2.VideoCapture(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips", f"{clip_id}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for frame_id in range(num_frames):
            clip_id_frame_id_ground_truth_label_indices_mapping[clip_id][frame_id] = []
            frame_time = frame_id / float(fps)
            for annotation in ground_truth_labels[clip_id]["annotations"]:
                annotation_start_time = annotation["segment"][0]
                annotation_end_time = annotation["segment"][1]
                annotation_label = annotation["label"]
                annotation_label_index = distinct_ground_truth_labels.index(annotation_label)
                if frame_time >= annotation_start_time and frame_time <= annotation_end_time:
                    clip_id_frame_id_ground_truth_label_indices_mapping[clip_id][frame_id].append(annotation_label_index)
            if len(clip_id_frame_id_ground_truth_label_indices_mapping[clip_id][frame_id]) == 0:
                clip_id_frame_id_ground_truth_label_indices_mapping[clip_id][frame_id].append(len(distinct_ground_truth_labels))

    with open(args.output_file_path, "wb") as writer:
        pickle.dump(clip_id_frame_id_ground_truth_label_indices_mapping, writer)

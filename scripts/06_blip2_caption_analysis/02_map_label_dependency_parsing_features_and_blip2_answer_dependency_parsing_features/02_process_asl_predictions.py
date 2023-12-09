import os
import cv2
import json
import argparse
import pickle
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append("../../04_extract_frame_features/")

import constants


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, default=os.path.join(os.environ["CODE"], "scripts/08_reproduce_mq_experiments/submission_final.json"))
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"], "scripts/06_blip2_caption_analysis/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/label_verb_noun_tool_mapping.json"
        ),
    )
    parser.add_argument("--output_file_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/asl_predictions", "asl_predictions.pickle"))
    args = parser.parse_args()

    clip_id_frame_id_asl_predicted_label_indices_mapping = dict()

    with open(
        args.input_file_path,
        "r",
    ) as reader:
        asl_predictions = json.load(reader)["detect_results"]

    with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)

    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    os.makedirs(Path(args.output_file_path).parent, exist_ok=True)

    for clip_id in tqdm(list(asl_predictions.keys())):
        clip_id_frame_id_asl_predicted_label_indices_mapping[clip_id] = dict()
        cap = cv2.VideoCapture(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips", f"{clip_id}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        for frame_id in range(num_frames):
            clip_id_frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id] = dict()
            clip_id_frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][constants.question_constant_mapping["asl"]] = dict()
            frame_time = frame_id / float(fps)
            for annotation in asl_predictions[clip_id]:
                annotation_start_time = annotation["segment"][0]
                annotation_end_time = annotation["segment"][1]
                annotation_label = annotation["label"]
                annotation_score = annotation["score"]
                annotation_label_index = distinct_ground_truth_labels.index(annotation_label)
                if frame_time >= annotation_start_time and frame_time <= annotation_end_time:
                    if annotation_label_index not in clip_id_frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][constants.question_constant_mapping["asl"]].keys():
                        clip_id_frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][constants.question_constant_mapping["asl"]][annotation_label_index] = []
                    clip_id_frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][constants.question_constant_mapping["asl"]][annotation_label_index].append(annotation_score)

            if len(clip_id_frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][constants.question_constant_mapping["asl"]].keys()) == 0:
                clip_id_frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][constants.question_constant_mapping["asl"]][len(distinct_ground_truth_labels)] = [1.0]

    with open(args.output_file_path, "wb") as writer:
        pickle.dump(clip_id_frame_id_asl_predicted_label_indices_mapping, writer)

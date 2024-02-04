import os
import json
import math
import pickle
import argparse
from tqdm import tqdm

import sys

sys.path.append("../../04_extract_frame_features/")

import constants


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/08_reproduce_mq_experiments/proposed_features_v5_submission_final.json",
        ),
    )
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/06_blip2_caption_analysis/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/label_verb_noun_tool_mapping.json",
        ),
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data/proposed_features_v5",
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

    with open(
        args.input_file_path,
        "r",
    ) as reader:
        asl_predictions = json.load(reader)["detect_results"]

    with open(args.annotations_file_path, "r") as reader:
        annotations = json.load(reader)

    with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)

    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    os.makedirs(args.output_folder_path, exist_ok=True)

    for clip_id in tqdm(list(asl_predictions.keys())):
        duration = annotations[clip_id]["duration"]
        frame_id_asl_predicted_label_indices_mapping = dict()
        frame_id_asl_predicted_label_indices_mapping[clip_id] = dict()
        fps = 30.0
        num_frames = int(math.ceil(duration * fps))
        for frame_id in range(num_frames):
            frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id] = dict()
            frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][
                constants.question_constant_mapping["asl"]
            ] = dict()

        for annotation in asl_predictions[clip_id]:
            annotation_start_time = annotation["segment"][0]
            annotation_end_time = annotation["segment"][1]
            annotation_label = annotation["label"]
            annotation_score = annotation["score"]
            annotation_label_index = distinct_ground_truth_labels.index(
                annotation_label
            )
            for frame_id in range(
                int(math.ceil(annotation_start_time * fps)),
                int(math.floor(annotation_end_time * fps)),
            ):
                if (
                    annotation_label_index
                    in frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][
                        constants.question_constant_mapping["asl"]
                    ].keys()
                ):
                    frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][
                        constants.question_constant_mapping["asl"]
                    ][annotation_label_index].append(annotation_score)
                else:
                    frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][
                        constants.question_constant_mapping["asl"]
                    ][annotation_label_index] = [annotation_score]

        for frame_id in range(num_frames):
            if (
                len(
                    frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][
                        constants.question_constant_mapping["asl"]
                    ].keys()
                )
                == 0
            ):
                frame_id_asl_predicted_label_indices_mapping[clip_id][frame_id][
                    constants.question_constant_mapping["asl"]
                ][len(distinct_ground_truth_labels)] = [1.0]

        output_file_path = os.path.join(
            args.output_folder_path,
            clip_id + ".pickle",
        )

        with open(output_file_path, "wb") as writer:
            pickle.dump(frame_id_asl_predicted_label_indices_mapping, writer)

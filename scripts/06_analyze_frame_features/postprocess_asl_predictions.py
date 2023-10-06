import os
import cv2
import json
from tqdm import tqdm


with open(
    os.path.join(
        os.environ["CODE"],
        "scripts/06_analyze_frame_features/label_verb_noun_tool_mapping.json",
    ),
    "r",
) as reader:
    label_verb_noun_tools_mapping = json.load(reader)

distinct_ground_truth_labels = ["background"] + sorted(
    list(label_verb_noun_tools_mapping.keys())
)

clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping = dict()

asl_predictions_path = os.path.join(
    os.environ["CODE"],
    "scripts/07_reproduce_baseline_results/submission_final.json",
)

with open(
    asl_predictions_path,
    "r",
) as reader:
    asl_predictions = json.load(reader)["detect_results"]

for clip_id in tqdm(list(asl_predictions.keys())):
    clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[clip_id] = dict()
    cap = cv2.VideoCapture(
        os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips", clip_id + ".mp4")
    )
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    for frame_id in range(frame_count):
        clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[clip_id][
            frame_id
        ] = dict()
        for label_index in range(len(distinct_ground_truth_labels)):
            clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[clip_id][
                frame_id
            ][label_index] = 0.0
        current_frame_time = frame_id / fps
        assigned_label_to_current_frame = False
        annotations = asl_predictions[clip_id]
        for annotation in annotations:
            annotation_start_time = annotation["segment"][0]
            annotation_end_time = annotation["segment"][1]
            annotation_label = annotation["label"]
            annotation_label_index = distinct_ground_truth_labels.index(
                annotation_label
            )
            annotation_score = annotation["score"]
            if (
                annotation_start_time <= current_frame_time
                and annotation_end_time >= current_frame_time
            ):
                assigned_label_to_current_frame = True
                clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[
                    clip_id
                ][frame_id][annotation_label_index] = max(
                    clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[
                        clip_id
                    ][frame_id][annotation_label_index],
                    annotation_score,
                )
        if not assigned_label_to_current_frame:
            clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[clip_id][
                frame_id
            ][0] = 1.0

        # Normalize scores per frame so that their sum is equal to 1.0.
        sum_scores = 0.0
        for label_index in range(len(distinct_ground_truth_labels)):
            sum_scores += (
                clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[
                    clip_id
                ][frame_id][label_index]
            )
        for label_index in range(len(distinct_ground_truth_labels)):
            clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[clip_id][
                frame_id
            ][
                label_index
            ] = clip_id_frame_id_asl_predicted_label_indices_and_scores_mapping[
                clip_id
            ][
                frame_id
            ][
                label_index
            ] / float(
                sum_scores
            )

import os
import json
import pickle
import argparse
import numpy as np

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"], "scripts/06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/label_verb_noun_tool_mapping.json"
        ),
    )
    parser.add_argument("--predictions_file_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/asl_predictions/asl_predictions.pickle"))
    parser.add_argument("--ground_truth_folder_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.pickle"))
    parser.add_argument("--output_folder_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/evaluation_results"))
    args = parser.parse_args()

    with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)

    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    asl_predictions_one_hot_vectors_dict = dict()

    with open(os.path.join(args.predictions_file_path), "rb") as reader:
        asl_predictions = pickle.load(reader)
        for clip_id in asl_predictions.keys():
            asl_predictions_one_hot_vectors_dict[clip_id] = dict()
            for frame_id in asl_predictions[clip_id].keys():
                current_one_hot_vector = np.zeros(len(distinct_ground_truth_labels) + 1)
                for label_index, score in asl_predictions[clip_id][frame_id]:
                    if score >= args.threshold:
                        current_one_hot_vector[label_index] = 1
                asl_predictions_one_hot_vectors_dict[clip_id][frame_id] = current_one_hot_vector

    ground_truths_one_hot_vectors_dict = dict()

    for current_file_name in os.listdir(os.path.join(args.ground_truth_folder_path)):
        current_ground_truth_file_path = os.path.join(args.ground_truth_folder_path, current_file_name)
        with open(current_ground_truth_file_path, "rb") as reader:
            current_ground_truths = pickle.load(reader)
            for clip_id, frame_id_labels_mapping in current_ground_truths.items():
                ground_truths_one_hot_vectors_dict[clip_id] = dict()
                for frame_id, label_indices in frame_id_labels_mapping.items():
                    current_one_hot_vector = np.zeros(len(distinct_ground_truth_labels) + 1)
                    for label_index in label_indices:
                        current_one_hot_vector[label_index] = 1
                    ground_truths_one_hot_vectors_dict[clip_id][frame_id] = current_one_hot_vector

    predicted_one_hot_vectors_list = []
    ground_truth_one_hot_vectors_list = []
    for clip_id in ground_truths_one_hot_vectors_dict.keys():
        for frame_id in ground_truths_one_hot_vectors_dict[clip_id].keys():
            ground_truth_one_hot_vectors_list.append(ground_truths_one_hot_vectors_dict[clip_id][frame_id])
            predicted_one_hot_vectors_list.append(asl_predictions_one_hot_vectors_dict[clip_id][frame_id])

    f1_weighted_average = f1_score(y_true=ground_truth_one_hot_vectors_list, y_pred=predicted_one_hot_vectors_list, average="weighted", zero_division=0)
    precision_weighted_average = precision_score(y_true=ground_truth_one_hot_vectors_list, y_pred=predicted_one_hot_vectors_list, average="weighted", zero_division=0)
    recall_weighted_average = recall_score(y_true=ground_truth_one_hot_vectors_list, y_pred=predicted_one_hot_vectors_list, average="weighted", zero_division=0)

    f1_per_label = f1_score(y_true=ground_truth_one_hot_vectors_list, y_pred=predicted_one_hot_vectors_list, average=None, zero_division=0).tolist()
    precision_per_label = precision_score(y_true=ground_truth_one_hot_vectors_list, y_pred=predicted_one_hot_vectors_list, average=None, zero_division=0).tolist()
    recall_per_label = recall_score(y_true=ground_truth_one_hot_vectors_list, y_pred=predicted_one_hot_vectors_list, average=None, zero_division=0).tolist()

    os.makedirs(args.output_folder_path, exist_ok=True)

    df = []
    for label, f1, precision, recall in zip(distinct_ground_truth_labels + ["background"], f1_per_label, precision_per_label, recall_per_label):
        df.append((label, f1, precision, recall))
    df.append(("weighted_average", f1_weighted_average, precision_weighted_average, recall_weighted_average))

    df = pd.DataFrame(data=df, columns=["label", "f1_score", "precision_score", "recall_score"])
    df.to_csv(os.path.join(args.output_folder_path, f"prediction_method_asl__threshold_{str(args.threshold).replace('.', '')}.tsv"), sep="\t")

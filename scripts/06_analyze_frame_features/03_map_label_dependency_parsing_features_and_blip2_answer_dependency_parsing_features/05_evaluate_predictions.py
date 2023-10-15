import os
import json
import pickle
import argparse
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_folder_name",
        type=str,
        choices=[
            "blip2_dictionary_matching_max_per_label_predictions",
            "blip2_sbert_matching_max_per_label_predictions",
            "asl_max_per_label_predictions",
            "blip2_dictionary_matching_max_per_label_transfusion_predictions",
            "blip2_sbert_matching_max_per_label_transfusion_predictions",
            "asl_max_per_label_transfusion_predictions",
            "blip2_dictionary_matching_max_per_label_mode_filter_predictions",
            "blip2_sbert_matching_max_per_label_mode_filter_predictions",
            "asl_max_per_label_mode_filter_predictions",
            "blip2_dictionary_matching_max_per_label_median_filter_predictions",
            "blip2_sbert_matching_max_per_label_median_filter_predictions",
            "asl_max_per_label_median_filter_predictions",
        ],
        required=True,
    )
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"], "scripts/06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/label_verb_noun_tool_mapping.json"
        ),
    )
    parser.add_argument("--ground_truth_file_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.pickle"))
    parser.add_argument("--output_folder_path", type=str, default=os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/evaluation_results"))
    args = parser.parse_args()

    predictions_folder_path = os.path.join(os.environ["SCRATCH"], f"ego4d_data/v2/analysis_data/{args.predictions_folder_path}")

    with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)

    prediction_one_hot_vectors = []
    ground_truth_one_hot_vectors = []

    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    predictions_one_hot_vectors_dict = dict()

    for file_name in os.listdir(predictions_folder_path):
        with open(os.path.join(predictions_folder_path, file_name), "rb") as reader:
            current_predictions_max_per_label_postprocessing_results = pickle.load(reader)
            for clip_id in current_predictions_max_per_label_postprocessing_results.keys():
                predictions_one_hot_vectors_dict[clip_id] = dict()
                for frame_id in current_predictions_max_per_label_postprocessing_results[clip_id].keys():
                    predictions_one_hot_vectors_dict[clip_id][frame_id] = dict()
                    for blip2_question_index in current_predictions_max_per_label_postprocessing_results[clip_id][frame_id].keys():
                        predictions_one_hot_vectors_dict[clip_id][frame_id][blip2_question_index] = dict()
                        current_one_hot_vector = np.zeros(len(distinct_ground_truth_labels) + 1)
                        for label_index in current_predictions_max_per_label_postprocessing_results[clip_id][frame_id][blip2_question_index].keys():
                            current_score = current_predictions_max_per_label_postprocessing_results[clip_id][frame_id][blip2_question_index][label_index][1]
                            if current_score >= args.threshold:
                                current_one_hot_vector[label_index] = 1
                        predictions_one_hot_vectors_dict[clip_id][frame_id][blip2_question_index] = current_one_hot_vector

    with open(args.ground_truth_file_path, "rb") as reader:
        ground_truths = pickle.load(reader)

    ground_truth_one_hot_vectors_dict = dict()
    for clip_id, frame_id_ground_truth_labels_mapping in ground_truths.items():
        ground_truth_one_hot_vectors_dict[clip_id] = dict()
        for frame_id, ground_truth_label_indices in frame_id_ground_truth_labels_mapping.items():
            current_one_hot_vector = np.zeros(len(distinct_ground_truth_labels) + 1)
            for ground_truth_label_index in ground_truth_label_indices:
                current_one_hot_vector[ground_truth_label_index] = 1
            ground_truth_one_hot_vectors_dict[clip_id][frame_id] = current_one_hot_vector

    ground_truth_one_hot_vectors_list = []
    blip2_question_predicted_one_hot_vectors_list_mapping = dict()
    for clip_id in ground_truth_one_hot_vectors_dict.keys():
        for frame_id in ground_truth_one_hot_vectors_dict[clip_id].keys():
            ground_truth_one_hot_vectors_list.append(ground_truth_one_hot_vectors_dict[clip_id][frame_id])
            # Fix here
            predicted_one_hot_vectors_list.append(predictions_one_hot_vectors_dict[clip_id][int((frame_id // 6) * 6)])

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
    df.to_csv(os.path.join(args.output_folder_path, f"prediction_method_{args.prediction_method}__threshold_{str(args.threshold).replace('.', '')}.tsv"), sep="\t")

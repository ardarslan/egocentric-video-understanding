import os
import json
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from typing import Dict, List


def median_temporal_aggregation_select_labels(
    frame_id_blip2_question_index_label_index_max_score_mapping: Dict[
        int, Dict[int, Dict[int, List[int]]]
    ],
    threshold: float,
):
    # median filtering
    updated_frame_id_blip2_question_index_label_index_max_score_mapping = dict()
    for (
        frame_id,
        blip2_question_index_label_index_scores_mapping,
    ) in frame_id_blip2_question_index_label_index_max_score_mapping.items():
        updated_frame_id_blip2_question_index_label_index_max_score_mapping[
            frame_id
        ] = dict()
        for (
            blip2_question_index,
            label_index_scores_mapping,
        ) in blip2_question_index_label_index_scores_mapping.items():
            updated_frame_id_blip2_question_index_label_index_max_score_mapping[
                frame_id
            ][blip2_question_index] = dict()
            for label_index, score_tuple in label_index_scores_mapping.items():
                current_score = score_tuple[1]
                if frame_id == 0:
                    next_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id + 6
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    next_next_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id + 12
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    updated_frame_id_blip2_question_index_label_index_max_score_mapping[
                        frame_id
                    ][blip2_question_index][label_index] = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id
                        ][blip2_question_index][label_index][0],
                        np.median([current_score, next_score, next_next_score]),
                    )
                elif frame_id == 6:
                    previous_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id - 6
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    next_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id + 6
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    next_next_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id + 12
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    updated_frame_id_blip2_question_index_label_index_max_score_mapping[
                        frame_id
                    ][blip2_question_index].get(label_index, [0, 0])[1] = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id
                        ][blip2_question_index][label_index][0],
                        np.median(
                            [previous_score, current_score, next_score, next_next_score]
                        ),
                    )
                elif frame_id == max(
                    list(
                        frame_id_blip2_question_index_label_index_max_score_mapping.keys()
                    )
                ):
                    previous_previous_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id - 12
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    previous_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id - 6
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    updated_frame_id_blip2_question_index_label_index_max_score_mapping[
                        frame_id
                    ][blip2_question_index][label_index] = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id
                        ][blip2_question_index][label_index][0],
                        np.median(
                            [previous_previous_score, previous_score, current_score]
                        ),
                    )
                elif (
                    frame_id
                    == max(
                        list(
                            frame_id_blip2_question_index_label_index_max_score_mapping.keys()
                        )
                    )
                    - 6
                ):
                    previous_previous_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id - 12
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    previous_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id - 6
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    next_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id + 6
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    updated_frame_id_blip2_question_index_label_index_max_score_mapping[
                        frame_id
                    ][blip2_question_index][label_index] = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id
                        ][blip2_question_index][label_index][0],
                        np.median(
                            [
                                previous_previous_score,
                                previous_score,
                                current_score,
                                next_score,
                            ]
                        ),
                    )
                else:
                    previous_previous_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id - 12
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    previous_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id - 6
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    next_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id + 6
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    next_next_score = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id + 12
                        ][blip2_question_index].get(label_index, [0, 0])[1]
                    )
                    updated_frame_id_blip2_question_index_label_index_max_score_mapping[
                        frame_id
                    ][blip2_question_index][label_index] = (
                        frame_id_blip2_question_index_label_index_max_score_mapping[
                            frame_id
                        ][blip2_question_index][label_index][0],
                        np.median(
                            [
                                previous_previous_score,
                                previous_score,
                                current_score,
                                next_score,
                                next_next_score,
                            ]
                        ),
                    )

    # select labels
    frame_id_blip2_question_index_selected_label_indices_mapping = dict()
    for (
        frame_id,
        blip2_question_index_label_index_max_score_mapping,
    ) in updated_frame_id_blip2_question_index_label_index_max_score_mapping.items():
        frame_id_blip2_question_index_selected_label_indices_mapping[frame_id] = dict()
        for (
            blip2_question_index,
            label_index_max_score_mapping,
        ) in blip2_question_index_label_index_max_score_mapping.items():
            frame_id_blip2_question_index_selected_label_indices_mapping[frame_id][
                blip2_question_index
            ] = set()
            for label_index, score_tuple in label_index_max_score_mapping.items():
                if score_tuple[1] >= threshold:
                    frame_id_blip2_question_index_selected_label_indices_mapping[
                        frame_id
                    ][blip2_question_index].add(label_index)

    return frame_id_blip2_question_index_selected_label_indices_mapping


def no_temporal_aggregation_select_labels(
    frame_id_blip2_question_index_label_index_max_score_mapping: Dict[
        int, Dict[int, Dict[int, List[int]]]
    ],
    threshold: float,
):
    # select labels
    frame_id_blip2_question_index_selected_label_indices_mapping = dict()
    for (
        frame_id,
        blip2_question_index_label_index_max_score_mapping,
    ) in frame_id_blip2_question_index_label_index_max_score_mapping.items():
        frame_id_blip2_question_index_selected_label_indices_mapping[frame_id] = dict()
        for (
            blip2_question_index,
            label_index_max_score_mapping,
        ) in blip2_question_index_label_index_max_score_mapping.items():
            frame_id_blip2_question_index_selected_label_indices_mapping[frame_id][
                blip2_question_index
            ] = set()
            for label_index, score_tuple in label_index_max_score_mapping.items():
                if score_tuple[1] >= threshold:
                    frame_id_blip2_question_index_selected_label_indices_mapping[
                        frame_id
                    ][blip2_question_index].add(label_index)
    return frame_id_blip2_question_index_selected_label_indices_mapping


def transfusion_temporal_aggregation_select_labels(
    frame_id_blip2_question_index_label_index_max_score_mapping: Dict[
        int, Dict[int, Dict[int, List[int]]]
    ],
    threshold: float,
):
    # select labels
    frame_id_blip2_question_index_selected_label_indices_mapping = dict()
    for (
        frame_id,
        blip2_question_index_label_index_max_score_mapping,
    ) in frame_id_blip2_question_index_label_index_max_score_mapping.items():
        frame_id_blip2_question_index_selected_label_indices_mapping[frame_id] = dict()
        for (
            blip2_question_index,
            label_index_max_score_mapping,
        ) in blip2_question_index_label_index_max_score_mapping.items():
            frame_id_blip2_question_index_selected_label_indices_mapping[frame_id][
                blip2_question_index
            ] = set()
            for label_index, score_tuple in label_index_max_score_mapping.items():
                if score_tuple[1] >= threshold:
                    frame_id_blip2_question_index_selected_label_indices_mapping[
                        frame_id
                    ][blip2_question_index].add(label_index)

    # transfusion filtering
    for (
        frame_id,
        blip2_question_index_selected_label_indices_mapping,
    ) in frame_id_blip2_question_index_selected_label_indices_mapping.items():
        for (
            blip2_question_index,
            selected_label_indices,
        ) in blip2_question_index_selected_label_indices_mapping.items():
            for selected_label_index in selected_label_indices:
                if (
                    frame_id
                    < len(
                        frame_id_blip2_question_index_selected_label_indices_mapping.keys()
                    )
                    - 12
                ):
                    next_selected_label_indices = (
                        frame_id_blip2_question_index_selected_label_indices_mapping[
                            frame_id + 6
                        ][blip2_question_index]
                    )
                    next_next_selected_label_indices = (
                        frame_id_blip2_question_index_selected_label_indices_mapping[
                            frame_id + 12
                        ][blip2_question_index]
                    )
                    if (selected_label_index in next_next_selected_label_indices) and (
                        selected_label_index not in next_selected_label_indices
                    ):
                        next_selected_label_indices = frame_id_blip2_question_index_selected_label_indices_mapping[
                            frame_id + 6
                        ][
                            blip2_question_index
                        ].add(
                            selected_label_index
                        )

    return frame_id_blip2_question_index_selected_label_indices_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_folder_name",
        type=str,
        choices=[
            "blip2_dictionary_matching_max_per_label_predictions",
            "blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions",
            "blip2_sbert_matching_paraphrase-MiniLM-L6-v2_max_per_label_predictions",
        ],
        required=True,
    )
    parser.add_argument(
        "--temporal_aggregations",
        type=List[str],
        default=[
            # "median_temporal_aggregation",
            # "transfusion_temporal_aggregation",
            "no_temporal_aggregation",
        ],
    )
    parser.add_argument(
        "--thresholds", type=List[float], default=[0.2, 0.4, 0.6, 0.8, 1.0]
    )
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/label_verb_noun_tool_mapping.json",
        ),
    )
    parser.add_argument(
        "--ground_truth_file_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data/ground_truth_labels/ground_truth_labels.pickle",
        ),
    )
    parser.add_argument(
        "--output_folder_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/analysis_data/evaluation_results"
        ),
    )
    args = parser.parse_args()

    predictions_folder_path = os.path.join(
        os.environ["SCRATCH"],
        f"ego4d_data/v2/analysis_data/{args.predictions_folder_name}",
    )

    with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)
    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    for temporal_aggregation in args.temporal_aggregations:
        for threshold in args.thresholds:
            prediction_one_hot_vectors = []
            ground_truth_one_hot_vectors = []

            predictions_one_hot_vectors_dict = dict()

            for file_name in tqdm(os.listdir(predictions_folder_path)):
                with open(
                    os.path.join(predictions_folder_path, file_name), "rb"
                ) as reader:
                    current_predictions_max_per_label_postprocessing_results = (
                        pickle.load(reader)
                    )

                    for (
                        clip_id
                    ) in (
                        current_predictions_max_per_label_postprocessing_results.keys()
                    ):
                        frame_id_blip2_question_index_label_index_max_score_mapping = (
                            current_predictions_max_per_label_postprocessing_results[
                                clip_id
                            ]
                        )
                        if temporal_aggregation == "no_temporal_aggregation":
                            frame_id_blip2_question_index_selected_label_indices_mapping = no_temporal_aggregation_select_labels(
                                frame_id_blip2_question_index_label_index_max_score_mapping=frame_id_blip2_question_index_label_index_max_score_mapping,
                                threshold=threshold,
                            )
                        elif temporal_aggregation == "median_temporal_aggregation":
                            frame_id_blip2_question_index_selected_label_indices_mapping = median_temporal_aggregation_select_labels(
                                frame_id_blip2_question_index_label_index_max_score_mapping=frame_id_blip2_question_index_label_index_max_score_mapping,
                                threshold=threshold,
                            )
                        elif temporal_aggregation == "transfusion_temporal_aggregation":
                            frame_id_blip2_question_index_selected_label_indices_mapping = transfusion_temporal_aggregation_select_labels(
                                frame_id_blip2_question_index_label_index_max_score_mapping=frame_id_blip2_question_index_label_index_max_score_mapping,
                                threshold=threshold,
                            )

                        predictions_one_hot_vectors_dict[clip_id] = dict()
                        for (
                            frame_id
                        ) in (
                            frame_id_blip2_question_index_selected_label_indices_mapping.keys()
                        ):
                            predictions_one_hot_vectors_dict[clip_id][frame_id] = dict()
                            for (
                                blip2_question_index
                            ) in frame_id_blip2_question_index_selected_label_indices_mapping[
                                frame_id
                            ].keys():
                                current_one_hot_vector = np.zeros(
                                    len(distinct_ground_truth_labels) + 1
                                )
                                for (
                                    label_index
                                ) in frame_id_blip2_question_index_selected_label_indices_mapping[
                                    frame_id
                                ][
                                    blip2_question_index
                                ]:
                                    current_one_hot_vector[label_index] = 1
                                predictions_one_hot_vectors_dict[clip_id][frame_id][
                                    blip2_question_index
                                ] = current_one_hot_vector

            with open(args.ground_truth_file_path, "rb") as reader:
                ground_truths = pickle.load(reader)

            ground_truth_one_hot_vectors_dict = dict()
            for clip_id, frame_id_ground_truth_labels_mapping in ground_truths.items():
                ground_truth_one_hot_vectors_dict[clip_id] = dict()
                for (
                    frame_id,
                    ground_truth_label_indices,
                ) in frame_id_ground_truth_labels_mapping.items():
                    current_one_hot_vector = np.zeros(
                        len(distinct_ground_truth_labels) + 1
                    )
                    for ground_truth_label_index in ground_truth_label_indices:
                        current_one_hot_vector[ground_truth_label_index] = 1
                    ground_truth_one_hot_vectors_dict[clip_id][
                        frame_id
                    ] = current_one_hot_vector

            ground_truth_one_hot_vectors_list = []
            question_index_predicted_one_hot_vectors_list_mapping = dict()
            for clip_id in ground_truth_one_hot_vectors_dict.keys():
                for frame_id in ground_truth_one_hot_vectors_dict[clip_id].keys():
                    ground_truth_one_hot_vectors_list.append(
                        ground_truth_one_hot_vectors_dict[clip_id][frame_id]
                    )
                    for question_index in predictions_one_hot_vectors_dict[clip_id][
                        int((frame_id // 6) * 6)
                    ].keys():
                        if (
                            question_index
                            not in question_index_predicted_one_hot_vectors_list_mapping.keys()
                        ):
                            question_index_predicted_one_hot_vectors_list_mapping[
                                question_index
                            ] = []
                        question_index_predicted_one_hot_vectors_list_mapping[
                            question_index
                        ].append(
                            predictions_one_hot_vectors_dict[clip_id][
                                int((frame_id // 6) * 6)
                            ][question_index]
                        )

            os.makedirs(args.output_folder_path, exist_ok=True)
            for (
                question_index
            ) in question_index_predicted_one_hot_vectors_list_mapping.keys():
                predicted_one_hot_vectors_list = (
                    question_index_predicted_one_hot_vectors_list_mapping[
                        question_index
                    ]
                )
                f1_weighted_average = f1_score(
                    y_true=ground_truth_one_hot_vectors_list,
                    y_pred=predicted_one_hot_vectors_list,
                    average="weighted",
                    zero_division=0,
                )
                precision_weighted_average = precision_score(
                    y_true=ground_truth_one_hot_vectors_list,
                    y_pred=predicted_one_hot_vectors_list,
                    average="weighted",
                    zero_division=0,
                )
                recall_weighted_average = recall_score(
                    y_true=ground_truth_one_hot_vectors_list,
                    y_pred=predicted_one_hot_vectors_list,
                    average="weighted",
                    zero_division=0,
                )

                f1_per_label = f1_score(
                    y_true=ground_truth_one_hot_vectors_list,
                    y_pred=predicted_one_hot_vectors_list,
                    average=None,
                    zero_division=0,
                ).tolist()
                precision_per_label = precision_score(
                    y_true=ground_truth_one_hot_vectors_list,
                    y_pred=predicted_one_hot_vectors_list,
                    average=None,
                    zero_division=0,
                ).tolist()
                recall_per_label = recall_score(
                    y_true=ground_truth_one_hot_vectors_list,
                    y_pred=predicted_one_hot_vectors_list,
                    average=None,
                    zero_division=0,
                ).tolist()

                df = []
                for label, f1, precision, recall in zip(
                    distinct_ground_truth_labels + ["background"],
                    f1_per_label,
                    precision_per_label,
                    recall_per_label,
                ):
                    df.append((label, f1, precision, recall))
                df.append(
                    (
                        "weighted_average",
                        f1_weighted_average,
                        precision_weighted_average,
                        recall_weighted_average,
                    )
                )

                df = pd.DataFrame(
                    data=df,
                    columns=["label", "f1_score", "precision_score", "recall_score"],
                )
                df.to_csv(
                    os.path.join(
                        args.output_folder_path,
                        f"{args.predictions_folder_name}__question_index_{question_index}__threshold_{str(threshold).replace('.', '')}__{temporal_aggregation}.tsv",
                    ),
                    sep="\t",
                )

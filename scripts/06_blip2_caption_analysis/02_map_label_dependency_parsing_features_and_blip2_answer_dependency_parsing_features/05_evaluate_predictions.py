import os
import pdb
import json
import pickle
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from typing import Dict, List, Union


# def median_temporal_aggregation_select_labels(
#     frame_id_blip2_question_index_label_index_max_score_mapping: Dict[
#         int, Dict[int, Dict[int, List[int]]]
#     ],
#     threshold: Union[float, str],
# ):
#     # median filtering
#     updated_frame_id_blip2_question_index_label_index_max_score_mapping = dict()
#     for (
#         frame_id,
#         blip2_question_index_label_index_scores_mapping,
#     ) in frame_id_blip2_question_index_label_index_max_score_mapping.items():
#         updated_frame_id_blip2_question_index_label_index_max_score_mapping[
#             frame_id
#         ] = dict()
#         for (
#             blip2_question_index,
#             label_index_scores_mapping,
#         ) in blip2_question_index_label_index_scores_mapping.items():
#             updated_frame_id_blip2_question_index_label_index_max_score_mapping[
#                 frame_id
#             ][blip2_question_index] = dict()
#             for label_index, current_score in label_index_scores_mapping.items():
#                 if frame_id == 0:
#                     next_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id + 6
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     next_next_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id + 12
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     updated_frame_id_blip2_question_index_label_index_max_score_mapping[
#                         frame_id
#                     ][blip2_question_index][label_index] = np.median(
#                         [current_score, next_score, next_next_score]
#                     )
#                 elif frame_id == 6:
#                     previous_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id - 6
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     next_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id + 6
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     next_next_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id + 12
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     updated_frame_id_blip2_question_index_label_index_max_score_mapping[
#                         frame_id
#                     ][blip2_question_index][label_index] = np.median(
#                         [previous_score, current_score, next_score, next_next_score]
#                     )
#                 elif frame_id == max(
#                     list(
#                         frame_id_blip2_question_index_label_index_max_score_mapping.keys()
#                     )
#                 ):
#                     previous_previous_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id - 12
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     previous_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id - 6
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     updated_frame_id_blip2_question_index_label_index_max_score_mapping[
#                         frame_id
#                     ][blip2_question_index][label_index] = np.median(
#                         [previous_previous_score, previous_score, current_score]
#                     )
#                 elif (
#                     frame_id
#                     == max(
#                         list(
#                             frame_id_blip2_question_index_label_index_max_score_mapping.keys()
#                         )
#                     )
#                     - 6
#                 ):
#                     previous_previous_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id - 12
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     previous_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id - 6
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     next_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id + 6
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     updated_frame_id_blip2_question_index_label_index_max_score_mapping[
#                         frame_id
#                     ][blip2_question_index][label_index] = (
#                         np.median(
#                             [
#                                 previous_previous_score,
#                                 previous_score,
#                                 current_score,
#                                 next_score,
#                             ]
#                         ),
#                     )
#                 else:
#                     previous_previous_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id - 12
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     previous_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id - 6
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     next_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id + 6
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     next_next_score = (
#                         frame_id_blip2_question_index_label_index_max_score_mapping[
#                             frame_id + 12
#                         ][blip2_question_index].get(label_index, 0)
#                     )
#                     updated_frame_id_blip2_question_index_label_index_max_score_mapping[
#                         frame_id
#                     ][blip2_question_index][label_index] = (
#                         np.median(
#                             [
#                                 previous_previous_score,
#                                 previous_score,
#                                 current_score,
#                                 next_score,
#                                 next_next_score,
#                             ]
#                         ),
#                     )

#     # select labels
#     frame_id_blip2_question_index_selected_label_indices_mapping = dict()
#     for (
#         frame_id,
#         blip2_question_index_label_index_max_score_mapping,
#     ) in updated_frame_id_blip2_question_index_label_index_max_score_mapping.items():
#         frame_id_blip2_question_index_selected_label_indices_mapping[frame_id] = dict()
#         for (
#             blip2_question_index,
#             label_index_max_score_mapping,
#         ) in blip2_question_index_label_index_max_score_mapping.items():
#             frame_id_blip2_question_index_selected_label_indices_mapping[frame_id][
#                 blip2_question_index
#             ] = set()
#             for label_index, current_score in label_index_max_score_mapping.items():
#                 if current_score >= threshold:
#                     frame_id_blip2_question_index_selected_label_indices_mapping[
#                         frame_id
#                     ][blip2_question_index].add(label_index)

#     return frame_id_blip2_question_index_selected_label_indices_mapping


def no_temporal_aggregation_select_labels(
    frame_id_blip2_question_index_label_index_max_score_mapping: Dict[
        int, Dict[int, Dict[int, List[int]]]
    ],
    threshold: Union[float, str],
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
            if threshold == "max":
                label_index_with_max_score = None
                max_score = -1
                for label_index, current_score in label_index_max_score_mapping.items():
                    if current_score > max_score:
                        label_index_with_max_score = label_index
                        max_score = current_score
                frame_id_blip2_question_index_selected_label_indices_mapping[frame_id][
                    blip2_question_index
                ] = set([label_index_with_max_score])
            else:
                for label_index, current_score in label_index_max_score_mapping.items():
                    if current_score >= float(threshold):
                        frame_id_blip2_question_index_selected_label_indices_mapping[
                            frame_id
                        ][blip2_question_index].add(label_index)
    return frame_id_blip2_question_index_selected_label_indices_mapping


# def transfusion_temporal_aggregation_select_labels(
#     frame_id_blip2_question_index_label_index_max_score_mapping: Dict[
#         int, Dict[int, Dict[int, List[int]]]
#     ],
#     threshold: float,
# ):
#     # select labels
#     frame_id_blip2_question_index_selected_label_indices_mapping = dict()
#     for (
#         frame_id,
#         blip2_question_index_label_index_max_score_mapping,
#     ) in frame_id_blip2_question_index_label_index_max_score_mapping.items():
#         frame_id_blip2_question_index_selected_label_indices_mapping[frame_id] = dict()
#         for (
#             blip2_question_index,
#             label_index_max_score_mapping,
#         ) in blip2_question_index_label_index_max_score_mapping.items():
#             frame_id_blip2_question_index_selected_label_indices_mapping[frame_id][
#                 blip2_question_index
#             ] = set()
#             for label_index, score in label_index_max_score_mapping.items():
#                 if score >= threshold:
#                     frame_id_blip2_question_index_selected_label_indices_mapping[
#                         frame_id
#                     ][blip2_question_index].add(label_index)

#     # transfusion filtering
#     for (
#         frame_id,
#         blip2_question_index_selected_label_indices_mapping,
#     ) in frame_id_blip2_question_index_selected_label_indices_mapping.items():
#         for (
#             blip2_question_index,
#             selected_label_indices,
#         ) in blip2_question_index_selected_label_indices_mapping.items():
#             for selected_label_index in selected_label_indices:
#                 if (
#                     frame_id
#                     < len(
#                         frame_id_blip2_question_index_selected_label_indices_mapping.keys()
#                     )
#                     - 12
#                 ):
#                     next_selected_label_indices = (
#                         frame_id_blip2_question_index_selected_label_indices_mapping[
#                             frame_id + 6
#                         ][blip2_question_index]
#                     )
#                     next_next_selected_label_indices = (
#                         frame_id_blip2_question_index_selected_label_indices_mapping[
#                             frame_id + 12
#                         ][blip2_question_index]
#                     )
#                     if (selected_label_index in next_next_selected_label_indices) and (
#                         selected_label_index not in next_selected_label_indices
#                     ):
#                         next_selected_label_indices = frame_id_blip2_question_index_selected_label_indices_mapping[
#                             frame_id + 6
#                         ][
#                             blip2_question_index
#                         ].add(
#                             selected_label_index
#                         )

#     return frame_id_blip2_question_index_selected_label_indices_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument(
        "--predictions_folder_name",
        type=str,
        default="ego4d_asl_features_max_per_label_predictions",
        choices=[
            "ego4d_asl_features_max_per_label_predictions",
            "proposed_features_v1_max_per_label_predictions",
            "proposed_features_v2_max_per_label_predictions",
            "proposed_features_v3_max_per_label_predictions",
            "proposed_features_v4_max_per_label_predictions",
            "proposed_features_v5_max_per_label_predictions",
            "proposed_features_v6_max_per_label_predictions",
            "blip2_dictionary_matching_max_per_label_predictions",
            "blip2_sbert_matching_all-distilroberta-v1_max_per_label_predictions",
            "blip2_sbert_matching_paraphrase-MiniLM-L6-v2_max_per_label_predictions",
        ],
    )
    parser.add_argument(
        "--temporal_aggregation",
        type=str,
        choices=[
            "no_temporal_aggregation",
        ],
        default="no_temporal_aggregation",
    )
    parser.add_argument("--threshold", default=0.2, type=float)
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/06_blip2_caption_analysis/02_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features/label_verb_noun_tool_mapping.json",
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
        "--annotations_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/08_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json",
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

    with open(args.annotations_file_path, "rb") as reader:
        annotations = json.load(reader)

    current_split_clip_ids = []
    for clip_id in annotations.keys():
        if annotations[clip_id]["subset"] == args.split:
            current_split_clip_ids.append(clip_id)

    predictions_folder_path = os.path.join(
        os.environ["SCRATCH"],
        f"ego4d_data/v2/analysis_data/{args.predictions_folder_name}",
    )

    with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
        label_verb_noun_tool_mapping = json.load(reader)
    distinct_ground_truth_labels = sorted(list(label_verb_noun_tool_mapping.keys()))

    prediction_one_hot_vectors = []
    ground_truth_one_hot_vectors = []

    predictions_one_hot_vectors_dict = dict()

    for file_name in tqdm(os.listdir(predictions_folder_path)):
        with open(os.path.join(predictions_folder_path, file_name), "rb") as reader:
            current_predictions_max_per_label_postprocessing_results = pickle.load(
                reader
            )

            for (
                clip_id
            ) in current_predictions_max_per_label_postprocessing_results.keys():
                if clip_id in current_split_clip_ids:
                    frame_id_blip2_question_index_label_index_max_score_mapping = (
                        current_predictions_max_per_label_postprocessing_results[
                            clip_id
                        ]
                    )
                    for (
                        frame_id
                    ) in (
                        frame_id_blip2_question_index_label_index_max_score_mapping.keys()
                    ):
                        blip2_question_index_label_index_max_score_mapping = (
                            frame_id_blip2_question_index_label_index_max_score_mapping[
                                frame_id
                            ]
                        )
                        # label_index_max_max_score_mapping = dict()
                        # for (
                        #     blip2_question_index
                        # ) in blip2_question_index_label_index_max_score_mapping.keys():
                        #     label_index_max_score_mapping = (
                        #         blip2_question_index_label_index_max_score_mapping[
                        #             blip2_question_index
                        #         ]
                        #     )
                        #     for label_index in label_index_max_score_mapping.keys():
                        #         if (
                        #             label_index
                        #             not in label_index_max_max_score_mapping.keys()
                        #         ):
                        #             label_index_max_max_score_mapping[
                        #                 label_index
                        #             ] = label_index_max_score_mapping[label_index]
                        #         else:
                        #             label_index_max_max_score_mapping[
                        #                 label_index
                        #             ] = max(
                        #                 label_index_max_score_mapping[label_index],
                        #                 label_index_max_max_score_mapping[label_index],
                        #             )

                        # new_key = "_".join(
                        #     [
                        #         str(key)
                        #         for key in blip2_question_index_label_index_max_score_mapping.keys()
                        #     ]
                        # )

                        # frame_id_blip2_question_index_label_index_max_score_mapping[
                        #     frame_id
                        # ][new_key] = label_index_max_max_score_mapping

                    if args.temporal_aggregation == "no_temporal_aggregation":
                        frame_id_blip2_question_index_selected_label_indices_mapping = no_temporal_aggregation_select_labels(
                            frame_id_blip2_question_index_label_index_max_score_mapping=frame_id_blip2_question_index_label_index_max_score_mapping,
                            threshold=args.threshold,
                        )
                    # elif args.temporal_aggregation == "median_temporal_aggregation":
                    #     frame_id_blip2_question_index_selected_label_indices_mapping = median_temporal_aggregation_select_labels(
                    #         frame_id_blip2_question_index_label_index_max_score_mapping=frame_id_blip2_question_index_label_index_max_score_mapping,
                    #         threshold=args.threshold,
                    #     )
                    # elif (
                    #     args.temporal_aggregation == "transfusion_temporal_aggregation"
                    # ):
                    #     frame_id_blip2_question_index_selected_label_indices_mapping = transfusion_temporal_aggregation_select_labels(
                    #         frame_id_blip2_question_index_label_index_max_score_mapping=frame_id_blip2_question_index_label_index_max_score_mapping,
                    #         threshold=args.threshold,
                    #     )

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
        if clip_id in current_split_clip_ids:
            ground_truth_one_hot_vectors_dict[clip_id] = dict()
            for (
                frame_id,
                ground_truth_label_indices,
            ) in frame_id_ground_truth_labels_mapping.items():
                current_one_hot_vector = np.zeros(len(distinct_ground_truth_labels) + 1)
                for ground_truth_label_index in ground_truth_label_indices:
                    current_one_hot_vector[ground_truth_label_index] = 1
                ground_truth_one_hot_vectors_dict[clip_id][
                    frame_id
                ] = current_one_hot_vector

    pdb.set_trace()

    ground_truth_one_hot_vectors_list_w_background = []
    ground_truth_one_hot_vectors_list_wo_background = []
    question_index_predicted_one_hot_vectors_list_mapping_w_background = dict()
    question_index_predicted_one_hot_vectors_list_mapping_wo_background = dict()
    for clip_id in predictions_one_hot_vectors_dict.keys():
        for frame_id in ground_truth_one_hot_vectors_dict[clip_id].keys():
            ground_truth_one_hot_vectors_list_w_background.append(
                ground_truth_one_hot_vectors_dict[clip_id][frame_id]
            )
            ground_truth_one_hot_vectors_list_wo_background.append(
                ground_truth_one_hot_vectors_dict[clip_id][frame_id][:-1]
            )

            for question_index in predictions_one_hot_vectors_dict[clip_id][
                int((frame_id // 6) * 6)
            ].keys():
                if (
                    question_index
                    not in question_index_predicted_one_hot_vectors_list_mapping_w_background.keys()
                ):
                    question_index_predicted_one_hot_vectors_list_mapping_w_background[
                        question_index
                    ] = []

                if (
                    question_index
                    not in question_index_predicted_one_hot_vectors_list_mapping_wo_background.keys()
                ):
                    question_index_predicted_one_hot_vectors_list_mapping_wo_background[
                        question_index
                    ] = []

                question_index_predicted_one_hot_vectors_list_mapping_w_background[
                    question_index
                ].append(
                    predictions_one_hot_vectors_dict[clip_id][int((frame_id // 6) * 6)][
                        question_index
                    ]
                )

                question_index_predicted_one_hot_vectors_list_mapping_wo_background[
                    question_index
                ].append(
                    predictions_one_hot_vectors_dict[clip_id][int((frame_id // 6) * 6)][
                        question_index
                    ][:-1]
                )

    pdb.set_trace()

    os.makedirs(args.output_folder_path, exist_ok=True)
    for (
        question_index
    ) in question_index_predicted_one_hot_vectors_list_mapping_w_background.keys():
        predicted_one_hot_vectors_list_w_background = (
            question_index_predicted_one_hot_vectors_list_mapping_w_background[
                question_index
            ]
        )
        predicted_one_hot_vectors_list_wo_background = (
            question_index_predicted_one_hot_vectors_list_mapping_wo_background[
                question_index
            ]
        )
        f1_weighted_average_w_background = f1_score(
            y_true=ground_truth_one_hot_vectors_list_w_background,
            y_pred=predicted_one_hot_vectors_list_w_background,
            average="weighted",
            zero_division=0,
        )
        precision_weighted_average_w_background = precision_score(
            y_true=ground_truth_one_hot_vectors_list_w_background,
            y_pred=predicted_one_hot_vectors_list_w_background,
            average="weighted",
            zero_division=0,
        )
        recall_weighted_average_w_background = recall_score(
            y_true=ground_truth_one_hot_vectors_list_w_background,
            y_pred=predicted_one_hot_vectors_list_w_background,
            average="weighted",
            zero_division=0,
        )

        f1_weighted_average_wo_background = f1_score(
            y_true=ground_truth_one_hot_vectors_list_wo_background,
            y_pred=predicted_one_hot_vectors_list_wo_background,
            average="weighted",
            zero_division=0,
        )
        precision_weighted_average_wo_background = precision_score(
            y_true=ground_truth_one_hot_vectors_list_wo_background,
            y_pred=predicted_one_hot_vectors_list_wo_background,
            average="weighted",
            zero_division=0,
        )
        recall_weighted_average_wo_background = recall_score(
            y_true=ground_truth_one_hot_vectors_list_wo_background,
            y_pred=predicted_one_hot_vectors_list_wo_background,
            average="weighted",
            zero_division=0,
        )

        f1_per_label = f1_score(
            y_true=ground_truth_one_hot_vectors_list_w_background,
            y_pred=predicted_one_hot_vectors_list_w_background,
            average=None,
            zero_division=0,
        ).tolist()
        precision_per_label = precision_score(
            y_true=ground_truth_one_hot_vectors_list_w_background,
            y_pred=predicted_one_hot_vectors_list_w_background,
            average=None,
            zero_division=0,
        ).tolist()
        recall_per_label = recall_score(
            y_true=ground_truth_one_hot_vectors_list_w_background,
            y_pred=predicted_one_hot_vectors_list_w_background,
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
                "weighted_average_w_background",
                f1_weighted_average_w_background,
                precision_weighted_average_w_background,
                recall_weighted_average_w_background,
            )
        )
        df.append(
            (
                "weighted_average_wo_background",
                f1_weighted_average_wo_background,
                precision_weighted_average_wo_background,
                recall_weighted_average_wo_background,
            )
        )

        df = pd.DataFrame(
            data=df,
            columns=["label", "f1_score", "precision_score", "recall_score"],
        )
        df.to_csv(
            os.path.join(
                args.output_folder_path,
                f"{args.predictions_folder_name}__question_index_{question_index}__threshold_{str(args.threshold).replace('.', '')}__{args.temporal_aggregation}__split__{args.split}.tsv",
            ),
            sep="\t",
        )

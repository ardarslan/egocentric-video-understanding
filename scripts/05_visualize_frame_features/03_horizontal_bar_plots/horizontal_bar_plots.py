import os
import cv2
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
from dash import Dash, html, dcc, Input, Output, no_update, dash_table

import sys

sys.path.append("../")
from utils import extract_frames

sys.path.append(
    "../../06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features"
)
import constants


from typing import List

random.seed(1903)

ground_truth_predicted_action_category_match_color_mapping = {
    True: "rgba(0, 255, 0, 1.0)",
    False: "rgba(255, 0, 0, 1.0)",
}

unique_action_categories = set(["background", "no_annotations_for_the_clip"])


def generate_random_color():
    random_int = np.random.randint(low=0, high=256, size=(3,))
    random_color = f"rgba({random_int[0]}, {random_int[1]}, {random_int[2]}, 1.0)"
    return random_color


def get_blip2_answer(current_blip2_rows, blip2_question):
    answer = current_blip2_rows[current_blip2_rows["question"] == blip2_question][
        "answer"
    ]
    if len(answer) == 0:
        return "NaN"
    else:
        return answer.values[0]


def concatenate_labels(labels: List[str]):
    return " + ".join(sorted(list(set(labels))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--clip_id",
        type=str,
        default="013559ff-eab2-4c25-a475-90bf56f5ae9e",  # "003c5ae8-3abd-4824-8efb-21a9a4f8eafe",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=8053,
    )
    parser.add_argument(
        "--label_verb_noun_tool_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["CODE"],
            "scripts/06_analyze_frame_features/03_map_label_dependency_parsing_features_and_blip2_answer_dependency_parsing_features",
            "label_verb_noun_tool_mapping.json",
        ),
    )
    parser.add_argument(
        "--ground_truth_action_instances_file_path",
        type=str,
        default=f"{os.environ['CODE']}/scripts/07_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json",
    )
    parser.add_argument(
        "--asl_predicted_action_instances_file_path",
        type=str,
        default=f"{os.environ['CODE']}/scripts/07_reproduce_mq_experiments/submission_final.json",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        choices=["asl", "blip2_dictionary_matching", "blip2_sbert_matching"],
        default="blip2_dictionary_matching",
    )
    parser.add_argument(
        "--question_index",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6],
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--clip_id_file_name_mapping_file_path",
        type=str,
        default=os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/analysis_data/max_per_label_predictions_clip_id_file_name_mapping.tsv",
        ),
    )
    parser.add_argument(
        "--blip2_dictionary_matching_predicted_action_instances_folder_path",
        type=str,
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/analysis_data/blip2_dictionary_matching_max_per_label_predictions",
    )
    parser.add_argument(
        "--blip2_sbert_matching_predicted_action_instances_folder_path",
        type=str,
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/analysis_data/blip2_sbert_matching_max_per_label_predictions",
    )
    parser.add_argument(
        "--dependency_parsing_results_file_path",
        type=str,
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/analysis_data/dependency_parsing_results/dependency_parsing_results.pickle",
    )
    parser.add_argument(
        "--assets_path",
        type=str,
        default=f"{os.environ['SCRATCH']}/ego4d_data/v2/frames",
    )
    args = parser.parse_args()

    extract_frames(clip_id=args.clip_id, output_folder_path=args.assets_path)

    with open(
        os.path.join(
            os.environ["CODE"],
            "scripts/07_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json",
        ),
        "r",
    ) as reader:
        annotations_dict = json.load(reader)

    if (
        not os.path.exists(
            os.path.join(
                os.environ["SCRATCH"],
                "ego4d_data/v2/frames",
                args.clip_id,
                "end.txt",
            )
        )
        or not (len(annotations_dict[args.clip_id]["annotations"]) > 0)
        or not os.path.exists(
            os.path.join(
                os.environ["SCRATCH"],
                "ego4d_data/v2/frame_features",
                args.clip_id,
                "blip2_vqa_features.tsv",
            )
        )
    ):
        raise Exception("Please choose another clip.")

    blip2_answers_folder_path = os.path.join(
        os.environ["SCRATCH"], "ego4d_data/v2/frame_features", args.clip_id
    )
    blip2_answers_file_names = [
        file_name
        for file_name in os.listdir(blip2_answers_folder_path)
        if file_name.startswith("blip2_")
    ]
    blip2_answers_file_paths = [
        os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/frame_features",
            args.clip_id,
            blip2_answers_file_name,
        )
        for blip2_answers_file_name in blip2_answers_file_names
    ]
    blip2_answers_dfs = pd.concat(
        [
            pd.read_csv(blip2_answers_file_path, sep="\t")
            for blip2_answers_file_path in blip2_answers_file_paths
        ],
        axis=0,
    )

    with open(args.dependency_parsing_results_file_path, "rb") as reader:
        dependency_parsing_results = pickle.load(reader)[args.clip_id]

    cap = cv2.VideoCapture(
        os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/clips", args.clip_id + ".mp4"
        )
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # ground truths
    ground_truth_action_instances = json.load(
        open(args.ground_truth_action_instances_file_path, "r")
    )[args.clip_id]["annotations"]
    frame_id_ground_truth_action_categories_mapping = {}
    for current_frame_id in range(num_frames):
        current_frame_time = current_frame_id / fps

        if len(ground_truth_action_instances) == 0:
            frame_id_ground_truth_action_categories_mapping[
                current_frame_id
            ] = "no_annotations_for_the_clip"
            unique_action_categories.add("no_annotations_for_the_clip")
        else:
            current_labels = []
            for ground_truth_action_instance in ground_truth_action_instances:
                if (
                    current_frame_time >= ground_truth_action_instance["segment"][0]
                    and current_frame_time <= ground_truth_action_instance["segment"][1]
                ):
                    assigned_to_an_action_category = True
                    current_label = ground_truth_action_instance["label"]
                    current_labels.append(current_label)

            if len(current_labels) > 0:
                concatenated_labels = concatenate_labels(current_labels)
                frame_id_ground_truth_action_categories_mapping[
                    current_frame_id
                ] = concatenated_labels
                unique_action_categories.add(concatenated_labels)
            else:
                frame_id_ground_truth_action_categories_mapping[
                    current_frame_id
                ] = "background"
                unique_action_categories.add("background")

    # predictions
    frame_id_predicted_action_categories_mapping = {}
    if args.prediction_type == "asl":
        frame_feature_extraction_stride = 1
        predicted_action_instances = json.load(
            open(args.asl_predicted_action_instances_file_path, "r")
        )["detect_results"][args.clip_id]

        for current_frame_id in range(num_frames):
            frame_id_predicted_action_categories_mapping[current_frame_id] = []
            current_frame_time = current_frame_id / fps
            assigned_to_an_action_category = False

            # asl predictions
            for predicted_action_instance in predicted_action_instances:
                if (
                    current_frame_time >= predicted_action_instance["segment"][0]
                    and current_frame_time <= predicted_action_instance["segment"][1]
                ):
                    if predicted_action_instance["score"] >= args.threshold:
                        assigned_to_an_action_category = True
                        frame_id_predicted_action_categories_mapping[
                            current_frame_id
                        ].append(
                            (
                                predicted_action_instance["label"],
                                predicted_action_instance["score"],
                            )
                        )
            frame_id_predicted_action_categories_mapping[current_frame_id] = sorted(
                frame_id_predicted_action_categories_mapping[current_frame_id],
                key=lambda x: x[0],
            )
            unique_action_categories.add(
                concatenate_labels(
                    [
                        label
                        for label, _ in frame_id_predicted_action_categories_mapping[
                            current_frame_id
                        ]
                    ]
                )
            )

            if not assigned_to_an_action_category:
                frame_id_predicted_action_categories_mapping[current_frame_id] = [
                    (
                        "background",
                        1.0,
                    )
                ]
                unique_action_categories.add("background")
    elif args.prediction_type == "blip2_dictionary_matching":
        frame_feature_extraction_stride = 6
        with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
            label_verb_noun_tool_mapping = json.load(reader)
        distinct_ground_truth_labels = sorted(
            list(label_verb_noun_tool_mapping.keys())
        ) + ["background"]
        clip_id_file_name_mapping = dict(
            pd.read_csv(args.clip_id_file_name_mapping_file_path, sep="\t").values
        )
        file_name = clip_id_file_name_mapping[args.clip_id]
        file_path = os.path.join(
            args.blip2_dictionary_matching_predicted_action_instances_folder_path,
            file_name,
        )
        with open(file_path, "rb") as reader:
            current_clip_id_frame_id_blip2_question_index_label_index_max_score_mapping = pickle.load(
                reader
            )
        if (
            args.clip_id
            in current_clip_id_frame_id_blip2_question_index_label_index_max_score_mapping.keys()
        ):
            frame_id_blip2_question_index_label_index_max_score_mapping = current_clip_id_frame_id_blip2_question_index_label_index_max_score_mapping[
                args.clip_id
            ]
            for (
                current_frame_id,
                blip2_question_index_label_index_max_score_mapping,
            ) in frame_id_blip2_question_index_label_index_max_score_mapping.items():
                assigned_to_an_action_category = False
                frame_id_predicted_action_categories_mapping[current_frame_id] = []
                label_index_max_score_mapping = (
                    blip2_question_index_label_index_max_score_mapping[
                        args.question_index
                    ]
                )
                for label_index, max_score in label_index_max_score_mapping.items():
                    if max_score[1] >= args.threshold:
                        assigned_to_an_action_category = True
                        label = distinct_ground_truth_labels[label_index]
                        frame_id_predicted_action_categories_mapping[
                            current_frame_id
                        ].append(
                            (
                                label,
                                max_score[1],
                            )
                        )
                frame_id_predicted_action_categories_mapping[current_frame_id] = sorted(
                    frame_id_predicted_action_categories_mapping[current_frame_id],
                    key=lambda x: x[0],
                )
                unique_action_categories.add(
                    concatenate_labels(
                        [
                            label
                            for label, _ in frame_id_predicted_action_categories_mapping[
                                current_frame_id
                            ]
                        ]
                    )
                )
                if not assigned_to_an_action_category:
                    frame_id_predicted_action_categories_mapping[current_frame_id] = [
                        (
                            "background",
                            1.0,
                        )
                    ]
                unique_action_categories.add("background")
    elif args.prediction_type == "blip2_sbert_matching":
        frame_feature_extraction_stride = 6
        with open(args.label_verb_noun_tool_mapping_file_path, "r") as reader:
            label_verb_noun_tool_mapping = json.load(reader)
        distinct_ground_truth_labels = sorted(
            list(label_verb_noun_tool_mapping.keys())
        ) + ["background"]
        for file_name in os.listdir(
            args.blip2_sbert_matching_predicted_action_instances_folder_path
        ):
            file_path = os.path.join(
                args.blip2_sbert_matching_predicted_action_instances_folder_path,
                file_name,
            )
            with open(file_path, "rb") as reader:
                current_clip_id_frame_id_blip2_question_index_label_index_max_score_mapping = pickle.load(
                    reader
                )
            if (
                args.clip_id
                in current_clip_id_frame_id_blip2_question_index_label_index_max_score_mapping.keys()
            ):
                frame_id_blip2_question_index_label_index_max_score_mapping = current_clip_id_frame_id_blip2_question_index_label_index_max_score_mapping[
                    args.clip_id
                ]
                for (
                    current_frame_id,
                    blip2_question_index_label_index_max_score_mapping,
                ) in (
                    frame_id_blip2_question_index_label_index_max_score_mapping.items()
                ):
                    assigned_to_an_action_category = False
                    frame_id_predicted_action_categories_mapping[current_frame_id] = []
                    label_index_max_score_mapping = (
                        blip2_question_index_label_index_max_score_mapping[
                            args.question_index
                        ]
                    )
                    for label_index, max_score in label_index_max_score_mapping.items():
                        if max_score[1] >= args.threshold:
                            label = distinct_ground_truth_labels[label_index]
                            frame_id_predicted_action_categories_mapping[
                                current_frame_id
                            ].append(
                                (
                                    label,
                                    max_score[1],
                                )
                            )
                    frame_id_predicted_action_categories_mapping[
                        current_frame_id
                    ] = sorted(
                        frame_id_predicted_action_categories_mapping[current_frame_id],
                        key=lambda x: x[0],
                    )
                    unique_action_categories.add(
                        concatenate_labels(
                            [
                                label
                                for label, _ in frame_id_predicted_action_categories_mapping[
                                    current_frame_id
                                ]
                            ]
                        )
                    )
                    if not assigned_to_an_action_category:
                        frame_id_predicted_action_categories_mapping[
                            current_frame_id
                        ] = [
                            (
                                "background",
                                1.0,
                            )
                        ]
                    unique_action_categories.add("background")
            else:
                continue

    action_category_color_mapping = dict(
        (action_category, generate_random_color())
        for action_category in sorted(list(unique_action_categories))
    )

    sequences_dict = {
        "gt_colors": [],
        "pred_colors": [],
        "match_colors": [],
        "gt_values": [],
        "pred_values": [],
        "match_values": [],
        "frame_ids": [],
        "blip2_happen_dependency_parsing_features": [],
        "blip2_do_dependency_parsing_features": [],
        "blip2_describe_dependency_parsing_features": [],
    }

    blip2_describe_question = "What does the image describe?"
    blip2_do_question = "What is the person in this picture doing?"
    blip2_happen_question = "What is happening in this picture?"

    for frame_id in range(num_frames):
        current_blip2_rows = blip2_answers_dfs[
            blip2_answers_dfs["frame_index"]
            == (frame_id // frame_feature_extraction_stride)
            * frame_feature_extraction_stride
        ]
        current_blip2_describe_answer = get_blip2_answer(
            current_blip2_rows=current_blip2_rows,
            blip2_question=blip2_describe_question,
        )
        current_blip2_do_answer = get_blip2_answer(
            current_blip2_rows=current_blip2_rows, blip2_question=blip2_do_question
        )
        current_blip2_happen_answer = get_blip2_answer(
            current_blip2_rows=current_blip2_rows,
            blip2_question=blip2_happen_question,
        )
        current_blip2_happen_answer_dependency_parsing_features = (
            dependency_parsing_results[
                int(
                    frame_id
                    // frame_feature_extraction_stride
                    * frame_feature_extraction_stride
                )
            ][constants.question_constant_mapping[blip2_happen_question]]
        )
        current_blip2_do_answer_dependency_parsing_features = (
            dependency_parsing_results[
                int(
                    frame_id
                    // frame_feature_extraction_stride
                    * frame_feature_extraction_stride
                )
            ][constants.question_constant_mapping[blip2_do_question]]
        )
        current_blip2_describe_answer_dependency_parsing_features = (
            dependency_parsing_results[
                int(
                    frame_id
                    // frame_feature_extraction_stride
                    * frame_feature_extraction_stride
                )
            ][constants.question_constant_mapping[blip2_describe_question]]
        )
        sequences_dict["frame_ids"].append(frame_id)
        sequences_dict["gt_values"].append(
            frame_id_ground_truth_action_categories_mapping[frame_id]
        )
        sequences_dict["gt_colors"].append(
            action_category_color_mapping[
                frame_id_ground_truth_action_categories_mapping[frame_id]
            ]
        )
        sequences_dict["blip2_happen_dependency_parsing_features"].append(
            current_blip2_happen_answer_dependency_parsing_features
        )
        sequences_dict["blip2_do_dependency_parsing_features"].append(
            current_blip2_do_answer_dependency_parsing_features
        )
        sequences_dict["blip2_describe_dependency_parsing_features"].append(
            current_blip2_describe_answer_dependency_parsing_features
        )

        sequences_dict["pred_values"].append(
            " + ".join(
                sorted(
                    [
                        f"{label} ({np.round(score, 2)})"
                        for label, score in frame_id_predicted_action_categories_mapping[
                            int(
                                (frame_id // frame_feature_extraction_stride)
                                * frame_feature_extraction_stride
                            )
                        ]
                    ]
                )
            )
        )
        sequences_dict["pred_colors"].append(
            action_category_color_mapping[
                concatenate_labels(
                    [
                        label
                        for label, _ in frame_id_predicted_action_categories_mapping[
                            int(
                                (frame_id // frame_feature_extraction_stride)
                                * frame_feature_extraction_stride
                            )
                        ]
                    ]
                )
            ]
        )

        current_ground_truth_predicted_action_category_match = False
        for predicted_label, _ in frame_id_predicted_action_categories_mapping[
            int(
                (frame_id // frame_feature_extraction_stride)
                * frame_feature_extraction_stride
            )
        ]:
            for ground_truth_label in frame_id_ground_truth_action_categories_mapping[
                frame_id
            ].split(" + "):
                if ground_truth_label == predicted_label:
                    current_ground_truth_predicted_action_category_match = True
        sequences_dict["match_values"].append(
            current_ground_truth_predicted_action_category_match
        )

        current_ground_truth_predicted_action_category_match_color = (
            ground_truth_predicted_action_category_match_color_mapping[
                current_ground_truth_predicted_action_category_match
            ]
        )
        sequences_dict["match_colors"].append(
            current_ground_truth_predicted_action_category_match_color
        )

    sequences_dict["frame_file_paths"] = [
        os.path.join(
            args.clip_id,
            frame_file_name,
        )
        for frame_file_name in sorted(
            os.listdir(
                os.path.join(
                    os.environ["SCRATCH"], "ego4d_data/v2/frames", args.clip_id
                )
            )
        )
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                orientation="h",
                x=[1] * num_frames,
                y=[name] * num_frames,
                marker=dict(
                    color=sequences_dict[f"{name}_colors"],
                    line=dict(color="rgb(255, 255, 255)", width=0),
                ),
                customdata=list(
                    zip(
                        sequences_dict["frame_file_paths"],
                        sequences_dict["frame_ids"],
                        sequences_dict["gt_values"],
                        sequences_dict["pred_values"],
                        sequences_dict["match_values"],
                        sequences_dict["blip2_describe_dependency_parsing_features"],
                        sequences_dict["blip2_do_dependency_parsing_features"],
                        sequences_dict["blip2_happen_dependency_parsing_features"],
                    )
                ),
            )
            for name in ["match", "pred", "gt"]
        ],
        layout=dict(
            title=f"Clip ID: {args.clip_id}",
            barmode="stack",
            barnorm="fraction",
            bargap=0.5,
            showlegend=False,
            xaxis=dict(range=[-0.02, 1.02], showticklabels=False, showgrid=False),
            height=max(600, 40 * len(sequences_dict.keys())),
            template=None,
            margin=dict(b=1),
        ),
    )

    fig.update_traces(hoverinfo="none", hovertemplate=None)
    fig.update_layout()

    app = Dash(__name__, assets_folder=args.assets_path)

    app.layout = html.Div(
        [
            dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip"),
        ]
    )

    @app.callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-2", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        bbox = hoverData["points"][0]["bbox"]

        prediction_str = f"Prediction Type: {args.prediction_type}"

        if args.prediction_type != "asl":
            for key, value in constants.question_constant_mapping.items():
                if value == args.question_index:
                    question = key
                    break
            prediction_str += f", Question: {question}"

        children = [
            html.Div(
                [
                    html.Img(
                        src=app.get_asset_url(hoverData["points"][0]["customdata"][0]),
                        style={"width": "50%"},
                    ),
                    dash_table.DataTable(
                        data=[
                            {
                                "Frame ID": hoverData["points"][0]["customdata"][1],
                                "Ground Truth": hoverData["points"][0]["customdata"][2],
                                f"Prediction ({prediction_str})": str(
                                    hoverData["points"][0]["customdata"][3]
                                ).replace("_", " "),
                                "Match": str(
                                    hoverData["points"][0]["customdata"][4]
                                ).replace("_", " "),
                                "What does the image describe?": str(
                                    hoverData["points"][0]["customdata"][5]
                                ),
                                "What is the person in this picture doing?": str(
                                    hoverData["points"][0]["customdata"][6]
                                ),
                                "What is happening in this picture?": str(
                                    hoverData["points"][0]["customdata"][7]
                                ),
                            }
                        ],
                        columns=[
                            {"id": c, "name": c}
                            for c in [
                                "Frame ID",
                                "Ground Truth",
                                f"Prediction ({prediction_str})",
                                "Match",
                                "What does the image describe?",
                                "What is the person in this picture doing?",
                                "What is happening in this picture?",
                            ]
                        ],
                        style_cell={
                            "textAlign": "center",
                            "whiteSpace": "pre-line",
                            "font-size": "10px",
                        },
                    ),
                ],
                style={
                    "width": "600px",
                    "white-space": "normal",
                    "textAlign": "center",
                },
            )
        ]

        return True, bbox, children

    app.run_server(debug=True, port=args.port)

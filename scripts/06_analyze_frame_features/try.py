import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random
from dash import Dash, html, dcc, Input, Output, no_update

random.seed(1903)

ground_truth_asl_predicted_action_category_match_color_mapping = {
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--clip_id",
        type=str,
        default="00182baf-e3fe-4bee-9416-825555bc4506",
    )
    parser.add_argument(
        "--ground_truth_action_instances_file_path",
        type=str,
        default=f"{os.environ['CODE']}/scripts/07_reproduce_baseline_results/data/ego4d/ego4d_clip_annotations_v3.json",
    )
    parser.add_argument(
        "--asl_predicted_action_instances_file_path",
        type=str,
        default=f"{os.environ['CODE']}/scripts/07_reproduce_baseline_results/submission_final.json",
    )
    parser.add_argument("--frame_feature_extraction_stride", type=int, default=6)
    args = parser.parse_args()

    ground_truth_action_instances = json.load(
        open(args.ground_truth_action_instances_file_path, "r")
    )[args.clip_id]["annotations"]
    asl_predicted_action_instances = json.load(
        open(args.asl_predicted_action_instances_file_path, "r")
    )["detect_results"][args.clip_id]
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

    cap = cv2.VideoCapture(
        os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/clips", args.clip_id + ".mp4"
        )
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    clip_id_for_last_image = None
    if os.path.exists(
        f"{os.environ['CODE']}/scripts/06_analyze_frame_features/assets/end.txt"
    ):
        with open(
            f"{os.environ['CODE']}/scripts/06_analyze_frame_features/assets/end.txt",
            "r",
        ) as reader:
            clip_id_for_last_image = reader.read().strip()

    if clip_id_for_last_image != args.clip_id:
        os.system(
            f"rm -rf {os.environ['CODE']}/scripts/06_analyze_frame_features/assets"
        )
        os.makedirs(
            f"{os.environ['CODE']}/scripts/06_analyze_frame_features/assets",
            exist_ok=True,
        )
        success = True
        frame_id = 0
        while success:
            success, frame = cap.read()
            if not success:
                break
            cv2.imwrite(
                os.path.join(
                    f"{os.environ['CODE']}/scripts/06_analyze_frame_features/assets",
                    str(frame_id).zfill(6) + ".jpg",
                ),
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    80,
                ],
            )
            frame_id += 1
        with open(
            f"{os.environ['CODE']}/scripts/06_analyze_frame_features/assets/end.txt",
            "w",
        ) as writer:
            writer.write(args.clip_id)

    frame_id_ground_truth_action_categories_mapping = {}
    frame_id_asl_predicted_action_categories_mapping = {}

    for current_frame_id in range(num_frames):
        frame_id_asl_predicted_action_categories_mapping[current_frame_id] = []
        current_frame_time = current_frame_id / fps
        assigned_to_an_action_category = False
        for asl_predicted_action_instance in asl_predicted_action_instances:
            if (
                current_frame_time >= asl_predicted_action_instance["segment"][0]
                and current_frame_time <= asl_predicted_action_instance["segment"][1]
            ):
                assigned_to_an_action_category = True
                frame_id_asl_predicted_action_categories_mapping[
                    current_frame_id
                ].append(
                    (
                        asl_predicted_action_instance["label"],
                        asl_predicted_action_instance["score"],
                    )
                )
                unique_action_categories.add(asl_predicted_action_instance["label"])
        if assigned_to_an_action_category:
            frame_id_asl_predicted_action_categories_mapping[current_frame_id] = sorted(
                frame_id_asl_predicted_action_categories_mapping[current_frame_id],
                key=lambda x: x[1],
            )[-1][0]
        else:
            frame_id_asl_predicted_action_categories_mapping[
                current_frame_id
            ] = "background"

        if len(ground_truth_action_instances) == 0:
            frame_id_ground_truth_action_categories_mapping[
                current_frame_id
            ] = "no_annotations_for_the_clip"
        else:
            assigned_to_an_action_category = False
            for ground_truth_action_instance in ground_truth_action_instances:
                if (
                    current_frame_time >= ground_truth_action_instance["segment"][0]
                    and current_frame_time <= ground_truth_action_instance["segment"][1]
                ):
                    assigned_to_an_action_category = True
                    frame_id_ground_truth_action_categories_mapping[
                        current_frame_id
                    ] = ground_truth_action_instance["label"]
                    unique_action_categories.add(ground_truth_action_instance["label"])
            if not assigned_to_an_action_category:
                frame_id_ground_truth_action_categories_mapping[
                    current_frame_id
                ] = "background"

    action_category_color_mapping = dict(
        (action_category, generate_random_color())
        for action_category in sorted(list(unique_action_categories))
    )

    sequences_dict = {
        "match": {
            "values": [],
            "frame_features": [],
            "colors": [],
            "frame_ids": [],
            "blip2_happen_answers": [],
            "blip2_do_answers": [],
            "blip2_describe_answers": [],
            "blip2_captioning_answers": [],
        },
        "asl_pred": {
            "values": [],
            "frame_features": [],
            "colors": [],
            "frame_ids": [],
            "blip2_happen_answers": [],
            "blip2_do_answers": [],
            "blip2_describe_answers": [],
            "blip2_captioning_answers": [],
        },
        "gt": {
            "values": [],
            "frame_features": [],
            "colors": [],
            "frame_ids": [],
            "blip2_happen_answers": [],
            "blip2_do_answers": [],
            "blip2_describe_answers": [],
            "blip2_captioning_answers": [],
        },
    }

    blip2_describe_question = "What does the image describe?"
    blip2_do_question = "What is the person in this picture doing?"
    blip2_happen_question = "What is happening in this picture?"
    blip2_captioning_question = "Image Caption"

    for frame_id in range(num_frames):
        current_blip2_rows = blip2_answers_dfs[
            blip2_answers_dfs["frame_index"]
            == (frame_id // args.frame_feature_extraction_stride)
            * args.frame_feature_extraction_stride
        ]
        current_blip2_describe_answer = get_blip2_answer(
            current_blip2_rows=current_blip2_rows,
            blip2_question=blip2_describe_question,
        )
        current_blip2_do_answer = get_blip2_answer(
            current_blip2_rows=current_blip2_rows, blip2_question=blip2_do_question
        )
        current_blip2_happen_answer = get_blip2_answer(
            current_blip2_rows=current_blip2_rows, blip2_question=blip2_happen_question
        )
        current_blip2_captioning_answer = get_blip2_answer(
            current_blip2_rows=current_blip2_rows,
            blip2_question=blip2_captioning_question,
        )

        current_ground_truth_action_category = (
            frame_id_ground_truth_action_categories_mapping[frame_id]
        )
        sequences_dict["gt"]["frame_ids"].append(frame_id)
        sequences_dict["gt"]["values"].append(current_ground_truth_action_category)
        current_ground_truth_action_category_color = action_category_color_mapping[
            current_ground_truth_action_category
        ]
        sequences_dict["gt"]["colors"].append(
            current_ground_truth_action_category_color
        )
        sequences_dict["gt"]["blip2_happen_answers"].append(current_blip2_happen_answer)
        sequences_dict["gt"]["blip2_do_answers"].append(current_blip2_do_answer)
        sequences_dict["gt"]["blip2_describe_answers"].append(
            current_blip2_describe_answer
        )
        sequences_dict["gt"]["blip2_captioning_answers"].append(
            current_blip2_captioning_answer
        )

        current_asl_predicted_action_category = (
            frame_id_asl_predicted_action_categories_mapping[frame_id]
        )
        current_asl_predicted_action_category_color = action_category_color_mapping[
            current_asl_predicted_action_category
        ]
        sequences_dict["asl_pred"]["frame_ids"].append(frame_id)
        sequences_dict["asl_pred"]["values"].append(
            current_asl_predicted_action_category
        )
        sequences_dict["asl_pred"]["colors"].append(
            current_asl_predicted_action_category_color
        )
        sequences_dict["asl_pred"]["blip2_happen_answers"].append(
            current_blip2_happen_answer
        )
        sequences_dict["asl_pred"]["blip2_do_answers"].append(current_blip2_do_answer)
        sequences_dict["asl_pred"]["blip2_describe_answers"].append(
            current_blip2_describe_answer
        )
        sequences_dict["asl_pred"]["blip2_captioning_answers"].append(
            current_blip2_captioning_answer
        )

        current_ground_truth_asl_predicted_action_category_match = (
            current_ground_truth_action_category
            == current_asl_predicted_action_category
        )
        current_ground_truth_asl_predicted_action_category_match_color = (
            ground_truth_asl_predicted_action_category_match_color_mapping[
                current_ground_truth_asl_predicted_action_category_match
            ]
        )
        sequences_dict["match"]["frame_ids"].append(frame_id)
        sequences_dict["match"]["values"].append(
            current_ground_truth_asl_predicted_action_category_match
        )
        sequences_dict["match"]["colors"].append(
            current_ground_truth_asl_predicted_action_category_match_color
        )
        sequences_dict["match"]["blip2_happen_answers"].append(
            current_blip2_happen_answer
        )
        sequences_dict["match"]["blip2_do_answers"].append(current_blip2_do_answer)
        sequences_dict["match"]["blip2_describe_answers"].append(
            current_blip2_describe_answer
        )
        sequences_dict["match"]["blip2_captioning_answers"].append(
            current_blip2_captioning_answer
        )

    frame_file_names = list(
        os.listdir(
            os.path.join(os.environ["CODE"], "scripts/06_analyze_frame_features/assets")
        )
    )
    for key in sequences_dict.keys():
        sequences_dict[key]["frame_file_names"] = frame_file_names

    fig = go.Figure(
        data=[
            go.Bar(
                orientation="h",
                x=[1] * num_frames,
                y=[name] * num_frames,
                marker=dict(
                    color=data_dict["colors"],
                    line=dict(color="rgb(255, 255, 255)", width=0),
                ),
                customdata=list(
                    zip(
                        data_dict["frame_file_names"],
                        data_dict["frame_ids"],
                        data_dict["values"],
                        data_dict["blip2_happen_answers"],
                        data_dict["blip2_do_answers"],
                        data_dict["blip2_describe_answers"],
                        data_dict["blip2_captioning_answers"],
                    )
                ),
            )
            for name, data_dict in sequences_dict.items()
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

    app = Dash(__name__)

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

    children = [
        html.Div(
            [
                html.Img(
                    src=app.get_asset_url(hoverData["points"][0]["customdata"][0]),
                    style={"width": "100%"},
                ),
                html.P(f"Frame ID: {hoverData['points'][0]['customdata'][1]}"),
                html.P(f"Value: {hoverData['points'][0]['customdata'][2]}"),
                html.P(
                    f"BLIP2Describe Answer: {hoverData['points'][0]['customdata'][3]}"
                ),
                html.P(f"BLIP2Do Answer: {hoverData['points'][0]['customdata'][4]}"),
                html.P(
                    f"BLIP2Happen Answer: {hoverData['points'][0]['customdata'][5]}"
                ),
                html.P(
                    f"BLIP2Captioning Answer: {hoverData['points'][0]['customdata'][6]}"
                ),
            ],
            style={"width": "200px", "white-space": "normal"},
        )
    ]

    return True, bbox, children


if __name__ == "__main__":
    app.run_server(debug=True)

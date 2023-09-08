import os
import cv2
import json
import argparse
import numpy as np
import plotly.graph_objects as go
import random

random.seed(1903)


ground_truth_asl_predicted_action_category_match_color_mapping = {
    True: "rgba(0, 255, 0, 1.0)",
    False: "rgba(255, 0, 0, 1.0)",
}

unique_action_categories = set(["background"])


def generate_random_color():
    random_int = np.random.randint(low=0, high=256, size=(3,))
    random_color = f"rgba({random_int[0]}, {random_int[1]}, {random_int[2]}, 1.0)"
    return random_color


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument(
        "--clip_id",
        type=str,
        default="00462c6f-c50f-4005-b3a7-6253fa6e9cc3",
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
    args = parser.parse_args()

    ground_truth_action_instances = json.load(
        open(args.ground_truth_action_instances_file_path, "r")
    )[args.clip_id]["annotations"]
    asl_predicted_action_instances = json.load(
        open(args.asl_predicted_action_instances_file_path, "r")
    )["detect_results"][args.clip_id]

    cap = cv2.VideoCapture(
        os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/clips", args.clip_id + ".mp4"
        )
    )
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        "match": {"frame_file_paths": [], "colors": [], "values": []},
        "asl_pred": {"frame_file_paths": [], "colors": [], "values": []},
        "gt": {"frame_file_paths": [], "colors": [], "values": []},
    }

    for frame_id in range(num_frames):
        current_ground_truth_action_category = (
            frame_id_ground_truth_action_categories_mapping[frame_id]
        )
        current_ground_truth_action_category_color = action_category_color_mapping[
            current_ground_truth_action_category
        ]
        sequences_dict["gt"]["values"].append(current_ground_truth_action_category)
        sequences_dict["gt"]["colors"].append(
            current_ground_truth_action_category_color
        )

        current_asl_predicted_action_category = (
            frame_id_asl_predicted_action_categories_mapping[frame_id]
        )
        current_asl_predicted_action_category_color = action_category_color_mapping[
            current_asl_predicted_action_category
        ]
        sequences_dict["asl_pred"]["values"].append(
            current_asl_predicted_action_category
        )
        sequences_dict["asl_pred"]["colors"].append(
            current_asl_predicted_action_category_color
        )

        current_ground_truth_asl_predicted_action_category_match = (
            current_ground_truth_action_category
            == current_asl_predicted_action_category
        )
        sequences_dict["match"]["values"].append(
            current_ground_truth_asl_predicted_action_category_match
        )
        current_ground_truth_asl_predicted_action_category_match_color = (
            ground_truth_asl_predicted_action_category_match_color_mapping[
                current_ground_truth_asl_predicted_action_category_match
            ]
        )
        sequences_dict["match"]["colors"].append(
            current_ground_truth_asl_predicted_action_category_match_color
        )

    fig = go.Figure(
        data=[
            go.Bar(
                orientation="h",
                y=[frame_id for frame_id in range(len(data_dict["colors"]))],
                x=[1] * len(data_dict["colors"]),
                customdata=data_dict["frame_file_paths"],
                text=data_dict["values"],
                marker=dict(
                    color=data_dict["colors"],
                    line=dict(color="rgb(255, 255, 255)", width=0),
                ),
                hovertemplate="<b>Frame ID:</b> $%{y}<br>"
                + f"<b>{name} value:</b>"
                + " $%{text}<br>"
                + "<a href={customdata}>"
                + '<img alt="Current Frame" src={customdata}>'
                + "</a>",
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

    frame_file_paths = [
        os.path.join(
            os.environ["SCRATCH"],
            "ego4d_data/v2/frames",
            args.clip_id,
            frame_file_name,
        )
        for frame_file_name in os.listdir(
            os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/frames", args.clip_id)
        )
    ]
    for key in sequences_dict.keys():
        sequences_dict[key]["frame_file_paths"] = frame_file_paths

    fig.show()

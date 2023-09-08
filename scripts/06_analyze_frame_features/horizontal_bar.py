import os
import cv2
import json
import numpy as np
import plotly.graph_objects as go
from plotly import express

from collections import defaultdict

clip_id = "9df49083-577b-43f9-9874-6e4b21f104b4"

ground_truth_prediction_match_color_mapping = {
    True: "rgba(38, 24, 74, 0.8)",
    False: "rgba(190, 192, 213, 0.8)",
}

ground_truth_action_instances_file_path = f"{os.environ['CODE']}/scripts/07_reproduce_baseline_results/data/ego4d/ego4d_clip_annotations_v3.json"
asl_predicted_action_instances_file_path = (
    f"{os.environ['CODE']}/scripts/07_reproduce_baseline_results/submission_final.json"
)

ground_truth_action_instances = json.load(
    open(ground_truth_action_instances_file_path, "r")
)[clip_id]["annotations"]
asl_predicted_action_instances = json.load(
    open(asl_predicted_action_instances_file_path, "r")
)["detect_results"][clip_id]

cap = cv2.VideoCapture(
    os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips", clip_id + ".mp4")
)
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_id_ground_truth_action_instances_mapping = defaultdict(lambda: [])
frame_id_asl_predicted_action_instances_mapping = defaultdict(lambda: [])
unique_action_categories = set(["background"])

for current_frame_id in range(num_frames):
    current_frame_time = current_frame_id / fps
    assigned_to_an_action_category = False
    for asl_predicted_action_instance in asl_predicted_action_instances:
        if (
            current_frame_time >= asl_predicted_action_instance["segment"][0]
            and current_frame_time <= asl_predicted_action_instance["segment"][1]
        ):
            assigned_to_an_action_category = True
            frame_id_asl_predicted_action_instances_mapping[current_frame_id].append(
                asl_predicted_action_instance["label"]
            )
    if not assigned_to_an_action_category:
        frame_id_asl_predicted_action_instances_mapping[current_frame_id].append(
            "background"
        )

    assigned_to_an_action_category = False
    for ground_truth_action_instance in ground_truth_action_instances:
        if (
            current_frame_time >= ground_truth_action_instance["segment"][0]
            and current_frame_time <= ground_truth_action_instance["segment"][1]
        ):
            assigned_to_an_action_category = True
            frame_id_ground_truth_action_instances_mapping[current_frame_id].append(
                ground_truth_action_instance["label"]
            )
            unique_action_categories.add(ground_truth_action_instance["label"])
    if not assigned_to_an_action_category:
        frame_id_ground_truth_action_instances_mapping[current_frame_id].append(
            "background"
        )


def generate_random_color():
    random_int = np.random.randint(low=0, high=256, size=(3,))
    random_color = f"rgba({random_int[0]}, {random_int[1]}, {random_int[2]}, 0.8)"
    return random_color


action_category_color_mapping = dict(
    (action_category, generate_random_color())
    for action_category in sorted(list(unique_action_categories))
)

fig = go.Figure()

for current_frame_id in range(num_frames):
    current_ground_truth_action_instance = (
        frame_id_ground_truth_action_instances_mapping[current_frame_id][0]
    )
    current_color = action_category_color_mapping[current_ground_truth_action_instance]
    fig.add_trace(
        go.Bar(
            x=[1],
            y=
            orientation="h",
            marker=dict(
                color=current_color,
                line=dict(color="rgb(248, 248, 249)", width=1),
            ),
        )
    )

# fig.update_layout(
#     xaxis=dict(
#         showgrid=False,
#         showline=False,
#         showticklabels=False,
#         zeroline=False,
#         domain=[0.15, 1],
#     ),
#     yaxis=dict(
#         showgrid=False,
#         showline=False,
#         showticklabels=False,
#         zeroline=False,
#     ),
#     barmode="stack",
#     paper_bgcolor="rgb(248, 248, 255)",
#     plot_bgcolor="rgb(248, 248, 255)",
#     margin=dict(l=120, r=10, t=140, b=80),
#     showlegend=False,
# )

fig = go.Figure()

for i in range(len()):
    for xd, yd in zip(x_data, y_data):
        fig.add_trace(
            go.Bar(
                x=[xd[i]],
                y=[yd],
                orientation="h",
                marker=dict(
                    color=colors[i], line=dict(color="rgb(248, 248, 249)", width=1)
                ),
            )
        )

fig.update_layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1],
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
    ),
    barmode="stack",
    paper_bgcolor="rgb(248, 248, 255)",
    plot_bgcolor="rgb(248, 248, 255)",
    margin=dict(l=120, r=10, t=140, b=80),
    showlegend=False,
)

fig.show()

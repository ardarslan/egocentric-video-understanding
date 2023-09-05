import os
import json
import pandas as pd
import plotly.express as px


clip_ids_with_blip2_vqa_features = [
    clip_id
    for clip_id in os.listdir(
        os.path.join(os.environ["SCRATCH"], "/ego4d_data/v2/frame_features")
    )
    if os.path.exists(
        os.path.join(
            os.environ["SCRATCH"],
            "/ego4d_data/v2/frame_features",
            clip_id,
            "blip2_vqa_features.tsv",
        )
    )
]

asl_predictions_file_path = os.path.join(
    os.environ["CODE"], "scripts/07_reproduce_baseline_results", "submission_final.json"
)
ground_truths_file_path = os.path.join(
    os.environ["CODE"],
    "scripts/07_reproduce_baseline_results/data/ego4d",
    "ego4d_clip_annotations_v3.json",
)

with open(asl_predictions_file_path, "r") as asl_predictions_file_reader:
    asl_predictions = json.load(asl_predictions_file_reader)
    asl_predictions = dict(
        (clip_id, asl_predictions[clip_id])
        for clip_id in clip_ids_with_blip2_vqa_features
    )

with open(ground_truths_file_path, "r") as ground_truths_file_reader:
    ground_truths = json.load(ground_truths_file_reader)
    ground_truths = dict(
        (clip_id, ground_truths[clip_id])
        for clip_id in clip_ids_with_blip2_vqa_features
    )

print(asl_predictions.keys())
print(ground_truths.keys())


# fig = px.bar(df, x="frame_index", color="ground_truth_label", orientation='h',
#              hover_data=["tip", "size"],
#              height=400,
#              title='Restaurant bills')
# fig.show()


# # df = pd.DataFrame(
# #     {
# #         "OS": ["Android", "Windows", "iOS", "OS X", "Linux", "Other"],
# #         "Usage": [44.17, 28.96, 17.46, 5.56, 0.92, 1.92],
# #     }
# # )

# # fig = px.bar(df, x="OS", y="Usage")

# # fig.show()

# # fig = px.bar(df, x="total_bill", y="day", orientation="h")
# # fig.show()

# # # fig = px.bar(df, x="total_bill", y="day", orientation='h')
# # # fig.show()

# # # fig = go.Figure(
# # #     data=[
# # #         go.Scatter(
# # #             x=df["LOGP"],
# # #             y=df["PKA"],
# # #             mode="markers",
# # #             marker=dict(
# # #                 colorscale="viridis",
# # #                 color=df["MW"],
# # #                 size=df["MW"],
# # #                 colorbar={"title": "Molecular<br>Weight"},
# # #                 line={"color": "#444"},
# # #                 reversescale=True,
# # #                 sizeref=45,
# # #                 sizemode="diameter",
# # #                 opacity=0.8,
# # #             ),
# # #         )
# # #     ]
# # # )

# # # # turn off native plotly.js hover effects - make sure to use
# # # # hoverinfo="none" rather than "skip" which also halts events.
# # # fig.update_traces(hoverinfo="none", hovertemplate=None)

# # # fig.update_layout(
# # #     xaxis=dict(title="Log P"),
# # #     yaxis=dict(title="pkA"),
# # #     plot_bgcolor="rgba(255,255,255,0.1)",
# # # )

# # # app = Dash(__name__)

# # # app.layout = html.Div(
# # #     [
# # #         dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
# # #         dcc.Tooltip(id="graph-tooltip"),
# # #     ]
# # # )


# # # @callback(
# # #     Output("graph-tooltip", "show"),
# # #     Output("graph-tooltip", "bbox"),
# # #     Output("graph-tooltip", "children"),
# # #     Input("graph-basic-2", "hoverData"),
# # # )
# # # def display_hover(hoverData):
# # #     if hoverData is None:
# # #         return False, no_update, no_update

# # #     # demo only shows the first point, but other points may also be available
# # #     pt = hoverData["points"][0]
# # #     bbox = pt["bbox"]
# # #     num = pt["pointNumber"]

# # #     df_row = df.iloc[num]
# # #     img_src = df_row["IMG_URL"]
# # #     name = df_row["NAME"]
# # #     form = df_row["FORM"]
# # #     desc = df_row["DESC"]
# # #     if len(desc) > 300:
# # #         desc = desc[:100] + "..."

# # #     children = [
# # #         html.Div(
# # #             [
# # #                 html.Img(src=img_src, style={"width": "100%"}),
# # #                 html.H2(
# # #                     f"{name}",
# # #                     style={"color": "darkblue", "overflow-wrap": "break-word"},
# # #                 ),
# # #                 html.P(f"{form}"),
# # #                 html.P(f"{desc}"),
# # #             ],
# # #             style={"width": "200px", "white-space": "normal"},
# # #         )
# # #     ]

# # #     return True, bbox, children


# # # if __name__ == "__main__":
# # #     app.run(debug=True)

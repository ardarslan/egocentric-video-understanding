import os
import cv2
import numpy as np
import pandas as pd
from django.template import loader
from django.http import HttpResponse, JsonResponse
from rest_framework import viewsets
from PIL import Image
from tqdm import tqdm

import sys
from collections import defaultdict

sys.path.append("../")
import utils.globals


def index(request):
    template = loader.get_template("webserver/index.html")
    context = {}
    context["videos"] = [
        {"name": video_name.split(".")[0]}
        for video_name in sorted(
            os.listdir(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips"))
        )[:20]
        if video_name.split(".")[-1] == "mp4"
    ]

    for global_name in dir(utils.globals):
        if global_name.startswith("__"):
            continue
        context[global_name] = getattr(utils.globals, global_name)

    return HttpResponse(template.render(context, request))


class VideoDataReader(viewsets.ViewSet):
    video_id = None
    frame_width = None
    frame_height = None
    feature_name_feature_df_dict = {}
    blip_vqa_feature_name_text_location_mapping = {
        "blip_describe": (100, 100),
        "blip_do": (100, 200),
        "blip_happen": (100, 300),
    }
    blip_vqa_feature_name_question_mapping = {
        "blip_describe": "What does the image describe?",
        "blip_do": "What is the person in this picture doing?",
        "blip_happen": "What is happening in this picture?",
    }

    @classmethod
    def _display_visor_hos_feature_on_frame(
        cls,
        frame,
        frame_id,
        video_file_path,
    ):
        video_id = video_file_path.split("/")[-1][:-4]
        if cls.video_id != video_id:
            cls.feature_name_feature_df_dict = {}
        if "visor_hos" not in cls.feature_name_feature_df_dict.keys():
            cls.feature_name_feature_df_dict["visor_hos"] = pd.read_csv(
                os.path.join(
                    video_file_path.replace("clips", "frame_features")[:-4],
                    "visor_hos_features.tsv",
                ),
                sep="\t",
            )

        visor_hos_feature_df = cls.feature_name_feature_df_dict["visor_hos"]

        visor_hos_feature_rows = visor_hos_feature_df[
            visor_hos_feature_df["frame_index"] == int(frame_id)
        ]

        for _, row in visor_hos_feature_rows.iterrows():
            start_point = (
                int(row["x_top_left"] * cls.frame_width / 750.0),
                int(row["y_top_left"] * cls.frame_height / 1333.0),
            )
            end_point = (
                int(row["x_bottom_right"] * cls.frame_width / 750.0),
                int(row["y_bottom_right"] * cls.frame_height / 1333.0),
            )
            cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=1)

            cv2.putText(
                frame,
                "VisorHOS "
                + row["text_label"]
                + " ("
                + str(np.round(row["detection_score"], 2))
                + ")",
                (
                    int(row["x_top_left"] * cls.frame_width / 750.0),
                    max(int(row["y_top_left"] * cls.frame_height / 1333.0) - 10, 10),
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 255, 255),
                thickness=2,
            )
        return frame

    @classmethod
    def _display_unidet_feature_on_frame(
        cls,
        frame,
        frame_id,
        video_file_path,
    ):
        video_id = video_file_path.split("/")[-1][:-4]
        if cls.video_id != video_id:
            cls.feature_name_feature_df_dict = {}
        if "unidet" not in cls.feature_name_feature_df_dict.keys():
            cls.feature_name_feature_df_dict["unidet"] = pd.read_csv(
                os.path.join(
                    video_file_path.replace("clips", "frame_features")[:-4],
                    "unidet_features.tsv",
                ),
                sep="\t",
            )

        unidet_feature_df = cls.feature_name_feature_df_dict["unidet"]

        unidet_feature_rows = unidet_feature_df[
            unidet_feature_df["frame_index"] == int(frame_id)
        ]

        for _, row in unidet_feature_rows.iterrows():
            start_point = (
                int(row["x_top_left"] * cls.frame_width / 750.0),
                int(row["y_top_left"] * cls.frame_height / 1333.0),
            )
            end_point = (
                int(row["x_bottom_right"] * cls.frame_width / 750.0),
                int(row["y_bottom_right"] * cls.frame_height / 1333.0),
            )
            cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=1)

            cv2.putText(
                frame,
                "Unidet "
                + row["text_label"]
                + " ("
                + str(np.round(row["detection_score"], 2))
                + ")",
                (
                    int(row["x_top_left"] * cls.frame_width / 750.0),
                    max(int(row["y_top_left"] * cls.frame_height / 1333.0) - 10, 10),
                ),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=(255, 255, 255),
                thickness=2,
            )
        return frame

    @classmethod
    def get_bb_iou_and_intersection_coordinates(
        cls,
        x_top_left_0,
        x_bottom_right_0,
        y_top_left_0,
        y_bottom_right_0,
        x_top_left_1,
        x_bottom_right_1,
        y_top_left_1,
        y_bottom_right_1,
    ):
        # determine the (x, y)-coordinates of the intersection rectangle
        if x_top_left_0 > x_bottom_right_1 or x_top_left_1 > x_bottom_right_0:
            x_intersection = 0
            x_intersection_coordinates = (None, None)
        else:
            max_x_top_left = max(x_top_left_0, x_top_left_1)
            min_x_bottom_right = min(x_bottom_right_0, x_bottom_right_1)
            x_intersection = min_x_bottom_right - max_x_top_left
            x_intersection_coordinates = (max_x_top_left, min_x_bottom_right)

        if y_top_left_0 > y_bottom_right_1 or y_top_left_1 > y_bottom_right_0:
            y_intersection = 0
            y_intersection_coordinates = (None, None)
        else:
            max_y_top_left = max(y_top_left_0, y_top_left_1)
            min_y_bottom_right = min(y_bottom_right_0, y_bottom_right_1)
            y_intersection = min_y_bottom_right - max_y_top_left
            y_intersection_coordinates = (max_y_top_left, min_y_bottom_right)

        intersection_area = y_intersection * x_intersection
        if intersection_area == 0:
            return (0, *x_intersection_coordinates, *y_intersection_coordinates)
        else:
            union_area = (
                (y_bottom_right_0 - y_top_left_0) * (x_bottom_right_0 - x_top_left_0)
                + (y_bottom_right_1 - y_top_left_1) * (x_bottom_right_1 - x_top_left_1)
                - intersection_area
            )
            return (
                intersection_area / float(union_area),
                *x_intersection_coordinates,
                *y_intersection_coordinates,
            )

    @classmethod
    def _display_unidet_visor_hos_feature_on_frame(
        cls,
        frame,
        frame_id,
        video_file_path,
    ):
        video_id = video_file_path.split("/")[-1][:-4]
        if cls.video_id != video_id:
            cls.feature_name_feature_df_dict = {}
        if "unidet" not in cls.feature_name_feature_df_dict.keys():
            cls.feature_name_feature_df_dict["unidet"] = pd.read_csv(
                os.path.join(
                    video_file_path.replace("clips", "frame_features")[:-4],
                    "unidet_features.tsv",
                ),
                sep="\t",
            )
        if "visor_hos" not in cls.feature_name_feature_df_dict.keys():
            cls.feature_name_feature_df_dict["visor_hos"] = pd.read_csv(
                os.path.join(
                    video_file_path.replace("clips", "frame_features")[:-4],
                    "visor_hos_features.tsv",
                ),
                sep="\t",
            )

        unidet_feature_df = cls.feature_name_feature_df_dict["unidet"]

        unidet_feature_rows = unidet_feature_df[
            unidet_feature_df["frame_index"] == int(frame_id)
        ]

        visor_hos_feature_df = cls.feature_name_feature_df_dict["visor_hos"]

        visor_hos_feature_rows = visor_hos_feature_df[
            visor_hos_feature_df["frame_index"] == int(frame_id)
        ]

        unidet_visor_hos_feature_rows = pd.merge(
            unidet_feature_rows,
            visor_hos_feature_rows,
            how="inner",
            on="frame_index",
            suffixes=["_unidet", "_visor_hos"],
        )

        for index, row in unidet_visor_hos_feature_rows.iterrows():
            (
                bb_iou,
                intersection_x_top_left,
                intersection_x_bottom_right,
                intersection_y_top_left,
                intersection_y_bottom_right,
            ) = cls.get_bb_iou_and_intersection_coordinates(
                x_top_left_0=row["x_top_left_unidet"],
                x_bottom_right_0=row["x_bottom_right_unidet"],
                y_top_left_0=row["y_top_left_unidet"],
                y_bottom_right_0=row["y_bottom_right_unidet"],
                x_top_left_1=row["x_top_left_visor_hos"],
                x_bottom_right_1=row["x_bottom_right_visor_hos"],
                y_top_left_1=row["y_top_left_visor_hos"],
                y_bottom_right_1=row["y_bottom_right_visor_hos"],
            )
            if bb_iou > 0.75:
                start_point = (
                    int(intersection_x_top_left * cls.frame_width / 750.0),
                    int(intersection_y_top_left * cls.frame_height / 1333.0),
                )
                end_point = (
                    int(intersection_x_bottom_right * cls.frame_width / 750.0),
                    int(intersection_y_bottom_right * cls.frame_height / 1333.0),
                )
                cv2.rectangle(
                    frame, start_point, end_point, color=(0, 0, 0), thickness=1
                )

                cv2.putText(
                    frame,
                    "Unidet-VisorHOS "
                    + row["text_label_unidet"]
                    + " ("
                    + str(np.round(row["detection_score_unidet"], 2))
                    + ")",
                    (
                        int(intersection_x_top_left * cls.frame_width / 750.0),
                        max(
                            int(intersection_y_top_left * cls.frame_height / 1333.0)
                            - 10,
                            10,
                        ),
                    ),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75,
                    color=(255, 255, 255),
                    thickness=2,
                )

        return frame

    @classmethod
    def _display_features_on_frame(
        cls,
        feature_names,
        frame,
        frame_id,
        video_file_path,
    ):
        for feature_name in feature_names:
            if feature_name == "unidet":
                cls._display_unidet_feature_on_frame(
                    frame=frame,
                    frame_id=frame_id,
                    video_file_path=video_file_path,
                )
            elif feature_name == "visor_hos":
                cls._display_visor_hos_feature_on_frame(
                    frame=frame,
                    frame_id=frame_id,
                    video_file_path=video_file_path,
                )
            elif feature_name == "unidet_visor_hos":
                cls._display_unidet_visor_hos_feature_on_frame(
                    frame=frame,
                    frame_id=frame_id,
                    video_file_path=video_file_path,
                )
            else:
                raise Exception(f"{feature_name} is not a valid feature name.")
        return frame

    @classmethod
    def get_video_data(cls, request, video_id):
        video_file_path = os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/clips", f"{video_id}.mp4"
        )
        cap = cv2.VideoCapture(video_file_path)
        video_data = {}

        cls.frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cls.frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(cls.frame_width, cls.frame_height)

        video_data["virtual_fps"] = cap.get(cv2.CAP_PROP_FPS)
        frames_folder_path = os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/frames", video_id
        )
        os.makedirs(frames_folder_path, exist_ok=True)

        if not os.path.exists(os.path.join(frames_folder_path, "end.txt")):
            success = True
            video_data["num_real_frames"] = 0
            frame_id = 0
            while success:
                success, frame = cap.read()
                if success:
                    frame = frame[:, :, ::-1]
                    cv2.imwrite(
                        os.path.join(
                            frames_folder_path, str(frame_id).zfill(6) + ".png"
                        ),
                        frame,
                    )
                    frame_id += 1
                    video_data["num_real_frames"] += 1
                else:
                    break
            with open(os.path.join(frames_folder_path, "end.txt"), "w") as writer:
                writer.write("\n")
        else:
            video_data["num_real_frames"] = len(
                [
                    file_name
                    for file_name in os.listdir(os.path.join(frames_folder_path))
                    if file_name[-4:] == ".png"
                ]
            )

        cap.release()
        return JsonResponse(video_data)

    @classmethod
    def get_video_frame(cls, request, video_id, frame_id):
        video_file_path = os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/clips", f"{video_id}.mp4"
        )
        frames_folder_path = os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/frames", video_id
        )
        frame = cv2.imread(
            os.path.join(frames_folder_path, str(frame_id).zfill(6) + ".png")
        )
        feature_names = (
            [
                strp
                for s in request.GET["feature_names"].split(",")
                if len(strp := s.strip()) > 0
            ]
            if "feature_names" in request.GET
            else []
        )
        frame = cls._display_features_on_frame(
            feature_names=feature_names,
            frame=frame,
            frame_id=frame_id,
            video_file_path=video_file_path,
        )

        frame = Image.fromarray(frame)

        response = HttpResponse(content_type="image/jpeg")
        frame.save(response, "JPEG")
        return response

import os
import cv2
import json
import numpy as np
import pandas as pd

from django.template import loader
from django.http import HttpResponse, JsonResponse
from rest_framework import viewsets
from PIL import Image

# from detectron2.config import get_cfg
# from detectron2.projects import point_rend
# import detectron2.data.transforms as T

# sys.path.append("../../04_extract_frame_features")
# from unidet.unidet.config import add_unidet_config

# from typing import Tuple


# def setup_cfg(
#     feature_name, config_file_path, confidence_threshold, model_file_path, device
# ):
#     if feature_name == "visor_hos":
#         return visor_hos_setup_cfg(
#             config_file_path=config_file_path,
#             confidence_threshold=confidence_threshold,
#             model_file_path=model_file_path,
#             device=device,
#         )
#     elif feature_name == "unidet":
#         return unidet_setup_cfg(
#             config_file_path=config_file_path,
#             confidence_threshold=confidence_threshold,
#             model_file_path=model_file_path,
#             device=device,
#         )
#     else:
#         raise Exception(f"Not a valid feature_name {feature_name}.")

# def visor_hos_setup_cfg(
#     config_file_path,
#     confidence_threshold,
#     model_file_path,
#     device,
# ):
#     cfg = get_cfg()
#     point_rend.add_pointrend_config(cfg)
#     cfg.merge_from_file(config_file_path)
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
#     cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
#     cfg.MODEL.WEIGHTS = model_file_path
#     cfg.MODEL.DEVICE = device
#     return cfg


# def unidet_setup_cfg(
#     config_file_path,
#     confidence_threshold,
#     model_file_path,
#     device,
# ):
#     # load config from file and command-line arguments
#     cfg = get_cfg()
#     add_unidet_config(cfg)
#     cfg.merge_from_file(config_file_path)
#     cfg.MODEL.DEVICE = device
#     cfg.MODEL.WEIGHTS = model_file_path
#     # Set score_threshold for builtin models
#     cfg.MULTI_DATASET.UNIFIED_LABEL_FILE = os.path.join(
#         os.environ["CODE"],
#         "scripts/04_extract_frame_features/unidet/datasets/label_spaces",
#         cfg.MULTI_DATASET.UNIFIED_LABEL_FILE,
#     )
#     cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
#     cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
#     cfg.freeze()
#     return cfg


def index(request):
    with open(
        os.path.join(
            os.environ["CODE"],
            "scripts/08_reproduce_mq_experiments/data/ego4d",
            "ego4d_clip_annotations_v3.json",
        ),
        "r",
    ) as annotations_reader:
        annotations_dict = json.load(annotations_reader)

    template = loader.get_template("webserver/index.html")
    context = {}
    context["videos"] = [
        {"name": clip_file_name.split(".")[0]}
        for clip_file_name in sorted(
            os.listdir(os.path.join(os.environ["SCRATCH"], "ego4d_data/v2/clips"))
        )
        if clip_file_name.split(".")[-1] == "mp4"
        and os.path.exists(
            os.path.join(
                os.environ["SCRATCH"],
                "ego4d_data/v2/frames",
                clip_file_name.split(".")[0],
                "end.txt",
            )
        )
        and os.path.exists(
            os.path.join(
                os.environ["SCRATCH"],
                "ego4d_data/v2/frame_features",
                clip_file_name.split(".")[0],
                "blip2_vqa_features_000000.tsv",
            )
        )
        and (len(annotations_dict[clip_file_name.split(".")[0]]["annotations"]) > 0)
    ][:20]

    # for global_name in dir(utils.globals):
    #     if global_name.startswith("__"):
    #         continue
    #     context[global_name] = getattr(utils.globals, global_name)

    return HttpResponse(template.render(context, request))


class VideoDataReader(viewsets.ViewSet):
    clip_id = None
    frame_width = None
    frame_height = None
    # visor_hos_aug = None
    # unidet_aug = None
    # visor_hos_cfg = None
    # unidet_cfg = None
    ground_truths_dict = None
    asl_baseline_predictions_dict = None

    feature_name_feature_df_mapping = {}
    feature_name_clip_id_mapping = {}
    clip_id_fps_mapping = {}

    blip2_vqa_feature_name_question_mapping = {
        "blip2_do": "Question: What is the person in this picture doing? Answer:",
    }

    # @classmethod
    # def get_multiplicative_coordinate_correction_factors(
    #     cls,
    #     feature_name,
    #     input_frame_height: int,
    #     input_frame_width: int,
    #     config_file_path: str,
    #     confidence_threshold: str,
    #     model_file_path: str,
    # ) -> Tuple[int, int]:
    #     if (
    #         feature_name == "visor_hos"
    #         and cls.visor_hos_aug is None
    #         or feature_name == "unidet"
    #         and cls.unidet_aug is None
    #     ):
    #         cfg = setup_cfg(
    #             feature_name=feature_name,
    #             config_file_path=config_file_path,
    #             confidence_threshold=confidence_threshold,
    #             model_file_path=model_file_path,
    #             device="cpu",
    #         )
    #         aug = T.ResizeShortestEdge(
    #             [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
    #             cfg.INPUT.MAX_SIZE_TEST,
    #         )
    #         if feature_name == "visor_hos":
    #             cls.visor_hos_cfg = cfg
    #             cls.visor_hos_aug = aug
    #         elif feature_name == "unidet":
    #             cls.unidet_cfg = cfg
    #             cls.unidet_aug = aug
    #     elif feature_name == "visor_hos" and cls.visor_hos_aug is not None:
    #         cfg = cls.visor_hos_cfg
    #         aug = cls.visor_hos_aug
    #     elif feature_name == "unidet" and cls.unidet_aug is not None:
    #         cfg = cls.unidet_cfg
    #         aug = cls.unidet_aug

    #     output_frame_height, output_frame_width = aug.get_output_shape(
    #         oldh=input_frame_height,
    #         oldw=input_frame_width,
    #         short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
    #         max_size=cfg.INPUT.MAX_SIZE_TEST,
    #     )
    #     return (
    #         input_frame_height / float(output_frame_height),
    #         input_frame_width / float(output_frame_width),
    #     )

    # @classmethod
    # def _display_visor_hos_bounding_box_on_frame(
    #     cls,
    #     frame,
    #     frame_id,
    #     video_file_path,
    # ):
    #     clip_id = video_file_path.split("/")[-1][:-4]
    #     if (
    #         "visor_hos" not in cls.feature_name_clip_id_mapping.keys()
    #         or cls.feature_name_clip_id_mapping["visor_hos"] != clip_id
    #     ):
    #         cls.feature_name_feature_df_mapping["visor_hos"] = pd.read_csv(
    #             os.path.join(
    #                 video_file_path.replace("clips", "frame_features")[:-4],
    #                 "visor_hos_features.tsv",
    #             ),
    #             sep="\t",
    #         )
    #         cls.feature_name_clip_id_mapping["visor_hos"] = clip_id

    #     visor_hos_feature_df = cls.feature_name_feature_df_mapping["visor_hos"]

    #     visor_hos_feature_rows = visor_hos_feature_df[
    #         visor_hos_feature_df["frame_index"] == int(frame_id // utils.globals.FRAME_FEATURE_EXTRACTION_STRIDE * utils.globals.FRAME_FEATURE_EXTRACTION_STRIDE)
    #     ]

    #     visor_hos_multiplicative_coordinate_correction_factors = cls.get_multiplicative_coordinate_correction_factors(
    #         feature_name="visor_hos",
    #         input_frame_height=frame.shape[0],
    #         input_frame_width=frame.shape[1],
    #         config_file_path=f"{os.environ['CODE']}/scripts/04_extract_frame_features/visor_hos/configs/hos_pointrend_rcnn_R_50_FPN_1x.yaml",
    #         confidence_threshold=0.4,
    #         model_file_path=f"{os.environ['SCRATCH']}/mq_libs/visor_hos/model_final_hos.pth",
    #     )

    #     for _, row in visor_hos_feature_rows.iterrows():
    #         start_point = (
    #             int(
    #                 row["x_top_left"]
    #                 * visor_hos_multiplicative_coordinate_correction_factors[1]
    #             ),
    #             int(
    #                 row["y_top_left"]
    #                 * visor_hos_multiplicative_coordinate_correction_factors[0]
    #             ),
    #         )
    #         end_point = (
    #             int(
    #                 row["x_bottom_right"]
    #                 * visor_hos_multiplicative_coordinate_correction_factors[1]
    #             ),
    #             int(
    #                 row["y_bottom_right"]
    #                 * visor_hos_multiplicative_coordinate_correction_factors[0]
    #             ),
    #         )
    #         cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=4)

    #         cv2.putText(
    #             frame,
    #             "VisorHOS "
    #             + row["text_label"]
    #             + " ("
    #             + str(np.round(row["detection_score"], 2))
    #             + ")",
    #             (
    #                 int(
    #                     row["x_top_left"]
    #                     * visor_hos_multiplicative_coordinate_correction_factors[1]
    #                 ),
    #                 max(
    #                     int(
    #                         row["y_top_left"]
    #                         * visor_hos_multiplicative_coordinate_correction_factors[0]
    #                     )
    #                     - 10,
    #                     50,
    #                 ),
    #             ),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=1.0,
    #             color=(255, 255, 255),
    #             thickness=4,
    #         )

    #     return frame

    # @classmethod
    # def _display_unidet_bounding_box_on_frame(
    #     cls,
    #     frame,
    #     frame_id,
    #     video_file_path,
    # ):
    #     clip_id = video_file_path.split("/")[-1][:-4]
    #     if (
    #         "unidet" not in cls.feature_name_clip_id_mapping.keys()
    #         or cls.feature_name_clip_id_mapping["unidet"] != clip_id
    #     ):
    #         cls.feature_name_feature_df_mapping["unidet"] = pd.read_csv(
    #             os.path.join(
    #                 video_file_path.replace("clips", "frame_features")[:-4],
    #                 "unidet_features.tsv",
    #             ),
    #             sep="\t",
    #         )
    #         cls.feature_name_clip_id_mapping["unidet"] = clip_id

    #     unidet_feature_df = cls.feature_name_feature_df_mapping["unidet"]

    #     unidet_feature_rows = unidet_feature_df[
    #         unidet_feature_df["frame_index"] == int(frame_id // utils.globals.FRAME_FEATURE_EXTRACTION_STRIDE * utils.globals.FRAME_FEATURE_EXTRACTION_STRIDE)
    #     ]

    #     unidet_multiplicative_coordinate_correction_factors = cls.get_multiplicative_coordinate_correction_factors(
    #         feature_name="unidet",
    #         input_frame_height=frame.shape[0],
    #         input_frame_width=frame.shape[1],
    #         config_file_path=f"{os.environ['CODE']}/scripts/04_extract_frame_features/unidet/configs/Unified_learned_OCIM_R50_6x+2x.yaml",
    #         confidence_threshold=0.4,
    #         model_file_path=f"{os.environ['SCRATCH']}/mq_libs/unidet/Unified_learned_OCIM_R50_6x+2x.pth",
    #     )

    #     for _, row in unidet_feature_rows.iterrows():
    #         start_point = (
    #             int(
    #                 row["x_top_left"]
    #                 * unidet_multiplicative_coordinate_correction_factors[1]
    #             ),
    #             int(
    #                 row["y_top_left"]
    #                 * unidet_multiplicative_coordinate_correction_factors[0]
    #             ),
    #         )
    #         end_point = (
    #             int(
    #                 row["x_bottom_right"]
    #                 * unidet_multiplicative_coordinate_correction_factors[1]
    #             ),
    #             int(
    #                 row["y_bottom_right"]
    #                 * unidet_multiplicative_coordinate_correction_factors[0]
    #             ),
    #         )
    #         cv2.rectangle(frame, start_point, end_point, color=(0, 0, 0), thickness=4)

    #         cv2.putText(
    #             frame,
    #             "Unidet "
    #             + row["text_label"]
    #             + " ("
    #             + str(np.round(row["detection_score"], 2))
    #             + ")",
    #             (
    #                 int(
    #                     row["x_top_left"]
    #                     * unidet_multiplicative_coordinate_correction_factors[1]
    #                 ),
    #                 max(
    #                     int(
    #                         row["y_top_left"]
    #                         * unidet_multiplicative_coordinate_correction_factors[0]
    #                     )
    #                     - 10,
    #                     50,
    #                 ),
    #             ),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=1.0,
    #             color=(255, 255, 255),
    #             thickness=4,
    #         )

    #     return frame

    # @classmethod
    # def get_bb_iou_and_intersection_coordinates(
    #     cls,
    #     x_top_left_0,
    #     x_bottom_right_0,
    #     y_top_left_0,
    #     y_bottom_right_0,
    #     x_top_left_1,
    #     x_bottom_right_1,
    #     y_top_left_1,
    #     y_bottom_right_1,
    # ):
    #     if x_top_left_0 > x_bottom_right_1 or x_top_left_1 > x_bottom_right_0:
    #         x_intersection = 0
    #         x_intersection_coordinates = (None, None)
    #     else:
    #         max_x_top_left = max(x_top_left_0, x_top_left_1)
    #         min_x_bottom_right = min(x_bottom_right_0, x_bottom_right_1)
    #         x_intersection = min_x_bottom_right - max_x_top_left
    #         x_intersection_coordinates = (max_x_top_left, min_x_bottom_right)

    #     if y_top_left_0 > y_bottom_right_1 or y_top_left_1 > y_bottom_right_0:
    #         y_intersection = 0
    #         y_intersection_coordinates = (None, None)
    #     else:
    #         max_y_top_left = max(y_top_left_0, y_top_left_1)
    #         min_y_bottom_right = min(y_bottom_right_0, y_bottom_right_1)
    #         y_intersection = min_y_bottom_right - max_y_top_left
    #         y_intersection_coordinates = (max_y_top_left, min_y_bottom_right)

    #     intersection_area = y_intersection * x_intersection
    #     if intersection_area == 0:
    #         return (0, *x_intersection_coordinates, *y_intersection_coordinates)
    #     else:
    #         union_area = (
    #             (y_bottom_right_0 - y_top_left_0) * (x_bottom_right_0 - x_top_left_0)
    #             + (y_bottom_right_1 - y_top_left_1) * (x_bottom_right_1 - x_top_left_1)
    #             - intersection_area
    #         )
    #         return (
    #             intersection_area / float(union_area),
    #             *x_intersection_coordinates,
    #             *y_intersection_coordinates,
    #         )

    # @classmethod
    # def _display_unidet_visor_hos_bounding_box_on_frame(
    #     cls,
    #     frame,
    #     frame_id,
    #     video_file_path,
    # ):
    #     clip_id = video_file_path.split("/")[-1][:-4]
    #     if (
    #         "unidet" not in cls.feature_name_clip_id_mapping.keys()
    #         or cls.feature_name_clip_id_mapping["unidet"] != clip_id
    #     ):
    #         cls.feature_name_feature_df_mapping["unidet"] = pd.read_csv(
    #             os.path.join(
    #                 video_file_path.replace("clips", "frame_features")[:-4],
    #                 "unidet_features.tsv",
    #             ),
    #             sep="\t",
    #         )
    #         cls.feature_name_clip_id_mapping["unidet"] = clip_id
    #     if (
    #         "visor_hos" not in cls.feature_name_clip_id_mapping.keys()
    #         or cls.feature_name_clip_id_mapping["visor_hos"] != clip_id
    #     ):
    #         cls.feature_name_feature_df_mapping["visor_hos"] = pd.read_csv(
    #             os.path.join(
    #                 video_file_path.replace("clips", "frame_features")[:-4],
    #                 "visor_hos_features.tsv",
    #             ),
    #             sep="\t",
    #         )
    #         cls.feature_name_clip_id_mapping["visor_hos"] = clip_id

    #     unidet_feature_df = cls.feature_name_feature_df_mapping["unidet"]

    #     unidet_feature_rows = unidet_feature_df[
    #         unidet_feature_df["frame_index"] == int(frame_id // utils.globals.FRAME_FEATURE_EXTRACTION_STRIDE * utils.globals.FRAME_FEATURE_EXTRACTION_STRIDE)
    #     ]

    #     visor_hos_feature_df = cls.feature_name_feature_df_mapping["visor_hos"]

    #     visor_hos_feature_rows = visor_hos_feature_df[
    #         visor_hos_feature_df["frame_index"] == int(frame_id // utils.globals.FRAME_FEATURE_EXTRACTION_STRIDE * utils.globals.FRAME_FEATURE_EXTRACTION_STRIDE)
    #     ]

    #     unidet_visor_hos_feature_rows = pd.merge(
    #         unidet_feature_rows,
    #         visor_hos_feature_rows,
    #         how="inner",
    #         on="frame_index",
    #         suffixes=["_unidet", "_visor_hos"],
    #     )

    #     unidet_multiplicative_coordinate_correction_factors = cls.get_multiplicative_coordinate_correction_factors(
    #         feature_name="unidet",
    #         input_frame_height=frame.shape[0],
    #         input_frame_width=frame.shape[1],
    #         config_file_path=f"{os.environ['CODE']}/scripts/04_extract_frame_features/unidet/configs/Unified_learned_OCIM_R50_6x+2x.yaml",
    #         confidence_threshold=0.4,
    #         model_file_path=f"{os.environ['SCRATCH']}/mq_libs/unidet/Unified_learned_OCIM_R50_6x+2x.pth",
    #     )

    #     visor_hos_multiplicative_coordinate_correction_factors = cls.get_multiplicative_coordinate_correction_factors(
    #         feature_name="visor_hos",
    #         input_frame_height=frame.shape[0],
    #         input_frame_width=frame.shape[1],
    #         config_file_path=f"{os.environ['CODE']}/scripts/04_extract_frame_features/visor_hos/configs/hos_pointrend_rcnn_R_50_FPN_1x.yaml",
    #         confidence_threshold=0.4,
    #         model_file_path=f"{os.environ['SCRATCH']}/mq_libs/visor_hos/model_final_hos.pth",
    #     )

    #     for index, row in unidet_visor_hos_feature_rows.iterrows():
    #         (
    #             bb_iou,
    #             intersection_x_top_left,
    #             intersection_x_bottom_right,
    #             intersection_y_top_left,
    #             intersection_y_bottom_right,
    #         ) = cls.get_bb_iou_and_intersection_coordinates(
    #             x_top_left_0=row["x_top_left_unidet"]
    #             * unidet_multiplicative_coordinate_correction_factors[1],
    #             x_bottom_right_0=row["x_bottom_right_unidet"]
    #             * unidet_multiplicative_coordinate_correction_factors[1],
    #             y_top_left_0=row["y_top_left_unidet"]
    #             * unidet_multiplicative_coordinate_correction_factors[0],
    #             y_bottom_right_0=row["y_bottom_right_unidet"]
    #             * unidet_multiplicative_coordinate_correction_factors[0],
    #             x_top_left_1=row["x_top_left_visor_hos"]
    #             * visor_hos_multiplicative_coordinate_correction_factors[1],
    #             x_bottom_right_1=row["x_bottom_right_visor_hos"]
    #             * visor_hos_multiplicative_coordinate_correction_factors[1],
    #             y_top_left_1=row["y_top_left_visor_hos"]
    #             * visor_hos_multiplicative_coordinate_correction_factors[0],
    #             y_bottom_right_1=row["y_bottom_right_visor_hos"]
    #             * visor_hos_multiplicative_coordinate_correction_factors[0],
    #         )

    #         if bb_iou > 0.60:
    #             start_point = (
    #                 int(intersection_x_top_left),
    #                 int(intersection_y_top_left),
    #             )
    #             end_point = (
    #                 int(intersection_x_bottom_right),
    #                 int(intersection_y_bottom_right),
    #             )
    #             cv2.rectangle(
    #                 frame, start_point, end_point, color=(0, 0, 0), thickness=1
    #             )

    #             cv2.putText(
    #                 frame,
    #                 "Unidet-VisorHOS "
    #                 + row["text_label_unidet"]
    #                 + " ("
    #                 + str(np.round(row["detection_score_unidet"], 2))
    #                 + ")",
    #                 (
    #                     int(intersection_x_top_left),
    #                     max(
    #                         int(intersection_y_top_left) - 10,
    #                         50,
    #                     ),
    #                 ),
    #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #                 fontScale=1.0,
    #                 color=(255, 255, 255),
    #                 thickness=4,
    #             )
    #     return frame

    # @classmethod
    # def _display_bounding_boxes_on_frame(
    #     cls,
    #     bounding_boxes,
    #     frame,
    #     frame_id,
    #     video_file_path,
    # ):
    #     for bounding_box in bounding_boxes:
    #         if bounding_box == "unidet":
    #             cls._display_unidet_bounding_box_on_frame(
    #                 frame=frame,
    #                 frame_id=frame_id,
    #                 video_file_path=video_file_path,
    #             )
    #         elif bounding_box == "visor_hos":
    #             cls._display_visor_hos_bounding_box_on_frame(
    #                 frame=frame,
    #                 frame_id=frame_id,
    #                 video_file_path=video_file_path,
    #             )
    #         elif bounding_box == "unidet_visor_hos":
    #             cls._display_unidet_visor_hos_bounding_box_on_frame(
    #                 frame=frame,
    #                 frame_id=frame_id,
    #                 video_file_path=video_file_path,
    #             )
    #         else:
    #             raise Exception(f"{bounding_box} is not a valid bounding box.")
    #     return frame

    @classmethod
    def get_video_data(cls, request, clip_id):
        video_file_path = os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/clips", f"{clip_id}.mp4"
        )
        cap = cv2.VideoCapture(video_file_path)
        video_data = {}

        cls.frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cls.frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        video_data["virtual_fps"] = cap.get(cv2.CAP_PROP_FPS)
        cls.clip_id_fps_mapping[clip_id] = video_data["virtual_fps"]
        frames_folder_path = os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/frames", clip_id
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
                            frames_folder_path, str(frame_id).zfill(6) + ".jpg"
                        ),
                        frame,
                        [
                            int(cv2.IMWRITE_JPEG_QUALITY),
                            80,
                        ],
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
                    if file_name[-4:] == ".jpg"
                ]
            )

        cap.release()
        return JsonResponse(video_data)

    @classmethod
    def get_video_frame(cls, request, clip_id, frame_id):
        # video_file_path = os.path.join(
        #     os.environ["SCRATCH"], "ego4d_data/v2/clips", f"{clip_id}.mp4"
        # )
        frames_folder_path = os.path.join(
            os.environ["SCRATCH"], "ego4d_data/v2/frames", clip_id
        )
        frame = cv2.imread(
            os.path.join(frames_folder_path, str(frame_id).zfill(6) + ".jpg")
        )[:, :, ::-1]

        # bounding_boxes = (
        #     [
        #         strp
        #         for s in request.GET["bounding_box_feature_names"].split(",")
        #         if len(strp := s.strip()) > 0
        #     ]
        #     if "bounding_box_feature_names" in request.GET
        #     else []
        # )
        # frame = cls._display_bounding_boxes_on_frame(
        #     bounding_boxes=bounding_boxes,
        #     frame=frame,
        #     frame_id=frame_id,
        #     video_file_path=video_file_path,
        # )

        frame = Image.fromarray(frame)

        response = HttpResponse(content_type="image/jpeg")
        frame.save(response, "JPEG")

        return response

    @classmethod
    def get_blip2_answers(cls, request, clip_id, frame_id):
        existing_blip2_feature_file_path = os.path.join(
            os.environ["SCRATCH"],
            f"ego4d_data/v2/frame_features/{clip_id}",
            f"blip2_vqa_features_{str(frame_id // 600).zfill(6)}.tsv",
        )

        df = pd.read_csv(existing_blip2_feature_file_path, sep="\t")

        current_clip_frame_rows = df[df["frame_index"] == int((frame_id // 6) * 6)]

        blip2_answers = (
            [
                strp
                for s in request.GET["blip2_feature_names"].split(",")
                if len(strp := s.strip()) > 0
            ]
            if "blip2_feature_names" in request.GET
            else []
        )

        blip2_answers_dict = {}
        for blip2_answer in blip2_answers:
            current_answer = current_clip_frame_rows.loc[
                (
                    current_clip_frame_rows["question"]
                    == cls.blip2_vqa_feature_name_question_mapping[blip2_answer]
                ),
                "caption",
            ]
            if len(current_answer) == 0:
                blip2_answers_dict[blip2_answer] = "NaN"
            else:
                blip2_answers_dict[blip2_answer] = current_answer.values[0]

            return HttpResponse(json.dumps(blip2_answers_dict))

    @classmethod
    def get_action_categories(cls, request, clip_id, frame_id):
        action_category_names = (
            [
                strp
                for s in request.GET["action_category_names"].split(",")
                if len(strp := s.strip()) > 0
            ]
            if "action_category_names" in request.GET
            else []
        )

        if cls.ground_truths_dict is None:
            ground_truth_json_file_path = os.path.join(
                os.environ["CODE"],
                "scripts/08_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json",
            )
            with open(
                ground_truth_json_file_path, "r"
            ) as ground_truth_json_file_reader:
                cls.ground_truths_dict = json.load(ground_truth_json_file_reader)

        action_categories_dict = {}
        for action_category_name in action_category_names:
            if action_category_name == "ground_truth_action_category":
                try:
                    current_ground_truths_dict = cls.ground_truths_dict[clip_id]
                except KeyError:
                    action_categories_dict[
                        action_category_name
                    ] = "Annotations are not available for this clip."
                    continue
                if len(current_ground_truths_dict["annotations"]) == 0:
                    action_categories_dict[
                        action_category_name
                    ] = "Annotations are not available for this clip."
                    continue

                frame_is_in_an_action_instance = False
                for current_ground_truth in current_ground_truths_dict["annotations"]:
                    frame_time = frame_id / cls.clip_id_fps_mapping[clip_id]
                    if (
                        frame_time >= current_ground_truth["segment"][0]
                        and frame_time <= current_ground_truth["segment"][1]
                    ):
                        action_categories_dict[
                            action_category_name
                        ] = current_ground_truth["label"]
                        frame_is_in_an_action_instance = True
                        break
                if frame_is_in_an_action_instance:
                    continue
                else:
                    action_categories_dict[
                        action_category_name
                    ] = "This frame does not belong to any action instance."
            elif action_category_name == "asl_baseline_predicted_action_category":
                if cls.asl_baseline_predictions_dict is None:
                    asl_baseline_predictions_file_path = os.path.join(
                        os.environ["CODE"],
                        "scripts/08_reproduce_mq_experiments/asl_original_predictions.json",
                    )
                    with open(
                        asl_baseline_predictions_file_path, "r"
                    ) as asl_baseline_predictions_file_reader:
                        cls.asl_baseline_predictions_dict = json.load(
                            asl_baseline_predictions_file_reader
                        )["detect_results"]

                if cls.ground_truths_dict is None:
                    ground_truth_json_file_path = os.path.join(
                        os.environ["CODE"],
                        "scripts/08_reproduce_mq_experiments/data/ego4d/ego4d_clip_annotations_v3.json",
                    )
                    with open(
                        ground_truth_json_file_path, "r"
                    ) as ground_truth_json_file_reader:
                        cls.ground_truths_dict = json.load(
                            ground_truth_json_file_reader
                        )

                frame_time = frame_id / cls.clip_id_fps_mapping[clip_id]
                for prediction_dict in cls.asl_baseline_predictions_dict[clip_id]:
                    if (
                        frame_time >= prediction_dict["segment"][0]
                        and frame_time <= prediction_dict["segment"][-1]
                    ):
                        action_categories_dict[action_category_name] = (
                            prediction_dict["label"]
                            + " ("
                            + str(np.round(prediction_dict["score"], 2))
                            + ")"
                        )
                        frame_is_in_an_action_instance = True
                        break
                if frame_is_in_an_action_instance:
                    continue
                else:
                    action_categories_dict[
                        action_category_name
                    ] = "This frame does not belong to any action instance."

        return HttpResponse(json.dumps(action_categories_dict))

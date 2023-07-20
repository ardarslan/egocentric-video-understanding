import json
import torch
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from unidet.unidet.config import add_unidet_config

from typing import List, Tuple, Any


class UnidetFrameFeatureExtractor(object):
    def __init__(self, args):
        self.device = args.device
        cfg = self.setup_cfg(args)
        metadata = MetadataCatalog.get("__unused")
        unified_label_file = json.load(open(cfg.MULTI_DATASET.UNIFIED_LABEL_FILE))
        metadata.thing_classes = ["{}".format([xx for xx in x["name"].split("_") if xx != ""][0]) for x in unified_label_file["categories"]]
        self.class_id_text_label_mapping = metadata.get("thing_classes", None)
        self.unidet_predictor = DefaultPredictor(cfg=cfg)

        self.column_names = [
            "frame_index",
            "detection_index",
            "x_top_left",
            "y_top_left",
            "x_bottom_right",
            "y_bottom_right",
            "text_label",
            "detection_score",
        ]
        self.file_name_wo_ext = "unidet_features"

    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_unidet_config(cfg)
        cfg.merge_from_file(args.unidet_config_file)
        cfg.merge_from_list(args.unidet_opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.unidet_confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.unidet_confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.unidet_confidence_threshold
        cfg.freeze()
        return cfg

    def extract_frame_features(self, frame_index: int, frame: np.array) -> List[Tuple[Any, ...]]:
        """frame is in BGR format. The model expects BGR as well."""
        detections = self.unidet_predictor(frame)["instances"].to(torch.device("cpu"))
        features = []
        for current_detection_index in range(len(detections)):
            current_detection = detections[current_detection_index]
            detection_score = float(current_detection.scores.numpy()[0])
            predicted_class_id = int(current_detection.pred_classes.numpy()[0])
            text_label = self.class_id_text_label_mapping[predicted_class_id]
            predicted_box_coordinates = current_detection.pred_boxes.tensor.tolist()[0]
            x_top_left, y_top_left, x_bottom_right, y_bottom_right = (
                predicted_box_coordinates[0],
                predicted_box_coordinates[1],
                predicted_box_coordinates[2],
                predicted_box_coordinates[3],
            )
            features.append(
                (
                    frame_index,
                    current_detection_index,
                    x_top_left,
                    y_top_left,
                    x_bottom_right,
                    y_bottom_right,
                    text_label,
                    detection_score,
                )
            )
        return features
